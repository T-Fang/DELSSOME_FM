"""Stage-1 heads (architecture.md §5.2): predictions of the simulated summary statistics
from the backbone outputs. Targets are in the units of train/pack.py, z-scored where noted.

    regional    (B, N, 6)  MLP_reg(r_i), weights shared across regions          z-scored
    pairwise    (B, E)     z-scored arctanh FC at the requested edges, by one of three heads
                           (`pair_head`):
                bilinear       phi(r_i) . phi(r_j) / sqrt(k) [+ b(h_CLS)], a learned rank-k
                               factorisation of FC (architecture.md §5.2). The optional offset
                               b, one scalar per draw, lets it set the level of all edges at
                               once: a Gram matrix cannot make every edge slightly negative
                               (that needs rank ~N), the target of ~70% of draws (FC ~ 0)
                mlp_symmetric  MLP([r_i + r_j, r_i * r_j, |r_i - r_j|, h_CLS]): symmetric in
                               (i, j) by construction, no rank or sign limit (2026-10-19)
                mlp_concat     (MLP([r_i, r_j, h_CLS]) + MLP([r_j, r_i, h_CLS])) / 2: plain
                               concatenation, symmetrised by averaging both orders (2026-10-19)
                latent         decoder(MLP(h_CLS)): a latent vector of the FC autoencoder
                               (nn/fc_autoencoder.py), decoded by its frozen decoder into all
                               2278 edges (2026-10-25)
    fc_moments  (B, 3)     MLP(h_CLS)                                            z-scored
    fcd         (B, L)     monotone: cumulative sum of a softmax over L levels, so it is a
                           valid CDF (non-decreasing, ends at 1) by construction   raw CDF
    catch24     (B, N, 24) MLP_c24(r_i), weights shared across regions, if configured
                                                                                 z-scored
    fc_net      (B, 21)    MLP(h_CLS): arctanh FC between the Yeo-7 networks' averaged
                           signals (data/networks.py), if configured             z-scored

With the linearised network's statistics as head inputs (architecture.md §7.14; `analytic_fc`,
`analytic_fcd`), the prediction starts from them and the head learns the correction:

    pairwise    g(h_CLS) a_ij + MLP([r_i, r_j, h_CLS, W_a [a_ij, valid]]) (mlp_concat only):
                a_ij the analytic z-scored arctanh FC of the edge, 0 for a draw without one
    fcd         softmax(MLP(h_CLS) + MLP_a([A, valid]) + g'(h_CLS) log(dA + eps)): A the
                analytic CDF and dA its increments; the last term is dropped without one

The gates g, g' are 1 + a linear map of h_CLS that starts at zero, so training starts from
the analytic statistics. (A raw scalar among 3 d inputs starts with next to no weight, hence
W_a and the explicit term; CLAUDE.md, 2026-10-29.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from delssome_fm.nn.backbone import BackboneOutput
from delssome_fm.nn.layers import MLP

N_REGIONAL_STATS = 6
N_FC_MOMENTS = 3
PAIR_HEADS = ("bilinear", "mlp_symmetric", "mlp_concat", "latent")
N_CATCH24 = 24
N_NETWORK_PAIRS = 21
ANALYTIC_EMBED = 16         # width of W_a
PDF_EPS = 1e-4              # floor of the analytic FCD's increments before the log


@dataclass
class Stage1Prediction:
    regional: torch.Tensor
    pairwise: torch.Tensor
    fc_moments: torch.Tensor
    fcd: torch.Tensor
    catch24: torch.Tensor | None = None
    fc_net: torch.Tensor | None = None


class Stage1Heads(nn.Module):
    def __init__(self, d: int, rank: int, n_fcd_levels: int, fc_offset: bool,
                 pair_head: str = "bilinear", pair_hidden: int = 128, catch24: bool = False,
                 fc_decoder: nn.Module | None = None, latent_dim: int = 0,
                 n_regions: int = 68, fc_net: bool = False, analytic_fc: bool = False,
                 analytic_fcd: bool = False):
        """fc_decoder, latent_dim: the trained FC autoencoder's decoder and latent width, for
        pair_head "latent" only (kept frozen). analytic_fc, analytic_fcd: the heads start
        from the analytic FC / FCD (module docstring)."""
        super().__init__()
        if analytic_fc and pair_head != "mlp_concat":
            raise ValueError(f"the analytic FC enters the mlp_concat pair head, found {pair_head!r}")
        if analytic_fcd and not analytic_fc:
            raise ValueError("the analytic FCD is an option on top of the analytic FC")
        self.analytic_fc, self.analytic_fcd = analytic_fc, analytic_fcd
        if pair_head not in PAIR_HEADS:
            raise ValueError(f"pair_head must be one of {PAIR_HEADS}, found {pair_head!r}")
        if fc_offset and pair_head != "bilinear":
            raise ValueError("fc_offset belongs to the bilinear head; the MLP heads read h_CLS")
        if (pair_head == "latent") != (fc_decoder is not None):
            raise ValueError("the latent pair head needs an FC decoder, and only it takes one")
        self.rank, self.pair_head = rank, pair_head
        self.fc_offset = MLP([d, d, 1]) if fc_offset else None
        self.regional = MLP([d, d, N_REGIONAL_STATS])
        self.catch24 = MLP([d, d, N_CATCH24]) if catch24 else None
        self.fc_net = MLP([d, d, N_NETWORK_PAIRS]) if fc_net else None
        position = torch.zeros(n_regions, n_regions, dtype=torch.long)
        i, j = torch.triu_indices(n_regions, n_regions, offset=1)
        position[i, j] = position[j, i] = torch.arange(len(i))
        self.register_buffer("edge_position", position, persistent=False)
        if pair_head == "bilinear":
            self.phi = MLP([d, d, rank])
        elif pair_head == "latent":
            self.fc_latent = MLP([d, d, latent_dim])
            self.fc_decoder = fc_decoder.requires_grad_(False)
        else:
            width = 4 * d if pair_head == "mlp_symmetric" else 3 * d
            self.pair_mlp = MLP([width + ANALYTIC_EMBED * analytic_fc, pair_hidden, pair_hidden, 1])
        self.fc_moments = MLP([d, d, N_FC_MOMENTS])
        self.fcd_logits = MLP([d, d, n_fcd_levels])
        if analytic_fc:
            self.analytic_embed = nn.Linear(2, ANALYTIC_EMBED)
            self.fc_gate = _zero_linear(d)
        if analytic_fcd:
            self.fcd_analytic = MLP([n_fcd_levels + 1, d, n_fcd_levels])
            self.fcd_gate = _zero_linear(d)

    def forward(self, out: BackboneOutput, edges: torch.Tensor, analytic=None) -> Stage1Prediction:
        """edges: (E, 2) region index pairs (i, j) at which to predict the pairwise target;
        analytic: the batch's AnalyticBatch (train/data.py), required with analytic heads."""
        if self.analytic_fc != (analytic is not None):
            raise ValueError("analytic statistics are given exactly when the heads use them")
        return Stage1Prediction(regional=self.regional(out.regions),
                                pairwise=self.pairwise(out, edges, analytic),
                                fc_moments=self.fc_moments(out.h_cls),
                                fcd=self.fcd(out, analytic),
                                catch24=None if self.catch24 is None
                                else self.catch24(out.regions),
                                fc_net=None if self.fc_net is None else self.fc_net(out.h_cls))

    def fcd(self, out: BackboneOutput, analytic=None) -> torch.Tensor:
        """(B, L) the FCD CDF: a cumulative softmax, from the analytic CDF if configured."""
        logits = self.fcd_logits(out.h_cls)
        if self.analytic_fcd:
            valid = analytic.fcd_valid[:, None].to(logits.dtype)
            cdf = analytic.fcd.to(logits.dtype)
            steps = torch.diff(cdf, dim=1, prepend=torch.zeros_like(cdf[:, :1])).clamp_min(0.0)
            start = (1.0 + self.fcd_gate(out.h_cls)) * torch.log(steps + PDF_EPS)
            logits = logits + self.fcd_analytic(torch.cat([cdf, valid], dim=1)) + valid * start
        return torch.softmax(logits, -1).cumsum(-1)

    def pairwise(self, out: BackboneOutput, edges: torch.Tensor, analytic=None) -> torch.Tensor:
        """edges (E, 2) -> (B, E) the prediction of the configured pair head."""
        if self.pair_head == "bilinear":
            f = self.phi(out.regions)                                 # (B, N, k)
            fc = (f[:, edges[:, 0]] * f[:, edges[:, 1]]).sum(-1) / math.sqrt(self.rank)
            return fc if self.fc_offset is None else fc + self.fc_offset(out.h_cls)
        if self.pair_head == "latent":
            full = self.fc_decoder(self.fc_latent(out.h_cls))           # (B, 2278)
            return full[:, self.edge_position[edges[:, 0], edges[:, 1]]]
        ri, rj = out.regions[:, edges[:, 0]], out.regions[:, edges[:, 1]]   # (B, E, d)
        cls = out.h_cls[:, None].expand(-1, len(edges), -1)
        if self.pair_head == "mlp_symmetric":
            x = torch.cat([ri + rj, ri * rj, (ri - rj).abs(), cls], dim=-1)
            return self.pair_mlp(x).squeeze(-1)
        if not self.analytic_fc:
            both = [self.pair_mlp(torch.cat([a, b, cls], dim=-1)) for a, b in ((ri, rj), (rj, ri))]
            return (0.5 * (both[0] + both[1])).squeeze(-1)
        a_ij = analytic.pairwise[:, self.edge_position[edges[:, 0], edges[:, 1]]].to(ri.dtype)
        valid = analytic.fc_valid[:, None].to(ri.dtype)                     # (B, 1)
        emb = self.analytic_embed(torch.stack([a_ij, valid.expand_as(a_ij)], dim=-1))
        both = [self.pair_mlp(torch.cat([a, b, cls, emb], dim=-1)) for a, b in ((ri, rj), (rj, ri))]
        start = (1.0 + self.fc_gate(out.h_cls)) * a_ij * valid
        return (0.5 * (both[0] + both[1])).squeeze(-1) + start


def _zero_linear(d: int) -> nn.Linear:
    """d -> 1, starting at zero: a gate 1 + this starts at 1."""
    layer = nn.Linear(d, 1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer
