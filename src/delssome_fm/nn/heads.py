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
    fc_moments  (B, 3)     MLP(h_CLS)                                            z-scored
    fcd         (B, L)     monotone: cumulative sum of a softmax over L levels, so it is a
                           valid CDF (non-decreasing, ends at 1) by construction   raw CDF
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
PAIR_HEADS = ("bilinear", "mlp_symmetric", "mlp_concat")


@dataclass
class Stage1Prediction:
    regional: torch.Tensor
    pairwise: torch.Tensor
    fc_moments: torch.Tensor
    fcd: torch.Tensor


class Stage1Heads(nn.Module):
    def __init__(self, d: int, rank: int, n_fcd_levels: int, fc_offset: bool,
                 pair_head: str = "bilinear", pair_hidden: int = 128):
        super().__init__()
        if pair_head not in PAIR_HEADS:
            raise ValueError(f"pair_head must be one of {PAIR_HEADS}, found {pair_head!r}")
        if fc_offset and pair_head != "bilinear":
            raise ValueError("fc_offset belongs to the bilinear head; the MLP heads read h_CLS")
        self.rank, self.pair_head = rank, pair_head
        self.fc_offset = MLP([d, d, 1]) if fc_offset else None
        self.regional = MLP([d, d, N_REGIONAL_STATS])
        if pair_head == "bilinear":
            self.phi = MLP([d, d, rank])
        else:
            width = 4 * d if pair_head == "mlp_symmetric" else 3 * d
            self.pair_mlp = MLP([width, pair_hidden, pair_hidden, 1])
        self.fc_moments = MLP([d, d, N_FC_MOMENTS])
        self.fcd_logits = MLP([d, d, n_fcd_levels])

    def forward(self, out: BackboneOutput, edges: torch.Tensor) -> Stage1Prediction:
        """edges: (E, 2) region index pairs (i, j) at which to predict the pairwise target."""
        return Stage1Prediction(regional=self.regional(out.regions),
                                pairwise=self.pairwise(out, edges),
                                fc_moments=self.fc_moments(out.h_cls),
                                fcd=torch.softmax(self.fcd_logits(out.h_cls), -1).cumsum(-1))

    def pairwise(self, out: BackboneOutput, edges: torch.Tensor) -> torch.Tensor:
        """edges (E, 2) -> (B, E) the prediction of the configured pair head."""
        if self.pair_head == "bilinear":
            f = self.phi(out.regions)                                 # (B, N, k)
            fc = (f[:, edges[:, 0]] * f[:, edges[:, 1]]).sum(-1) / math.sqrt(self.rank)
            return fc if self.fc_offset is None else fc + self.fc_offset(out.h_cls)
        ri, rj = out.regions[:, edges[:, 0]], out.regions[:, edges[:, 1]]   # (B, E, d)
        cls = out.h_cls[:, None].expand(-1, len(edges), -1)
        if self.pair_head == "mlp_symmetric":
            x = torch.cat([ri + rj, ri * rj, (ri - rj).abs(), cls], dim=-1)
            return self.pair_mlp(x).squeeze(-1)
        both = [self.pair_mlp(torch.cat([a, b, cls], dim=-1)) for a, b in ((ri, rj), (rj, ri))]
        return (0.5 * (both[0] + both[1])).squeeze(-1)
