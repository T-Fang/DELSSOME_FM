"""The backbone's input tokens, by token scheme (Stage1Config.token_scheme).

`RegionTokens`, scheme "region" (architecture.md §3): one d-vector per region from its
parameters, its row of structural connectivity, and a global offset g shared by all regions.

    t_{i,p}  = W_reg * theta[i, p] + kappa_p                   one scalar per parameter
    t_par_i  = MLP_theta(mean_p t_{i,p})                        (d/2)
    t_SC_i   = MLP_SC(C[i, :])                                  (d/2)
    s_k      = W_glob * psi[k] + lambda_k
    g        = MLP_g(mean_k s_k) + z + u + o + m_agg            (d)
    token_i  = [t_par_i || t_SC_i] + g

Embedding each parameter as a scalar tagged by kappa_p (where it sits in the equations) and
averaging over p is what makes the network free in P and K: never a P-wide input layer.
theta and psi arrive encoded to about [-1, 1] (train/data.encode_parameters): log10 of the
multipliers for the multiplier-policy corpora, asinh(x / 1e-4) / 19.1 of the actual values
for the sinh-policy corpus.

`ParameterTokens`, scheme "parameter" (architecture.md §3.4, decided 2026-10-29): one token
per regional parameter per region, P x N in all, with no pooling and no global offset,

    token_{i,p} = MLP_tok([theta[i, p] || kappa_p || C[i, :]])              (1 + d + N -> d)
    token_{i,p} = MLP_tok([W_theta theta[i, p] || kappa_p || C[i, :]])      (2d + N -> d)

the second with `embed_theta` (W_theta: d x 1); one MLP shared by all parameters and regions.
Global parameters and constants are values in the template encoder instead
(nn/template_encoder.py), and reach the tokens by attention. Both forms give the same
functions (the MLP's first layer is linear in theta either way); they differ in how strongly
theta acts at initialisation: as one raw input among 1 + d + N it starts with ~1/5 of the
weight it has through W_theta. Both are trained (2026-10-29) to see whether that matters
beyond the first steps.
"""

from __future__ import annotations

import torch
from torch import nn

from delssome_fm.nn.layers import MLP, masked_mean
from delssome_fm.nn.template_encoder import Readouts


class RegionTokens(nn.Module):
    def __init__(self, d: int, n_regions: int):
        super().__init__()
        if d % 2:
            raise ValueError(f"width d must be even (two halves), found {d}")
        self.w_reg = nn.Linear(1, d, bias=False)
        self.mlp_theta = MLP([d, d, d // 2])
        self.mlp_sc = MLP([n_regions, d, d // 2])
        self.w_glob = nn.Linear(1, d, bias=False)
        self.mlp_g = MLP([d, d, d])

    def forward(self, theta: torch.Tensor, psi: torch.Tensor, sc: torch.Tensor,
                r: Readouts) -> torch.Tensor:
        """
        theta: (B, N, P)  regional parameters (0 where the model has no p-th parameter)
        psi:   (B, K)     global parameters (0 where absent)
        sc:    (B, N, N)  structural connectivity, scaled to max 1
        returns: (B, N, d) one token per region
        """
        t = self.w_reg(theta[..., None]) + r.kappa[:, None]            # (B, N, P, d)
        t_par = self.mlp_theta(masked_mean(t, r.kappa_mask[:, None, :, None], 2))
        t_sc = self.mlp_sc(sc)
        s = self.w_glob(psi[..., None]) + r.lam                        # (B, K, d)
        g = self.mlp_g(masked_mean(s, r.lam_mask[..., None], 1)) + r.z + r.u + r.o + r.m_agg
        return torch.cat([t_par, t_sc], dim=-1) + g[:, None, :]


class ParameterTokens(nn.Module):
    def __init__(self, d: int, n_regions: int, embed_theta: bool):
        super().__init__()
        self.w_theta = nn.Linear(1, d, bias=False) if embed_theta else None
        self.mlp = MLP([(d if embed_theta else 1) + d + n_regions, d, d])

    def forward(self, theta: torch.Tensor, sc: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """
        theta: (B, N, P)  regional parameters (0 where the model has no p-th parameter)
        sc:    (B, N, N)  structural connectivity, scaled to max 1
        kappa: (B, P, d)  the regional parameters' Par-node embeddings (Readouts.kappa)
        returns: (B, N, P, d) one token per region and parameter; those of absent parameters
                 are to be masked (Readouts.kappa_mask)
        """
        B, N, P = theta.shape
        t = theta[..., None] if self.w_theta is None else self.w_theta(theta[..., None])
        x = torch.cat([t, kappa[:, None].expand(B, N, P, -1),
                       sc[:, :, None].expand(B, N, P, N)], dim=-1)
        return self.mlp(x)
