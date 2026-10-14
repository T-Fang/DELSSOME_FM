"""Region tokens (architecture.md §3): one d-vector per region from its parameters, its row
of structural connectivity, and a global offset g shared by all regions.

    t_{i,p}  = W_reg * theta[i, p] + kappa_p                   one scalar per parameter
    t_par_i  = MLP_theta(mean_p t_{i,p})                        (d/2)
    t_SC_i   = MLP_SC(C[i, :])                                  (d/2)
    s_k      = W_glob * psi[k] + lambda_k
    g        = MLP_g(mean_k s_k) + z + u + o + m_agg            (d)
    token_i  = [t_par_i || t_SC_i] + g

Embedding each parameter as a scalar tagged by kappa_p (where it sits in the equations) and
averaging over p is what makes the network free in P and K: never a P-wide input layer.
theta and psi are given as log10 of the multipliers (train/data.py), so the sampling range
10^U(-1, 1) becomes [-1, 1].
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
