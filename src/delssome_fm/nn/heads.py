"""Stage-1 heads (architecture.md §5.2): predictions of the simulated summary statistics
from the backbone outputs. Targets are in the units of train/pack.py, z-scored where noted.

    regional    (B, N, 6)  MLP_reg(r_i), weights shared across regions          z-scored
    pairwise    (B, E)     phi(r_i) . phi(r_j) / sqrt(k) [+ b(h_CLS)] for the requested
                           edges; z-scored arctanh FC; a learned rank-k factorisation of FC.
                           The optional offset b, one scalar per draw, lets the head set the
                           level of all edges at once: a Gram matrix phi phi^T cannot make
                           every edge slightly negative (that needs rank ~N), which is the
                           target of the ~70% of draws with FC ~ 0 (decided 2026-10-16)
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


@dataclass
class Stage1Prediction:
    regional: torch.Tensor
    pairwise: torch.Tensor
    fc_moments: torch.Tensor
    fcd: torch.Tensor


class Stage1Heads(nn.Module):
    def __init__(self, d: int, rank: int, n_fcd_levels: int, fc_offset: bool):
        super().__init__()
        self.rank = rank
        self.fc_offset = MLP([d, d, 1]) if fc_offset else None
        self.regional = MLP([d, d, N_REGIONAL_STATS])
        self.phi = MLP([d, d, rank])
        self.fc_moments = MLP([d, d, N_FC_MOMENTS])
        self.fcd_logits = MLP([d, d, n_fcd_levels])

    def forward(self, out: BackboneOutput, edges: torch.Tensor) -> Stage1Prediction:
        """edges: (E, 2) region index pairs (i, j) at which to predict the pairwise target."""
        return Stage1Prediction(regional=self.regional(out.regions),
                                pairwise=self.pairwise(out, edges),
                                fc_moments=self.fc_moments(out.h_cls),
                                fcd=torch.softmax(self.fcd_logits(out.h_cls), -1).cumsum(-1))

    def pairwise(self, out: BackboneOutput, edges: torch.Tensor) -> torch.Tensor:
        """edges (E, 2) -> (B, E) the bilinear prediction, plus the offset if enabled."""
        f = self.phi(out.regions)                                     # (B, N, k)
        fc = (f[:, edges[:, 0]] * f[:, edges[:, 1]]).sum(-1) / math.sqrt(self.rank)
        return fc if self.fc_offset is None else fc + self.fc_offset(out.h_cls)
