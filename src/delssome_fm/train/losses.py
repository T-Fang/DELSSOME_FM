"""Stage-1 loss (architecture.md §5.3): a mean squared error per target group, combined by
uncertainty weighting rather than hand-tuned weights,

    L = sum_g  L_g / (2 s_g^2) + log s_g,      s_g learned (as log s_g, for stability)

Every training row is a usable draw (train/pack.py), so no per-statistic masking is needed.
"""

from __future__ import annotations

import torch
from torch import nn

from delssome_fm.nn.heads import Stage1Prediction

GROUPS: tuple[str, ...] = ("regional", "pairwise", "fcd", "fc_moments")


def group_losses(pred: Stage1Prediction, targets: dict[str, torch.Tensor],
                 edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean squared error of each group. edge_index (E,): the upper-triangle edges that
    pred.pairwise was computed at, as positions in the targets' 2278 edges."""
    return {"regional": _mse(pred.regional, targets["regional"]),
            "pairwise": _mse(pred.pairwise, targets["pairwise"][:, edge_index]),
            "fcd": _mse(pred.fcd, targets["fcd"]),
            "fc_moments": _mse(pred.fc_moments, targets["fc_moments"])}


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred.float() - target.float()) ** 2)


class UncertaintyWeighting(nn.Module):
    """Learned per-group scales s_g (Kendall et al. 2018)."""

    def __init__(self, groups: tuple[str, ...] = GROUPS):
        super().__init__()
        self.groups = groups
        self.log_s = nn.Parameter(torch.zeros(len(groups)))

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = [0.5 * torch.exp(-2 * log_s) * losses[g] + log_s
                 for g, log_s in zip(self.groups, self.log_s)]
        return torch.stack(terms).sum()

    def scales(self) -> dict[str, float]:
        """The current s_g, for logging."""
        return {g: float(torch.exp(log_s.detach())) for g, log_s in zip(self.groups, self.log_s)}
