"""Stage-1 loss (architecture.md §5.3): a mean squared error per target group, combined by
uncertainty weighting rather than hand-tuned weights,

    L = sum_g  L_g / (2 s_g^2) + log s_g,      s_g learned (as log s_g, for stability)

With cost_loss (decided 2026-10-19), the DELSSOME cost terms between prediction and
simulation (1 - r, d, KS in raw units over all edges, train/costs.py) join as three more
groups, COST_GROUPS, each a mean over the batch's rows.

Which groups train (2026-10-25): GROUPS with stat_loss, "catch24" with catch24_loss,
"fc_net" (Yeo-7 network FC, 2026-10-26) with fc_net_loss, the cost groups with cost_loss
(`loss_groups`). Validation and evaluation report every group the
model can predict, trained or not.

Every training row is a usable draw (train/pack.py), so no per-statistic masking is needed.
"""

from __future__ import annotations

import torch
from torch import nn

from delssome_fm.nn.heads import Stage1Prediction

GROUPS: tuple[str, ...] = ("regional", "pairwise", "fcd", "fc_moments")
COST_GROUPS: tuple[str, ...] = ("cost_corr", "cost_mean", "cost_ks")


OPTIONAL_GROUPS: tuple[str, ...] = ("catch24", "fc_net")   # each with its own head


def loss_groups(stat_loss: bool, catch24_loss: bool, cost_loss: bool,
                fc_net_loss: bool = False) -> tuple[str, ...]:
    """The groups a configuration trains on, in the order of their learned scales."""
    if not (stat_loss or catch24_loss or fc_net_loss):
        raise ValueError("nothing to train: set stat_loss, catch24_loss and/or fc_net_loss")
    if cost_loss and not stat_loss:
        raise ValueError("cost_loss compares predicted FC and FCD: it needs stat_loss")
    return ((GROUPS if stat_loss else ()) + (("catch24",) if catch24_loss else ())
            + (("fc_net",) if fc_net_loss else ()) + (COST_GROUPS if cost_loss else ()))


def group_losses(pred: Stage1Prediction, targets: dict[str, torch.Tensor],
                 edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean squared error of each group the prediction holds (catch24 when it has a catch24
    head). edge_index (E,): the upper-triangle edges that pred.pairwise was computed at, as
    positions in the targets' 2278 edges."""
    out = {"regional": _mse(pred.regional, targets["regional"]),
           "pairwise": _mse(pred.pairwise, targets["pairwise"][:, edge_index]),
           "fcd": _mse(pred.fcd, targets["fcd"]),
           "fc_moments": _mse(pred.fc_moments, targets["fc_moments"])}
    for g in OPTIONAL_GROUPS:
        if getattr(pred, g) is not None:
            out[g] = _mse(getattr(pred, g), targets[g])
    return out


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
