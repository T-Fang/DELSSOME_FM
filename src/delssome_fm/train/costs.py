"""The original DELSSOME cost terms (docs/data.md §7) between two sets of summary statistics,
row by row, in torch so they run on the GPU during validation and in numpy-fed evaluation:

    corr  1 - r: one minus the Pearson correlation of the two raw-FC upper triangles
    mean  d = |mean of one upper triangle - mean of the other|
    ks    max |CDF difference| over the stored FCD levels (100; within 0.0004 of the
          10,000-bin KS, data.md §8)

Stage-1 targets are z-scored arctanh FC, so `raw_fc` maps them (and predictions) back first.
"""

from __future__ import annotations

import torch

TERMS = ("corr", "mean", "ks")


def raw_fc(z: torch.Tensor, mean: float, sd: float) -> torch.Tensor:
    """z-scored arctanh FC -> raw FC r."""
    return torch.tanh(z.float() * sd + mean)


def cost_terms(fc_a: torch.Tensor, fc_b: torch.Tensor, cdf_a: torch.Tensor,
               cdf_b: torch.Tensor) -> dict[str, torch.Tensor]:
    """fc (R, E) raw FC upper triangles, cdf (R, L) CDFs -> each term, (R,)."""
    a = fc_a - fc_a.mean(1, keepdim=True)
    b = fc_b - fc_b.mean(1, keepdim=True)
    r = (a * b).sum(1) / torch.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return {"corr": 1 - r, "mean": (fc_a.mean(1) - fc_b.mean(1)).abs(),
            "ks": (cdf_a - cdf_b).abs().amax(1)}
