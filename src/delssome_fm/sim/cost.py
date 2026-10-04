"""The DELSSOME FC+FCD cost between a simulated and an empirical summary.

As coded in the original DELSSOME (tzeng `DELSSOME_plus/scripts/utils/CBIG_pFIC_utils.py`,
lines 155-308; docs/data.md §7), not as SI S2 words it:

    corr = 1 - r, r the Pearson correlation of the *raw* upper-triangle FC entries
           (Fisher z is used only to average FC across runs and subjects, never here)
    d    = |mean(upper FC_sim) - mean(upper FC_emp)|   (difference of means, not MAE)
    KS   = max over bins |CDF_sim - CDF_emp|, both CDFs normalised to end at 1
    total = corr + d + KS

Pure jax.numpy, safe under jit and vmap. Averaging simulated FC and FCD over noise repeats
is the caller's job (reproduce.py), because the original averages FC arithmetically and
sums FCD histograms before taking the cost.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from delssome_fm.sim.summary import upper

Array = Any


class Cost(NamedTuple):
    corr: Array   # 1 - r
    mean: Array   # d
    ks: Array

    @property
    def total(self) -> Array:
        return self.corr + self.mean + self.ks


def fc_cost(fc_sim: Array, fc_emp: Array) -> tuple[Array, Array]:
    """fc_sim, fc_emp: (N, N) -> (1 - r, d) over the strict upper triangle."""
    rows, cols = upper(fc_emp.shape[0])
    s, e = fc_sim[rows, cols], fc_emp[rows, cols]
    sc, ec = s - jnp.mean(s), e - jnp.mean(e)
    r = jnp.sum(sc * ec) / jnp.sqrt(jnp.sum(sc ** 2) * jnp.sum(ec ** 2))
    return 1.0 - r, jnp.abs(jnp.mean(s) - jnp.mean(e))


def ks_cost(cdf_sim: Array, cdf_emp: Array) -> Array:
    """cdf_sim, cdf_emp: (B,) cumulative counts (any scale) -> max |difference| after
    normalising each to end at 1."""
    return jnp.max(jnp.abs(cdf_sim / cdf_sim[-1] - cdf_emp / cdf_emp[-1]))


def delssome_cost(fc_sim: Array, cdf_sim: Array, fc_emp: Array, cdf_emp: Array) -> Cost:
    corr, mean = fc_cost(fc_sim, fc_emp)
    return Cost(corr=corr, mean=mean, ks=ks_cost(cdf_sim, cdf_emp))
