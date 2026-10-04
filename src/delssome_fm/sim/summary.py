"""Summary statistics of an observed signal: one pipeline for simulated and empirical data.

`summarize` is the single entry point that stage 1 and stage 2 both call (brief §7.3), so
the TR, the FCD window and stride, and every transform are the same everywhere. All
functions are pure jax.numpy, run on the GPU, and are safe under jit and vmap. Batches are
reduced to summaries on the device; raw signals never need to leave it.

Definitions match the empirical pipeline that produced the HCP-YA reference data
(docs/data.md §4.2; tested against stored run-level FC and FCD CDFs):

    FC        Pearson correlation between regions over all frames
    FCD       windows of `fcd_window` frames at stride 1; Pearson FC per window; the
              correlation between the windows' upper-triangle FC vectors (no arctanh);
              the upper triangle of that matrix, histogrammed into `fcd_bins` equal bins on
              [-1, 1] (last bin closed), cumulatively summed: counts, not normalised

and the stage-1 statistics of architecture.md §5.1:

    regional (N, 6)   mean, SD, skewness, excess kurtosis, FC node strength
                      sum_{j != i} FC_ij / (N - 1), lag-1 autocorrelation
    pairwise          arctanh(FC_ij) over the upper triangle
    fc_moments (3,)   mean, SD, skewness of the upper-triangle FC

Not decided here (docs/data.md §8): reducing the 10,000-bin FCD CDF to the 100 levels of
architecture.md. The cost functions of build step 6 also live elsewhere.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from delssome_fm.config import SimConfig

Array = Any


class Summary(NamedTuple):
    """
    fc:         (N, N)
    fcd_cdf:    (B,)        cumulative counts, B = fcd_bins
    regional:   (N, 6)
    pairwise:   (N(N-1)/2,) arctanh of the upper-triangle FC
    fc_moments: (3,)
    """

    fc: Array
    fcd_cdf: Array
    regional: Array
    pairwise: Array
    fc_moments: Array


def upper(n: int) -> tuple[Array, Array]:
    """Row and column indices of the strict upper triangle of an n x n matrix."""
    return jnp.triu_indices(n, k=1)


def _standardise(x: Array, axis: int) -> Array:
    centred = x - jnp.mean(x, axis=axis, keepdims=True)
    return centred / jnp.sqrt(jnp.sum(centred ** 2, axis=axis, keepdims=True))


def functional_connectivity(x: Array) -> Array:
    """x: (T, N) -> (N, N) Pearson correlation between columns."""
    z = _standardise(x, axis=0)
    return z.T @ z


def window_fc_vectors(x: Array, window: int) -> Array:
    """x: (T, N) -> (T - window + 1, N(N-1)/2) upper-triangle FC of each sliding window."""
    T, N = x.shape
    if T < window:
        raise ValueError(f"signal has {T} frames, fewer than the FCD window {window}")
    starts = jnp.arange(T - window + 1)
    windows = jax.vmap(lambda s: jax.lax.dynamic_slice(x, (s, 0), (window, N)))(starts)
    z = _standardise(windows, axis=1)
    fcs = jnp.einsum("wtn,wtm->wnm", z, z)
    rows, cols = upper(N)
    return fcs[:, rows, cols]


def fcd_cdf(x: Array, window: int, bins: int) -> Array:
    """x: (T, N) -> (bins,) cumulative histogram counts of the upper-triangle FCD values."""
    vecs = window_fc_vectors(x, window)
    z = _standardise(vecs, axis=1)
    fcd = z @ z.T
    rows, cols = upper(vecs.shape[0])
    counts, _ = jnp.histogram(fcd[rows, cols], bins=bins, range=(-1.0, 1.0))
    return jnp.cumsum(counts)


def regional_statistics(x: Array, fc: Array) -> Array:
    """x: (T, N), fc: (N, N) -> (N, 6), columns as in the module docstring."""
    mean = jnp.mean(x, axis=0)
    centred = x - mean
    var = jnp.mean(centred ** 2, axis=0)
    sd = jnp.sqrt(var)
    skew = jnp.mean(centred ** 3, axis=0) / var ** 1.5
    kurt = jnp.mean(centred ** 4, axis=0) / var ** 2 - 3.0
    N = fc.shape[0]
    strength = (jnp.sum(fc, axis=1) - jnp.diagonal(fc)) / (N - 1)
    a, b = centred[:-1] - jnp.mean(centred[:-1], 0), centred[1:] - jnp.mean(centred[1:], 0)
    lag1 = jnp.sum(a * b, 0) / jnp.sqrt(jnp.sum(a ** 2, 0) * jnp.sum(b ** 2, 0))
    return jnp.stack([mean, sd, skew, kurt, strength, lag1], axis=-1)


def fc_moments(fc: Array) -> Array:
    """fc: (N, N) -> (3,) mean, SD and skewness of the upper-triangle entries."""
    rows, cols = upper(fc.shape[0])
    v = fc[rows, cols]
    c = v - jnp.mean(v)
    var = jnp.mean(c ** 2)
    return jnp.stack([jnp.mean(v), jnp.sqrt(var), jnp.mean(c ** 3) / var ** 1.5])


def summarize(x: Array, cfg: SimConfig) -> Summary:
    """x: (T, N) observed signal at TR -> Summary. The one function both stages use."""
    fc = functional_connectivity(x)
    rows, cols = upper(fc.shape[0])
    return Summary(fc=fc,
                   fcd_cdf=fcd_cdf(x, cfg.fcd_window, cfg.fcd_bins),
                   regional=regional_statistics(x, fc),
                   pairwise=jnp.arctanh(fc[rows, cols]),
                   fc_moments=fc_moments(fc))
