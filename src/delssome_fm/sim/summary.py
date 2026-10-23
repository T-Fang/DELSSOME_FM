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
    pairwise          stable_arctanh(FC_ij) over the upper triangle
    fc_moments (3,)   mean, SD, skewness of the upper-triangle FC

The FCD CDF is computed at full resolution (needed for the original cost's KS). The stage-1
corpus stores it reduced by `fcd_levels`: the CDF at every k-th bin's right edge. With
k = 100 that is 100 levels at FCD values -0.98, -0.96, ..., 1.00 (decided 2026-10-05).
Stage 2 must reduce empirical CDFs with the same function. The cost functions of build step 6
live in cost.py.

`catch24` (added 2026-10-23) is the exception to "pure jax.numpy": the 24 catch24 features
(the 22 of catch22, chosen from hctsa, plus mean and SD) of each region's signal, from the
pycatch22 C library on the host, about 0.1 s per (1200, 68) signal. Stage 2 must compute the
empirical data's features with the same function.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pycatch22

from delssome_fm.config import SimConfig

Array = Any


class Summary(NamedTuple):
    """
    fc:         (N, N)
    fcd_cdf:    (B,)        cumulative counts, B = fcd_bins
    regional:   (N, 6)
    pairwise:   (N(N-1)/2,) stable_arctanh of the upper-triangle FC
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


def fcd_matrix(x: Array, window: int) -> Array:
    """x: (T, N) -> (W, W) FCD matrix, W = T - window + 1: the Pearson correlation between
    the windows' upper-triangle FC vectors."""
    z = _standardise(window_fc_vectors(x, window), axis=1)
    return z @ z.T


def fcd_cdf(x: Array, window: int, bins: int) -> Array:
    """x: (T, N) -> (bins,) cumulative histogram counts of the upper-triangle FCD values."""
    fcd = fcd_matrix(x, window)
    rows, cols = upper(fcd.shape[0])
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


def stable_arctanh(r: Array) -> Array:
    """arctanh with |r| >= 1 clipped to the largest value below 1 of r's dtype, sign kept.

    In float32, the FC of near-perfectly synchronised regions (true r = 0.99999999) rounds to
    1.0 or 1.0000001, where plain arctanh is inf or NaN. As CBIG_StableAtanh (data/empirical.py)
    this maps them to a large finite value (8.66 in float32). Every |r| < 1 is unchanged.
    NaN passes through.
    """
    bound = 1.0 - jnp.finfo(r.dtype).epsneg
    return jnp.arctanh(jnp.clip(r, -bound, bound))


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
                   pairwise=stable_arctanh(fc[rows, cols]),
                   fc_moments=fc_moments(fc))


def fcd_levels(cdf: Array, stride: int) -> Array:
    """
    cdf: (..., B) cumulative counts at B bins on [-1, 1]
    returns: (..., B // stride) the CDF at every stride-th bin's right edge. The last level
             is the total count, so the result can still be normalised.
    """
    B = cdf.shape[-1]
    if stride < 1 or B % stride:
        raise ValueError(f"fcd stride {stride} must divide the {B} bins")
    return cdf[..., stride - 1::stride]


# pycatch22 1.0.2's order and names (catch24=True): the 22 catch22 features, then mean and SD
CATCH24_NAMES = (
    "DN_HistogramMode_5", "DN_HistogramMode_10", "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd", "CO_FirstCrossing_acf_1e_pointOfCrossing", "CO_FirstMin_ac",
    "SP_Summaries_fft_area_5_1", "SP_Summaries_fft_wq_50", "FC_LocalSimple_mean3_stde",
    "FC_LocalSimple_mean1_taurat", "MD_pNN_pnn40", "SB_BinaryStats_mean_longstretch1",
    "SB_BinaryStats_diff_longstretch0", "SB_MotifThree_quantile_hh",
    "CO_HistogramAMI_even_5bin_ami2", "CO_trev_1_num", "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "SB_TransitionMatrix_quantile_3_ac_sumdiagcov", "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "DN_mean", "DN_Spread_std")


def catch24(x: np.ndarray) -> np.ndarray:
    """x: (T, N) observed signal at TR -> (N, 24) float32, the catch24 features of each
    region's signal in CATCH24_NAMES order (numpy on the host, not jit-able)."""
    x = np.asarray(x, np.float64)
    out = np.empty((x.shape[1], len(CATCH24_NAMES)), np.float32)
    for i in range(x.shape[1]):
        r = pycatch22.catch22_all(x[:, i].tolist(), catch24=True)
        if tuple(r["names"]) != CATCH24_NAMES:
            raise RuntimeError(f"pycatch22 returned features {r['names']}, expected "
                               f"{CATCH24_NAMES} (a different pycatch22 version?)")
        out[i] = r["values"]
    return out
