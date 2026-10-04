"""sim/summary.py against independent computations.

- FC on synthetic signals with a known correlation matrix, and against numpy.corrcoef;
- FCD CDF against a plain NumPy loop that mirrors the MATLAB FCD_cdf_calculate.m;
- moments against scipy.stats;
- on the cluster: the stored HCP-YA run-level FC and FCD CDF, recomputed from the stored
  time courses, so the summary pipeline is the empirical pipeline.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from conftest import data_available
from delssome_fm.config import SimConfig
from delssome_fm.data.empirical import FCD_BINS, read_csv_array
from delssome_fm.sim.summary import (fc_moments, fcd_cdf, functional_connectivity,
                                     regional_statistics, summarize, window_fc_vectors)

CFG = SimConfig(dt=0.001, tr=0.72, n_frames=1200, burn_in_frames=0, divergence_bound=1e6,
                fcd_window=83, fcd_bins=10000)


def _numpy_fcd_cdf(x, window, bins):
    """Reference: the MATLAB pipeline, written as loops (x: (T, N))."""
    T, N = x.shape
    iu = np.triu_indices(N, 1)
    vecs = np.stack([np.corrcoef(x[t:t + window].T)[iu] for t in range(T - window + 1)])
    fcd = np.corrcoef(vecs)
    counts, _ = np.histogram(fcd[np.triu_indices(len(vecs), 1)], bins=bins, range=(-1, 1))
    return np.cumsum(counts)


def test_fc_recovers_known_correlation():
    rng = np.random.default_rng(0)
    R = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.1], [-0.3, 0.1, 1.0]])
    x = rng.normal(size=(200_000, 3)) @ np.linalg.cholesky(R).T + np.array([5.0, -2.0, 0.0])
    fc = np.asarray(functional_connectivity(jnp.asarray(x, jnp.float32)))
    np.testing.assert_allclose(fc, R, atol=0.01)


def test_fc_matches_numpy_corrcoef():
    x = np.random.default_rng(1).normal(size=(300, 10))
    with jax.enable_x64(True):
        fc = np.asarray(functional_connectivity(jnp.asarray(x)))
    np.testing.assert_allclose(fc, np.corrcoef(x.T), rtol=1e-12, atol=1e-12)


def test_window_fc_vectors_match_loop():
    x = np.random.default_rng(2).normal(size=(120, 6))
    with jax.enable_x64(True):
        got = np.asarray(window_fc_vectors(jnp.asarray(x), 30))
    iu = np.triu_indices(6, 1)
    ref = np.stack([np.corrcoef(x[t:t + 30].T)[iu] for t in range(120 - 30 + 1)])
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_fcd_cdf_matches_numpy_reference():
    x = np.random.default_rng(3).normal(size=(400, 12)).cumsum(0)  # slow drifts: varied FCD
    with jax.enable_x64(True):
        got = np.asarray(fcd_cdf(jnp.asarray(x), 40, 1000))
    np.testing.assert_array_equal(got, _numpy_fcd_cdf(x, 40, 1000))
    n_win = 400 - 40 + 1
    assert got[-1] == n_win * (n_win - 1) // 2


def test_fcd_separates_a_switch_in_correlation_structure():
    """Two halves with opposite correlation patterns: windows from different halves are
    anticorrelated, so the FCD has mass near -1 that a stationary signal lacks."""
    rng = np.random.default_rng(4)
    T, N = 600, 8
    pattern = rng.choice([-1.0, 1.0], size=N)
    common = rng.normal(size=(T, 1))
    sign = np.where(np.arange(T)[:, None] < T // 2, 1.0, -1.0)
    switching = 2 * common * pattern * np.where(pattern > 0, 1.0, sign) + rng.normal(size=(T, N))
    stationary = 2 * common * pattern + rng.normal(size=(T, N))
    with jax.enable_x64(True):
        cdf_switch = np.asarray(fcd_cdf(jnp.asarray(switching), 60, 200))
        cdf_stat = np.asarray(fcd_cdf(jnp.asarray(stationary), 60, 200))
    below_zero = 100  # bins below 0 on [-1, 1] with 200 bins
    assert cdf_switch[below_zero - 1] / cdf_switch[-1] > 0.3
    assert cdf_stat[below_zero - 1] / cdf_stat[-1] < 0.05


def test_regional_statistics_match_scipy():
    rng = np.random.default_rng(5)
    x = rng.gamma(2.0, size=(500, 7)).cumsum(0) % 13.0
    with jax.enable_x64(True):
        fc = functional_connectivity(jnp.asarray(x))
        reg = np.asarray(regional_statistics(jnp.asarray(x), fc))
        mom = np.asarray(fc_moments(fc))
    np.testing.assert_allclose(reg[:, 0], x.mean(0), rtol=1e-12)
    np.testing.assert_allclose(reg[:, 1], x.std(0), rtol=1e-12)
    np.testing.assert_allclose(reg[:, 2], stats.skew(x, axis=0), rtol=1e-10)
    np.testing.assert_allclose(reg[:, 3], stats.kurtosis(x, axis=0), rtol=1e-10)
    R = np.corrcoef(x.T)
    np.testing.assert_allclose(reg[:, 4], (R.sum(1) - 1) / 6, rtol=1e-10)
    lag1 = [np.corrcoef(x[:-1, i], x[1:, i])[0, 1] for i in range(7)]
    np.testing.assert_allclose(reg[:, 5], lag1, rtol=1e-10)
    up = R[np.triu_indices(7, 1)]
    np.testing.assert_allclose(mom, [up.mean(), up.std(), stats.skew(up)], rtol=1e-10)


def test_summarize_shapes_and_pairwise_is_arctanh():
    x = jnp.asarray(np.random.default_rng(6).normal(size=(200, 5)), jnp.float32)
    cfg = SimConfig(0.001, 0.72, 200, 0, 1e6, 50, 100)
    s = summarize(x, cfg)
    assert s.fc.shape == (5, 5) and s.fcd_cdf.shape == (100,) and s.regional.shape == (5, 6)
    assert s.pairwise.shape == (10,) and s.fc_moments.shape == (3,)
    np.testing.assert_allclose(np.tanh(np.asarray(s.pairwise)),
                               np.asarray(s.fc)[np.triu_indices(5, 1)], rtol=1e-5)


def test_summarize_works_under_vmap_and_jit():
    x = jnp.asarray(np.random.default_rng(7).normal(size=(3, 200, 5)), jnp.float32)
    cfg = SimConfig(0.001, 0.72, 200, 0, 1e6, 50, 100)
    batched = jax.jit(jax.vmap(lambda a: summarize(a, cfg)))(x)
    single = summarize(x[1], cfg)
    np.testing.assert_allclose(np.asarray(batched.fc[1]), np.asarray(single.fc), rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(batched.fcd_cdf[1]), np.asarray(single.fcd_cdf))


# ---------------------------------------------------------------- the empirical pipeline

RUNS = ("100206_bld001", "459453_bld003", "996782_bld002")


@pytest.mark.cluster
@pytest.mark.skipif(not data_available(), reason="HCP-YA data not reachable")
@pytest.mark.parametrize("run", RUNS)
def test_summary_reproduces_stored_run_fc_and_fcd(data_cfg, run):
    root = Path(data_cfg.run_fc_dir).parents[1]
    tc_path = root / "TC" / "DK68" / f"{run}.csv"
    if not tc_path.is_file():
        pytest.skip(f"time course not available: {tc_path}")
    tc = read_csv_array(tc_path, (68, 1200)).T  # (T, N)
    with jax.enable_x64(True):
        s = summarize(jnp.asarray(tc), CFG)
        fc, cdf = np.asarray(s.fc), np.asarray(s.fcd_cdf)
    fc_ref = read_csv_array(data_cfg.run_fc_dir / f"{run}.csv", (68, 68))
    cdf_ref = read_csv_array(data_cfg.run_fcd_dir / f"{run}.csv", (FCD_BINS,))
    np.testing.assert_allclose(fc, fc_ref, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(cdf, cdf_ref)
