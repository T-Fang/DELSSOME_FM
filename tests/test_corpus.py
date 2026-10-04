"""sim/corpus.py: the screen, determinism of the draws, and a tiny end-to-end chunk."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from delssome_fm.config import CorpusConfig, SimConfig
from delssome_fm.sim.corpus import (K_MAX, P_MAX, _DRAW, draws, is_flat, run_candidates,
                                    sc_group, write_chunk)
from delssome_fm.spec.sampler import sample_card

N = 68


def _cfg(tmp_path: Path, n_param_sets: int = 3) -> CorpusConfig:
    return CorpusConfig(seed=11, n_param_sets=n_param_sets, first_draw=0, batch_size=2,
                        screen_draws=2, screen_frames=40, screen_burn_in_frames=5,
                        flat_rel_sd=1e-5, initial_state_scale=0.1, fcd_store_stride=10,
                        output_dir=tmp_path)


SIM = SimConfig(dt=0.001, tr=0.72, n_frames=40, burn_in_frames=5, divergence_bound=1e6,
                fcd_window=10, fcd_bins=100)


def test_flat_rule():
    t = np.linspace(0, 1, 50)[None, :, None]
    lively = np.broadcast_to(np.sin(20 * t), (1, 50, 3))
    still = np.full((1, 50, 3), 7.0) + 1e-7 * np.sin(20 * t)   # SD ~7e-8 << 1e-5 * 7
    mixed = np.concatenate([lively[..., :1], still[..., 1:]], axis=2)
    assert is_flat(np.concatenate([lively, still, mixed]), 1e-5).tolist() == [False, True, False]


def test_draws_do_not_depend_on_how_they_are_split(tmp_path):
    cfg, card = _cfg(tmp_path), sample_card(11, 3)
    theta, psi, x0, keys = draws(cfg, card, 3, _DRAW, 0, 10)
    theta5, psi5, x05, keys5 = draws(cfg, card, 3, _DRAW, 5, 5)
    np.testing.assert_array_equal(theta[5:], theta5)
    np.testing.assert_array_equal(x0[5:], x05)
    np.testing.assert_array_equal(jax.random.key_data(keys[5:]), jax.random.key_data(keys5))
    assert theta.shape == (10, N, len(card.regional_params))
    assert np.abs(x0).max() <= cfg.initial_state_scale


def test_sc_group_is_deterministic_and_covers_the_groups(tmp_path):
    cfg = _cfg(tmp_path)
    picks = [sc_group(cfg, i, 64) for i in range(2000)]
    assert picks == [sc_group(cfg, i, 64) for i in range(2000)]
    assert set(picks) == set(range(64))


def test_tiny_chunk_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(0)
    scs = rng.uniform(0, 0.02, (64, N, N))
    logs, arrays, cards = run_candidates(cfg, SIM, scs, 0, 6)
    assert [r.index for r in logs] == list(range(6))
    kept = [r.index for r in logs if r.kept]
    assert [c["index"] for c in cards] == kept
    if kept:
        n = len(kept) * cfg.n_param_sets
        assert arrays["theta"].shape == (n, N, P_MAX) and arrays["psi"].shape == (n, K_MAX)
        assert arrays["pairwise"].shape == (n, N * (N - 1) // 2)
        assert arrays["fcd_cdf"].shape == (n, SIM.fcd_bins // cfg.fcd_store_stride)
        assert sorted(set(arrays["model"].tolist())) == kept
        div = arrays["diverged"]
        assert np.isnan(arrays["regional"][div]).all()
        assert np.isfinite(arrays["regional"][~div]).all()
    paths = write_chunk(cfg, 0, 6, logs, arrays, cards)
    assert len(paths["log"].read_text().splitlines()) == 6
    assert all(json.loads(line)["index"] in kept for line in paths["cards"].read_text().splitlines())


def test_kept_model_path_with_a_stable_card(tmp_path):
    """The simulate-and-summarise path, independent of the sampler's acceptance rate."""
    from delssome_fm.sim.corpus import simulate_kept
    from delssome_fm.spec.compile import compile_card
    from delssome_fm.spec.reference import load_reference

    cfg = _cfg(tmp_path, n_param_sets=3)  # batch_size 2: exercises padding of the last batch
    # Hopf: bounded by its quadratic gate for any multipliers, observed directly. (Linear
    # through Balloon-Windkessel is not: an input of +-20 drives the flow f negative.)
    model = compile_card(load_reference("hopf"))
    sc = np.zeros((N, N))
    res = simulate_kept(cfg, SIM, model, 0, sc)
    assert res["draw"].tolist() == [0, 1, 2]
    assert not res["diverged"].any()
    assert res["regional"].shape == (3, N, 6) and np.isfinite(res["regional"]).all()
    assert np.isnan(res["theta"][..., 3:]).all() and np.isfinite(res["theta"][..., :3]).all()
    cdf = res["fcd_cdf"]
    assert np.all(np.diff(cdf, axis=1) >= 0)  # stored CDF subsample is non-decreasing
