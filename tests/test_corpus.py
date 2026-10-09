"""sim/corpus.py: the flat rule, determinism and split-invariance of the draws, splits as
disjoint populations, validity, extensions, and the status of a small corpus."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from delssome_fm.config import CorpusConfig, SimConfig
from delssome_fm.sim.corpus import (K_MAX, P_MAX, _DRAW, card_for, corpus_status, draws,
                                    extend_models, first_valid, flat_from_moments, is_flat,
                                    read_logs, run_candidates, sc_group, simulate_draws,
                                    write_chunk)
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.reference import load_reference

N = 68


def _cfg(tmp_path: Path, n_param_sets: int = 3, first_draw: int = 0) -> CorpusConfig:
    return CorpusConfig(seed=11, n_param_sets=n_param_sets, first_draw=first_draw, batch_size=2,
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
    y = np.concatenate([lively, still, mixed])
    assert is_flat(y, 1e-5).tolist() == [False, True, False]
    # the corpus applies the same rule to stored per-region moments
    assert flat_from_moments(y.mean(1), y.std(1), 1e-5).tolist() == [False, True, False]


def test_draws_do_not_depend_on_how_they_are_split(tmp_path):
    cfg = _cfg(tmp_path)
    card = card_for(cfg, "train", 3)
    theta, psi, x0, keys = draws(cfg, card, "train", 3, _DRAW, 0, 10)
    theta5, psi5, x05, keys5 = draws(cfg, card, "train", 3, _DRAW, 5, 5)
    np.testing.assert_array_equal(theta[5:], theta5)
    np.testing.assert_array_equal(x0[5:], x05)
    np.testing.assert_array_equal(jax.random.key_data(keys[5:]), jax.random.key_data(keys5))
    assert theta.shape == (10, N, len(card.regional_params))
    assert np.abs(x0).max() <= cfg.initial_state_scale


def test_splits_are_different_model_populations(tmp_path):
    cfg = _cfg(tmp_path)
    assert card_for(cfg, "train", 7) != card_for(cfg, "val", 7) != card_for(cfg, "test", 7)
    assert card_for(cfg, "val", 7) == card_for(cfg, "val", 7)


def test_sc_group_is_deterministic_and_covers_the_groups(tmp_path):
    cfg = _cfg(tmp_path)
    picks = [sc_group(cfg, "train", i, 64) for i in range(2000)]
    assert picks == [sc_group(cfg, "train", i, 64) for i in range(2000)]
    assert set(picks) == set(range(64))
    assert set(sc_group(cfg, "test", i, 13) for i in range(500)) == set(range(13))


def test_tiny_chunk_end_to_end_and_status(tmp_path):
    cfg = _cfg(tmp_path)
    scs = np.random.default_rng(0).uniform(0, 0.02, (64, N, N))
    logs, arrays, cards = run_candidates(cfg, SIM, "train", scs, 0, 8)
    assert [r.index for r in logs] == list(range(8))
    for r in logs:
        assert r.valid == (r.kept and r.n_draws_diverged + r.n_draws_flat < r.n_draws)
    valid = [r.index for r in logs if r.valid]
    assert [c["index"] for c in cards] == valid
    if valid:
        n = len(valid) * cfg.n_param_sets
        assert arrays["theta"].shape == (n, N, P_MAX) and arrays["psi"].shape == (n, K_MAX)
        assert arrays["pairwise"].shape == (n, N * (N - 1) // 2)
        assert arrays["fcd_cdf"].shape == (n, SIM.fcd_bins // cfg.fcd_store_stride)
        assert sorted(set(arrays["model"].tolist())) == valid
        assert np.isnan(arrays["regional"][arrays["diverged"]]).all()
        assert not (arrays["diverged"] & arrays["flat"]).any()
    paths = write_chunk(cfg, "train", 0, 8, logs, arrays, cards)
    assert len(paths["log"].read_text().splitlines()) == 8
    assert all(json.loads(line)["index"] in valid for line in paths["cards"].read_text().splitlines())
    status = corpus_status(cfg, "train")
    assert status["screened"] == 8 and status["n_contiguous"] == 8
    assert status["valid"] == len(valid)
    assert first_valid(cfg, "train", len(valid)) == valid
    with pytest.raises(ValueError, match="only"):
        first_valid(cfg, "train", len(valid) + 1)
    assert [r.index for r in read_logs(cfg, "train")] == list(range(8))


def test_extension_gives_the_same_draws_as_a_longer_base_run(tmp_path):
    """Draw 2 computed by an extension (first_draw 2) equals draw 2 of a base run."""
    model = compile_card(load_reference("hopf"))  # bounded for any multipliers
    sc = np.zeros((N, N))
    base = simulate_draws(_cfg(tmp_path, 3), SIM, model, "train", 5, sc, 0, 3)
    ext = simulate_draws(_cfg(tmp_path, 1, first_draw=2), SIM, model, "train", 5, sc, 2, 1)
    assert ext["draw"].tolist() == [2]
    np.testing.assert_array_equal(ext["theta"][0], base["theta"][2])
    np.testing.assert_allclose(ext["regional"][0], base["regional"][2], rtol=1e-6)


def test_extend_requires_later_draws(tmp_path):
    with pytest.raises(ValueError, match="first_draw must be > 0"):
        extend_models(_cfg(tmp_path), SIM, "train", np.zeros((1, N, N)), [0])
    with pytest.raises(ValueError, match="first_draw must be 0"):
        run_candidates(_cfg(tmp_path, first_draw=3), SIM, "train", np.zeros((1, N, N)), 0, 1)


def test_simulate_draws_with_a_stable_card(tmp_path):
    cfg = _cfg(tmp_path, n_param_sets=3)  # batch_size 2: exercises padding of the last batch
    res = simulate_draws(cfg, SIM, compile_card(load_reference("hopf")), "train", 0,
                         np.zeros((N, N)), 0, 3)
    assert res["draw"].tolist() == [0, 1, 2]
    assert not res["diverged"].any() and not res["flat"].any()
    assert res["regional"].shape == (3, N, 6) and np.isfinite(res["regional"]).all()
    assert np.isnan(res["theta"][..., 3:]).all() and np.isfinite(res["theta"][..., :3]).all()
    assert np.all(np.diff(res["fcd_cdf"], axis=1) >= 0)
    assert res["regional"].dtype == np.float32 and res["theta"].dtype == np.float32
