"""sim/corpus.py: the flat rule, determinism and split-invariance of the draws, splits as
disjoint populations, validity, extensions, and the status of a small corpus."""

import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
import pytest

from delssome_fm.config import CorpusConfig, SimConfig
from delssome_fm.sim.corpus import (K_MAX, P_MAX, _DRAW, card_for, corpus_status, draws,
                                    extend_models, first_valid, flat_draws, flat_from_moments,
                                    is_flat, read_chunk, read_logs, rebuild_candidates,
                                    run_candidates, sc_group, simulate_draws, write_chunk)
from delssome_fm.sim import corpus as corpus_mod
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.reference import load_reference

N = 68


def _cfg(tmp_path: Path, n_param_sets: int = 3, first_draw: int = 0) -> CorpusConfig:
    return CorpusConfig(seed=11, n_param_sets=n_param_sets, first_draw=first_draw, batch_size=2,
                        screen_draws=2, screen_frames=40, screen_burn_in_frames=5,
                        flat_rel_sd=1e-6, initial_state_scale=0.1, fcd_store_stride=10,
                        parameter_policy="multiplier", gain_scope="draw", rate_floor=0.0,
                        gain_prior="absolute", term_prior="flat", level_prior="flat", noise_prior="flat", observable_rest="free",
                        output_dir=tmp_path)


SIM = SimConfig(dt=0.001, tr=0.72, n_frames=40, burn_in_frames=5, divergence_bound=1e6,
                fcd_window=10, fcd_bins=100)


def test_flat_rule():
    t = np.linspace(0, 1, 50)[None, :, None]
    lively = np.broadcast_to(np.sin(20 * t), (1, 50, 3))
    still = np.full((1, 50, 3), 7.0) + 1e-7 * np.sin(20 * t)   # SD ~7e-8 <= 1e-6 * 7
    one_still = np.concatenate([lively[..., :1], still[..., 1:2], lively[..., 2:]], axis=2)
    tiny = 1e-9 * np.sin(20 * t) * np.ones((1, 50, 3))         # tiny but resolved: not flat
    zero = np.zeros((1, 50, 3))                                 # SD 0 at mean 0: flat
    y = np.concatenate([lively, still, one_still, tiny, zero])
    expected = [False, True, True, False, True]
    assert is_flat(y, 1e-6).tolist() == expected
    # the corpus applies the same rule to stored per-region moments
    assert flat_from_moments(y.mean(1), y.std(1), 1e-6).tolist() == expected


def test_flat_draws_also_flags_non_finite_statistics():
    B = 3
    res = {"diverged": np.array([False, False, True]),
           "regional": np.ones((B, N, 6)), "pairwise": np.zeros((B, 5)),
           "fcd_cdf": np.zeros((B, 4)), "fc_moments": np.zeros((B, 3))}
    res["pairwise"][1, 2] = np.nan
    res["regional"][2] = np.nan                       # diverged draws are never flat
    assert flat_draws(res, 1e-6).tolist() == [False, True, False]


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
    assert res["catch24"].shape == (3, N, 24) and np.isfinite(res["catch24"]).all()


def test_rebuild_of_an_old_chunk_equals_a_fresh_run(tmp_path, monkeypatch):
    """An "old" chunk is made from a fresh one by applying the old flat rule and putting an
    inf into one model's pairwise statistics (as float32 FC saturation did). Rebuilding it
    gives the fresh chunk exactly, simulating base draws only for that model."""
    cfg = _cfg(tmp_path)
    scs = np.random.default_rng(0).uniform(0, 0.02, (64, N, N))
    logs, arrays, cards = run_candidates(cfg, SIM, "train", scs, 0, 12)
    assert np.isnan(arrays["catch24"][arrays["diverged"]]).all()      # NaN where diverged
    valid = [r.index for r in logs if r.valid]
    assert len(valid) >= 2 and any(not r.kept and r.screen_diverged == cfg.screen_draws
                                   for r in logs)
    old = {k: v.copy() for k, v in arrays.items()}
    mean, sd = old["regional"][..., 0], old["regional"][..., 1]
    old["flat"] = ~old["diverged"] & np.all(sd <= 1e-5 * np.maximum(1, np.abs(mean)), axis=1)
    hit = np.flatnonzero((old["model"] == valid[0]) & ~old["diverged"])[0]
    old["pairwise"][hit, 0] = np.inf
    write_chunk(cfg, "train", 0, 12, logs, old, cards)
    old_logs, old_arrays = read_chunk(cfg, "train", 0, 12)

    simulated = []
    real = corpus_mod.simulate_draws
    monkeypatch.setattr(corpus_mod, "simulate_draws",
                        lambda *a: simulated.append(a[4]) or real(*a))
    new_logs, new_arrays, new_cards = rebuild_candidates(cfg, SIM, "train", scs, 0, 12,
                                                         old_logs, old_arrays)
    # simulated again: the doctored model, kept models without stored draws, and models
    # whose own pairwise statistics were already non-finite outside diverged draws
    unfinite = {int(m) for m in np.unique(arrays["model"])
                if not np.isfinite(arrays["pairwise"][(arrays["model"] == m)
                                                      & ~arrays["diverged"]]).all()}
    expected = sorted({valid[0]} | unfinite | {r.index for r in logs if r.kept and not r.valid})
    assert simulated == expected and len(simulated) < sum(r.kept for r in logs)
    as_json = lambda rows: [json.dumps(asdict(r)) for r in rows]   # NaN-safe equality
    assert as_json(new_logs) == as_json(logs) and new_cards == cards
    assert new_arrays.keys() == arrays.keys()
    for k in arrays:
        np.testing.assert_array_equal(new_arrays[k], arrays[k], err_msg=k)


@pytest.mark.parametrize("policy", ["multiplier", "sinh"])
def test_model_gains_are_shared_by_all_draws_and_leave_the_rest_unchanged(tmp_path, policy):
    from dataclasses import replace
    from delssome_fm.sim.corpus import _SCREEN, model_gains
    per_draw = replace(_cfg(tmp_path), parameter_policy=policy)
    per_model = replace(per_draw, gain_scope="model")
    index = next(i for i in range(50) if len(card_for(per_draw, "train", i).global_params) > 0)
    card = card_for(per_draw, "train", index)
    a = draws(per_draw, card, "train", index, _DRAW, 0, 6)
    b = draws(per_model, card, "train", index, _DRAW, 0, 6)
    screen = draws(per_model, card, "train", index, _SCREEN, 0, 3)
    gains = model_gains(per_model, card, "train", index)
    assert gains.shape == (len(card.global_params),)
    np.testing.assert_array_equal(b[1], np.broadcast_to(gains, b[1].shape))     # every draw
    np.testing.assert_array_equal(screen[1], np.broadcast_to(gains, screen[1].shape))
    assert len(np.unique(a[1][:, 0])) == 6                                       # per draw: all differ
    for k in (0, 2):                                                             # theta, x0
        np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_array_equal(jax.random.key_data(a[3]), jax.random.key_data(b[3]))
    other = next(i for i in range(index + 1, 80)
                 if len(card_for(per_draw, "train", i).global_params) > 0)
    assert not np.array_equal(model_gains(per_model, card_for(per_model, "train", other), "train",
                                          other)[:1], gains[:1])
    with pytest.raises(ValueError, match="gain_scope must be one of"):
        draws(replace(per_draw, gain_scope="row"), card, "train", index, _DRAW, 0, 1)


@pytest.mark.cluster
def test_per_draw_gains_still_reproduce_the_stored_sinh_corpus():
    from delssome_fm.config import load_config
    cfg = load_config(Path("configs/corpus_sinh.yaml"), CorpusConfig)
    _, arrays = read_chunk(cfg, "train", 0, 5)
    index = int(arrays["model"][0])
    card = card_for(cfg, "train", index)
    theta, psi, _, _ = draws(cfg, card, "train", index, _DRAW, 0, 3)
    rows = np.flatnonzero(arrays["model"] == index)[:3]
    P, K = theta.shape[2], psi.shape[1]
    np.testing.assert_allclose(arrays["theta"][rows][..., :P], theta, rtol=1e-6)
    np.testing.assert_allclose(arrays["psi"][rows][:, :K], psi, rtol=1e-6)


def test_relative_gains_fix_the_ratio_per_model(tmp_path):
    """gain_prior relative, gain_scope model: every draw of a model has the model's coupling
    ratio; its gain follows the draw's reference. Theta, x0 and noise are those of absolute
    gains with the same floor."""
    from dataclasses import replace
    from delssome_fm.sim.corpus import _SCREEN, base_level_spread, model_gains, model_ratios
    from delssome_fm.spec.sampler import SC_ROW_SUM, gain_references
    absolute = replace(_cfg(tmp_path), parameter_policy="sinh", gain_scope="model",
                       rate_floor=0.1)
    relative = replace(absolute, gain_prior="relative")
    constant = varying = False
    for index in range(60):
        card = card_for(relative, "train", index)
        if not card.global_params:
            continue
        assert card == card_for(absolute, "train", index)
        ratio = model_ratios(relative, card, "train", index)
        assert ratio.shape == (len(card.global_params),)
        assert ((ratio >= 1e-2) & (ratio <= 10)).all()
        a = draws(absolute, card, "train", index, _DRAW, 0, 5)
        b = draws(relative, card, "train", index, _DRAW, 0, 5)
        for k in (0, 2):                                                         # theta, x0
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(jax.random.key_data(a[3]), jax.random.key_data(b[3]))
        z = np.stack([base_level_spread(relative, card, "train", index, j)[0] for j in range(5)])
        reference = gain_references(card, z)
        np.testing.assert_allclose(b[1], ratio * reference / SC_ROW_SUM, rtol=1e-12)
        screen = draws(relative, card, "train", index, _SCREEN, 0, 2)
        assert screen[1].shape == (2, len(card.global_params))
        if np.ptp(reference[:, 0]) == 0:
            constant = True
            assert np.ptp(b[1][:, 0]) == 0                    # a constant reference: one gain
        else:
            varying = True
    assert constant and varying
    card = card_for(relative, "train", index)
    with pytest.raises(ValueError, match="absolute gains only"):
        model_gains(relative, card, "train", index)
    with pytest.raises(ValueError, match="relative or loop gains only"):
        model_ratios(absolute, card, "train", index)


def test_loop_gains_and_scaled_terms_reach_the_corpus_draws(tmp_path):
    from dataclasses import replace
    from delssome_fm.sim.corpus import base_level_spread, model_ratios
    from delssome_fm.spec.sampler import SC_ROW_SUM, gain_references, sample_card
    cfg = replace(_cfg(tmp_path), parameter_policy="sinh", gain_scope="model", rate_floor=0.1,
                  gain_prior="loop", term_prior="scaled")
    checked = 0
    for index in range(40):
        card = card_for(cfg, "train", index)
        assert card == sample_card(cfg.seed, 0, index, "sinh", 0.1, "scaled")
        if not card.global_params:
            continue
        theta, psi, _, _ = draws(cfg, card, "train", index, _DRAW, 0, 4)
        z = np.stack([base_level_spread(cfg, card, "train", index, j)[0] for j in range(4)])
        ratio = model_ratios(cfg, card, "train", index)
        np.testing.assert_allclose(psi, ratio * gain_references(card, z, "loop") / SC_ROW_SUM,
                                   rtol=1e-12)
        np.testing.assert_array_equal(
            theta, np.stack([base_level_spread(cfg, card, "train", index, j)[2]
                             for j in range(4)]))
        checked += 1
    assert checked > 10


def test_the_anchored_prior_reaches_the_corpus_draws(tmp_path):
    from dataclasses import replace
    from delssome_fm.sim.corpus import base_level_spread, model_gains, model_sample
    from delssome_fm.spec.sampler import LEVEL_WINDOW_LOG10, SINH_SCALE
    flat = replace(_cfg(tmp_path), parameter_policy="sinh", gain_scope="model", rate_floor=0.1,
                   gain_prior="anchored", term_prior="anchored")
    nominal = replace(flat, level_prior="nominal")
    checked = 0
    for index in range(40):
        sample = model_sample(flat, "train", index)
        card = sample.card
        assert card == card_for(nominal, "train", index)
        a = draws(flat, card, "train", index, _DRAW, 0, 4)
        b = draws(nominal, card, "train", index, _DRAW, 0, 4)
        np.testing.assert_array_equal(a[1], np.broadcast_to(sample.gains, a[1].shape))
        np.testing.assert_array_equal(a[1], b[1])                        # the model's gains
        np.testing.assert_array_equal(model_gains(flat, card, "train", index), sample.gains)
        if sample.rest is not None and (sample.rest != 0).all():         # starts at rest
            assert (np.abs(a[2] / sample.rest - 1) <= 0.1 + 1e-9).all()
            np.testing.assert_array_equal(a[2], b[2])
        for j in range(4):                                               # levels near nominal
            z = base_level_spread(nominal, card, "train", index, j)[0]
            for k, p in enumerate(card.regional_params):
                if p != "sigma":
                    r = np.log10(SINH_SCALE * np.sinh(z[k]) / sample.nominal[k])
                    assert r <= LEVEL_WINDOW_LOG10[1] + 1e-6
                    checked += 1
    assert checked > 100
    with pytest.raises(ValueError, match="anchored prior needs"):
        card_for(replace(flat, gain_prior="loop"), "train", 0)
    with pytest.raises(ValueError, match="anchored prior needs"):
        card_for(replace(flat, gain_scope="draw"), "train", 0)
    with pytest.raises(ValueError, match="level_prior must be one of"):
        card_for(replace(flat, level_prior="window"), "train", 0)
    with pytest.raises(ValueError, match="not the model's card"):
        draws(flat, card_for(flat, "train", 1), "train", 0, _DRAW, 0, 1)


def test_rest_noise_and_bold_rest_need_the_anchored_prior(tmp_path):
    """noise_prior rest: a draw's sigma follows the model's resting state and nothing else of
    the draw changes; both options raise without the anchored prior."""
    from dataclasses import replace
    from delssome_fm.sim.corpus import _DRAW, base_level_spread, draws, model_sample
    flat = replace(_cfg(tmp_path), parameter_policy="sinh", gain_scope="model",
                   rate_floor=0.1, gain_prior="anchored", term_prior="anchored",
                   level_prior="nominal")
    rest = replace(flat, noise_prior="rest")
    seen = 0
    for index in range(40):
        m = model_sample(rest, "train", index)
        assert m.card == model_sample(flat, "train", index).card
        if m.noise_level is None or "sigma" not in m.card.regional_params:
            continue
        j = m.card.regional_params.index("sigma")
        a = draws(flat, m.card, "train", index, _DRAW, 0, 6)
        b = draws(rest, m.card, "train", index, _DRAW, 0, 6)
        others = [k for k in range(a[0].shape[-1]) if k != j]
        np.testing.assert_array_equal(a[0][..., others], b[0][..., others])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        level = np.stack([np.sinh(base_level_spread(rest, m.card, "train", index, k)[0][j])
                          for k in range(6)]) * 1e-4
        assert (level <= m.noise_level * (1 + 1e-9)).all()
        assert (level >= m.noise_level * 1e-3 * (1 - 1e-9)).all()
        seen += 1
    assert seen > 10
    with pytest.raises(ValueError, match="noise_prior rest needs the anchored prior"):
        model_sample(replace(_cfg(tmp_path), noise_prior="rest"), "train", 0)
    with pytest.raises(ValueError, match="noise_prior must be one of"):
        model_sample(replace(flat, noise_prior="state"), "train", 0)
    with pytest.raises(ValueError, match="needs term_prior anchored"):
        model_sample(replace(_cfg(tmp_path), parameter_policy="sinh",
                             observable_rest="bold"), "train", 0)
