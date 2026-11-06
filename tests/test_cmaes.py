"""sim/cmaes.py: coordinates, start draws, the cost of stored draws, determinism (pycma never
touches the global RNG), base draws' levels and spreads, and a tiny end-to-end run."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from delssome_fm.config import CmaesConfig, CorpusConfig, SimConfig, SplitSizes
from delssome_fm.sim import cmaes
from delssome_fm.sim.corpus import (base_level_spread, card_for, draws, model_gains,
                                    run_candidates, write_chunk, _DRAW)
from delssome_fm.sim.cost import delssome_cost
from delssome_fm.spec.sampler import SINH_Z, sign_fixed

N = 68
SIM = SimConfig(dt=0.001, tr=0.72, n_frames=40, burn_in_frames=5, divergence_bound=1e6,
                fcd_window=10, fcd_bins=100)


def _corpus(tmp_path: Path) -> CorpusConfig:
    return CorpusConfig(seed=11, n_param_sets=4, first_draw=0, batch_size=2, screen_draws=2,
                        screen_frames=40, screen_burn_in_frames=5, flat_rel_sd=1e-6,
                        initial_state_scale=0.1, fcd_store_stride=10, parameter_policy="sinh",
                        gain_scope="model", output_dir=tmp_path / "corpus")


def _cfg(tmp_path: Path, **kw) -> CmaesConfig:
    base = dict(corpus_config=tmp_path / "corpus.yaml", n_models=SplitSizes(1, 1, 1),
                coupled_threshold=-1.0, max_coupled_share=1.0, max_runs=2,
                min_coupled_share=0.5, population=4,
                generations=2, sigma0=0.05, failed_cost=5.0, output_dir=tmp_path / "cma")
    return CmaesConfig(**{**base, **kw})


def _card_with_fixed_and_free(cfg: CorpusConfig):
    for i in range(200):
        card = card_for(cfg, "train", i)
        fixed = sign_fixed(card)
        if fixed.any() and not fixed.all():
            return card
    raise AssertionError("no card with both kinds of regional parameter")


def test_encode_decode_round_trip(tmp_path):
    card = _card_with_fixed_and_free(_corpus(tmp_path))
    P = len(card.regional_params)
    rng = np.random.default_rng(0)
    z, kappa = rng.uniform(-SINH_Z, SINH_Z, P), rng.uniform(0, 1, P)
    u = cmaes.encode(card, z, kappa)
    assert u.shape == (2 * P,) and np.all((u >= 0) & (u <= 1))
    z2, kappa2 = cmaes.decode(card, u)
    fixed = sign_fixed(card)
    np.testing.assert_allclose(z2, np.where(fixed, np.abs(z), z))   # only |z| for a fixed sign
    np.testing.assert_allclose(kappa2, kappa)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        cmaes.decode(card, np.full(2 * P, 1.5))
    with pytest.raises(ValueError, match="entries"):
        cmaes.decode(card, np.zeros(2 * P + 1))


def test_start_draws_rank_coupled_draws_and_reuse_the_best():
    total = np.array([0.5, 1.0, 0.2, np.nan, 0.9, 0.1])
    coupled = np.array([True, True, True, True, False, False])
    assert cmaes.start_draws(total, coupled, 2) == [2, 0]
    assert cmaes.start_draws(total, coupled, 4) == [2, 0, 1, 2]   # option a: the lowest again
    one = np.array([False, True, False, False, False, False])
    assert cmaes.start_draws(total, one, 2) == [1, 1]
    with pytest.raises(ValueError, match="no coupled draw"):
        cmaes.start_draws(total, np.array([False, False, False, True, False, False]), 2)


def test_draw_costs_equal_the_delssome_cost_of_full_matrices():
    rng = np.random.default_rng(1)
    rows, cols = np.triu_indices(N, 1)
    emp_fc = np.eye(N)
    emp_fc[rows, cols] = emp_fc[cols, rows] = rng.uniform(-0.2, 0.8, len(rows))
    emp_cdf = np.cumsum(rng.uniform(0, 1, 10))
    fcs = rng.uniform(-0.5, 0.9, (3, len(rows)))
    cdf = np.cumsum(rng.uniform(0, 1, (3, 10)), axis=1)
    arrays = {"pairwise": np.arctanh(fcs).astype(np.float32), "fcd_cdf": cdf}
    got = cmaes.draw_costs(arrays, (emp_fc[rows, cols], emp_cdf))
    with jax.enable_x64(True):
        for b in range(3):
            fc = np.eye(N)
            fc[rows, cols] = fc[cols, rows] = np.tanh(arrays["pairwise"][b].astype(np.float64))
            want = delssome_cost(jnp.asarray(fc), jnp.asarray(cdf[b]), jnp.asarray(emp_fc),
                                 jnp.asarray(emp_cdf))
            np.testing.assert_allclose(got[b], [want.corr, want.mean, want.ks], rtol=1e-12)


def test_strategy_ignores_the_global_rng(tmp_path):
    cfg = _cfg(tmp_path, population=5)
    asks = []
    for seed in (1, 2):
        np.random.seed(seed)                       # must not matter
        es = cmaes.strategy(cfg, np.full(4, 0.5), np.random.default_rng(7))
        u = np.asarray(es.ask())
        es.tell(list(u), np.sum((u - 0.3) ** 2, axis=1).tolist())
        asks.append(np.concatenate([u, np.asarray(es.ask())]))
    np.testing.assert_array_equal(asks[0], asks[1])
    assert np.all((asks[0] >= 0) & (asks[0] <= 1))
    other = cmaes.strategy(cfg, np.full(4, 0.5), np.random.default_rng(8))
    assert not np.array_equal(np.asarray(other.ask()), asks[0][:5])


def test_base_level_spread_replays_the_base_draw(tmp_path):
    cfg = _corpus(tmp_path)
    card = card_for(cfg, "train", 3)
    z, kappa, theta = base_level_spread(cfg, card, "train", 3, 2)
    stored = draws(cfg, card, "train", 3, _DRAW, 2, 1)[0][0]
    np.testing.assert_array_equal(theta, stored)
    assert z.shape == kappa.shape == (len(card.regional_params),)
    with pytest.raises(ValueError, match="sinh policy only"):
        base_level_spread(dataclasses.replace(cfg, parameter_policy="multiplier"), card,
                          "train", 3, 2)


def test_candidates_are_deterministic_and_fresh(tmp_path):
    cfg = _corpus(tmp_path)
    card = card_for(cfg, "train", 3)
    P = len(card.regional_params)
    z, kappa = np.full((2, P), 5.0), np.full((2, P), 0.3)
    a = cmaes.candidates(cfg, card, "train", 3, 0, 1, z, kappa)
    b = cmaes.candidates(cfg, card, "train", 3, 0, 1, z, kappa)
    for x, y in zip(a[:2], b[:2]):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[0][0], a[0][1])                  # fresh noise per member
    c = cmaes.candidates(cfg, card, "train", 3, 0, 2, z, kappa)   # and per generation
    assert not np.array_equal(a[0], c[0])
    d = cmaes.candidates(cfg, card, "train", 3, 1, 1, z, kappa)   # and per run
    assert not np.array_equal(a[0], d[0])


def test_tiny_end_to_end(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    scs = np.random.default_rng(0).uniform(0, 0.02, (64, N, N))
    logs, arrays, cards = run_candidates(corpus, SIM, "train", scs, 0, 6)
    valid = [r.index for r in logs if r.valid]
    assert valid, "no valid model among the first candidates"
    write_chunk(corpus, "train", 0, 6, logs, arrays, cards)
    cmaes._base_chunks.cache_clear()
    rng = np.random.default_rng(3)
    emp = (rng.uniform(-0.1, 0.5, N * (N - 1) // 2), np.cumsum(rng.uniform(0, 1, 10)))
    monkeypatch.setattr(cmaes, "empirical", lambda *a: emp)
    cfg = _cfg(tmp_path, n_models=SplitSizes(len(valid), 1, 1))

    plan = cmaes.plan_split(cfg, corpus, "train")
    assert [r["index"] for r in plan] == valid                       # threshold -1: all coupled
    cmaes.write_plan(cfg, "train", plan)
    assert cmaes.read_plan(cfg, "train") == plan
    with pytest.raises(ValueError, match="was made with"):
        cmaes.read_plan(dataclasses.replace(cfg, max_coupled_share=0.15), "train")
    index = valid[0]
    out, rows = cmaes.run_model(cfg, corpus, SIM, None, "train", index, scs)
    again, _ = cmaes.run_model(cfg, corpus, SIM, None, "train", index, scs)
    for k in out:
        np.testing.assert_array_equal(out[k], again[k], err_msg=k)
    n = cfg.generations * cfg.population
    assert len(out["model"]) == n and set(out["model"].tolist()) == {index}
    assert out["run"].tolist() == [0] * n                            # all coupled: run 0 succeeds
    assert out["generation"].tolist() == np.repeat([0, 1], 4).tolist()
    K = len(card_for(corpus, "train", index).global_params)
    np.testing.assert_allclose(out["psi"][:, :K], np.broadcast_to(
        model_gains(corpus, card_for(corpus, "train", index), "train", index), (n, K)), rtol=1e-6)
    ok = ~out["diverged"] & ~out["flat"]
    assert np.isfinite(out["cost"][ok]).all() and np.isnan(out["cost"][out["diverged"]]).all()
    assert len(rows) == 1 and rows[0]["succeeded"] and rows[0]["kept"]
    assert rows[0]["generations"] == 2 and rows[0]["usable"] == int(ok.sum())

    # every run fails (no share reaches 110%): run 1 starts, and only its candidates are kept
    strict = dataclasses.replace(cfg, min_coupled_share=1.1)
    fell, fell_rows = cmaes.run_model(strict, corpus, SIM, None, "train", index, scs)
    assert [(r["run"], r["succeeded"], r["kept"]) for r in fell_rows] == [(0, False, False),
                                                                          (1, False, True)]
    assert fell_rows[0]["history"] == rows[0]["history"]             # run 0 is the same run
    assert fell["run"].tolist() == [1] * n
    assert not np.array_equal(fell["theta"], out["theta"])
    base = cmaes.model_arrays(corpus, "train", index)
    usable = ~base["diverged"] & ~base["flat"]
    total = cmaes.draw_costs(base, emp).sum(axis=1)
    assert [r["start_draw"] for r in fell_rows] == cmaes.start_draws(total, usable, 2)

    paths = cmaes.run_chunk(cfg, corpus, SIM, None, "train", 0, 1, scs)
    with np.load(paths["arrays"]) as d:
        np.testing.assert_array_equal(d["cost"], out["cost"])
    assert len(paths["log"].read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="outside"):
        cmaes.run_chunk(cfg, corpus, SIM, None, "train", 0, len(plan) + 1, scs)
    with pytest.raises(ValueError, match="gain_scope model"):
        cmaes.run_model(cfg, dataclasses.replace(corpus, gain_scope="draw"), SIM, None, "train",
                        index, scs)

    # no coupled base draw with a finite cost (here: every FCD empty): skipped, not an error
    real = cmaes.model_arrays
    monkeypatch.setattr(cmaes, "model_arrays", lambda *a: {
        **real(*a), "fcd_cdf": np.zeros_like(real(*a)["fcd_cdf"])})
    none, skip = cmaes.run_model(cfg, corpus, SIM, None, "train", index, scs)
    assert none is None and len(skip) == 1 and "empty FCD" in skip[0]["skipped"]
    paths = cmaes.run_chunk(cfg, corpus, SIM, None, "train", 0, 1, scs)
    with np.load(paths["arrays"]) as d:
        assert d.files == []                             # every model skipped: no arrays
    assert "skipped" in json.loads(paths["log"].read_text())
