"""train/pack.py: target transforms, equation-graph tensors, and packing a tiny corpus."""

import json

import numpy as np
import pytest

from delssome_fm.config import SimConfig
from delssome_fm.sim.corpus import read_logs
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.dag import shortest_paths
from delssome_fm.spec.reference import REFERENCE_NAMES, load_reference
from delssome_fm.train.pack import (graph_arrays, n_window_pairs, transform_fc_moments,
                                    transform_regional)


def test_regional_transforms_by_hand():
    raw = np.array([[2.0, 100.0, 0.5, 0.0, 0.3, 0.9]])      # mean, SD, skew, excess kurtosis, ...
    out = transform_regional(raw)[0]
    expected = [np.arcsinh(0.02), 2.0, np.arcsinh(0.5), np.log(3.0), 0.3, 0.9]
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    assert np.isfinite(transform_regional(np.array([[0.0, 1e-9, 0.0, -2.0, 0.0, 0.0]]))).all()


def test_fc_moment_transform_takes_log10_of_the_sd_only():
    np.testing.assert_allclose(transform_fc_moments(np.array([0.2, 0.01, -1.5])),
                               [0.2, -2.0, -1.5], rtol=1e-6)


def test_window_pairs_match_the_empirical_count():
    cfg = SimConfig(dt=0.001, tr=0.72, n_frames=1200, burn_in_frames=200, divergence_bound=1e6,
                    fcd_window=83, fcd_bins=10000)
    assert n_window_pairs(cfg) == 624_403          # docs/data.md §3


@pytest.mark.parametrize("name", REFERENCE_NAMES)
def test_shortest_paths_and_graph_arrays(name):
    dag = compile_card(load_reference(name)).dag
    d = shortest_paths(dag)
    assert (d == d.T).all() and (np.diag(d) == 0).all() and (d >= 0).all()   # connected
    for parent, child in dag.edges():
        assert d[parent, child] == 1
    g = graph_arrays(dag)
    assert g["degree"][:, 0].sum() == g["degree"][:, 1].sum() == len(dag.edges())
    assert list(g["regional_par"][g["regional_par"] >= 0]) == list(dag.regional_par_nodes)
    assert g["features"].shape == (dag.n_nodes, 21)


def test_packed_tiny_corpus(tiny_store):
    cfg, corpus_cfg, sim_cfg = tiny_store
    root = cfg.store_dir / "train"
    summary = json.loads((root / "summary.json").read_text())
    logs = {r.index: r for r in read_logs(corpus_cfg, "train")}
    valid = [i for i, r in logs.items() if r.valid]
    models = np.load(root / "models.npz")
    assert models["index"].tolist() == valid
    model = np.load(root / "model.npy")
    assert len(model) == summary["rows"] > 0 and (np.diff(model) >= 0).all()
    # every stored row is usable, so each model has at most its usable draws
    for row, i in enumerate(valid):
        r = logs[i]
        assert models["n_rows"][row] <= r.n_draws - r.n_draws_diverged - r.n_draws_flat
    fcd = np.load(root / "fcd.npy")
    np.testing.assert_allclose(fcd[:, -1], 1.0)                 # complete FCD, normalised
    assert (np.diff(fcd, axis=1) >= 0).all()
    for k in ("regional", "pairwise", "fc_moments", "theta"):
        a = np.load(root / f"{k}.npy")
        assert np.isfinite(a if k != "theta" else a[..., :1]).all()
    # graph arrays line up with the models' DAG sizes
    sizes = np.diff(models["node_offset"])
    assert len(np.load(root / "node_features.npy")) == sizes.sum()
    assert len(np.load(root / "distances.npy")) == (sizes ** 2).sum()
    norm = np.load(cfg.store_dir / "normalisation.npz")
    z = (np.load(root / "regional.npy") - norm["regional_mean"]) / norm["regional_sd"]
    np.testing.assert_allclose(z.reshape(-1, 6).mean(0), 0, atol=1e-4)


def test_catch24_is_packed_transformed_and_z_scored(tiny_store):
    """The tiny corpus stores catch24, so the store has it: finite, (R, N, 24), with
    per-feature normalisation statistics, and the loader serves it z-scored."""
    from delssome_fm.train.pack import CATCH24_TRANSFORMS, transform_catch24
    cfg, _, _ = tiny_store
    x = np.load(cfg.store_dir / "train" / "catch24.npy")
    assert x.shape[1:] == (68, 24) and np.isfinite(x).all() and len(CATCH24_TRANSFORMS) == 24
    with np.load(cfg.store_dir / "normalisation.npz") as n:
        assert n["catch24_mean"].shape == (24,) and (n["catch24_sd"] > 0).all()
    raw = np.ones((1, 2, 24)) * 10.0
    t = transform_catch24(raw)
    assert t[0, 0, 4] == 1.0 and t[0, 0, 12] == np.float32(np.log10(11.0))   # log10, log10(1+x)
    assert t[0, 0, 22] == np.float32(np.arcsinh(1.0)) and t[0, 0, 0] == 10.0  # mean/SD, raw


def test_select_cma_takes_the_first_members_then_a_random_rest():
    from delssome_fm.train.pack import select_cma
    generation, member = np.repeat(np.arange(5), 4), np.tile(np.arange(4), 5)
    usable = np.ones(20, bool)
    usable[[0, 5, 6]] = False        # generation 0 member 0, generation 1 members 1 and 2

    def pick(k, seed=0):
        return select_cma(generation, member, usable, k, 2, np.random.default_rng(seed))

    chosen = pick(12)
    assert len(set(chosen.tolist())) == 12 and usable[chosen].all()
    assert {1, 2, 4, 7, 8, 9, 12, 13, 16, 17} <= set(chosen.tolist())   # 2 per generation
    assert (np.diff(chosen) > 0).all()                    # generation, then member order
    np.testing.assert_array_equal(chosen, pick(12))       # the rng decides the rest
    assert len({tuple(pick(12, s).tolist()) for s in range(10)}) > 1
    assert pick(100).tolist() == np.flatnonzero(usable).tolist()       # fewer than k: all
    with pytest.raises(ValueError, match="more than"):
        pick(5)


def test_cma_chunks_must_cover_the_plan(tmp_path):
    from delssome_fm.train.pack import cma_chunks
    (tmp_path / "train").mkdir()
    for name in ("plan000000-000002.npz", "plan000003-000005.npz"):
        (tmp_path / "train" / name).touch()
    with pytest.raises(ValueError, match="no chunk, e.g. \\[\\(2, 3\\)\\]"):
        cma_chunks(tmp_path, "train", 5)
    (tmp_path / "train" / "plan000002-000003.npz").touch()
    assert [c[:2] for c in cma_chunks(tmp_path, "train", 5)] == [(0, 2), (2, 3), (3, 5)]
    with pytest.raises(ValueError, match="the plan has 4"):
        cma_chunks(tmp_path, "train", 4)
    (tmp_path / "train" / "plan000001-000002.npz").touch()
    with pytest.raises(ValueError, match="overlaps"):
        cma_chunks(tmp_path, "train", 5)


def test_cma_draws_are_packed_beside_their_split(tiny_cma_store):
    from delssome_fm.train.pack import _row_flags, n_window_pairs, pack_cma_split
    store, cfg, corpus_cfg, sim_cfg = tiny_cma_store
    root = store / "train_cma"
    summary = json.loads((root / "summary.json").read_text())
    model, info = np.load(root / "model.npy"), np.load(root / "cma.npz")
    assert len(model) == summary["rows"] > 0 and (np.diff(model) >= 0).all()
    index = np.load(store / "train" / "models.npz")["index"]
    np.testing.assert_array_equal(index[model], info["index"])     # rows point at base models
    np.testing.assert_array_equal(np.load(root / "draw.npy"),
                                  info["generation"] * cfg.population + info["member"])
    for path in sorted((cfg.output_dir / "train").glob("plan*.npz")):
        with np.load(path) as d:
            f = _row_flags(d, n_window_pairs(sim_cfg))
            ok = f["usable"] & f["complete"] & f["finite"]
            for i in np.unique(d["model"]):
                at, mine = d["model"] == i, info["index"] == i
                assert mine.sum() == min(7, ok[at].sum())
                for g in range(3):                # the first 2 usable members are always in
                    first = np.sort(d["member"][at & ok & (d["generation"] == g)])[:2]
                    assert set(first) <= set(info["member"][mine & (info["generation"] == g)])
                assert not (mine & (info["generation"] == 0) & (info["member"] == 0)).any()
    np.testing.assert_allclose(np.load(root / "fcd.npy")[:, -1], 1.0)
    assert np.isfinite(np.load(root / "regional.npy")).all()
    before = {k: np.load(root / f"{k}.npy") for k in ("model", "theta", "pairwise")}
    pack_cma_split(cfg, corpus_cfg, sim_cfg, store, "train", per_model=7, per_generation=2)
    for k, v in before.items():
        np.testing.assert_array_equal(np.load(root / f"{k}.npy"), v, err_msg=k)


def test_node_values_are_packed_and_can_be_backfilled(tiny_store, tmp_path):
    """node_value.npy holds each Const node's value (NaN elsewhere), consistent with the sign
    and log10 feature columns; pack_node_values rebuilds it from the cards and refuses a
    split whose stored features the cards do not compile to."""
    import shutil
    from delssome_fm.train.pack import pack_node_values
    cfg, corpus_cfg, _ = tiny_store
    split = cfg.store_dir / "train"
    values = np.load(split / "node_value.npy")
    features = np.load(split / "node_features.npy")
    const = features[:, 8] == 1
    assert len(values) == len(features) and const.any()
    assert np.isnan(values[~const]).all() and np.isfinite(values[const]).all()
    np.testing.assert_array_equal(np.sign(values[const]), features[const, 15])
    np.testing.assert_allclose(np.clip(np.log10(np.abs(values[const])), -4, 4),
                               features[const, 16], rtol=1e-6)
    copy = tmp_path / "train"
    shutil.copytree(split, copy)
    (copy / "node_value.npy").unlink()
    chunks = sorted((corpus_cfg.output_dir / "train").glob("candidates*_draws0-*.npz"))
    assert pack_node_values(copy, chunks) == len(np.load(copy / "models.npz")["index"])
    np.testing.assert_array_equal(np.load(copy / "node_value.npy"), values)
    features[0, 0] += 1.0
    np.save(copy / "node_features.npy", features)
    with pytest.raises(ValueError, match="other node features"):
        pack_node_values(copy, chunks)
