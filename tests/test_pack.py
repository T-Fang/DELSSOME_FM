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
