"""train/analytic.py on the tiny sinh store: the rows do not depend on the chunking, the
arrays hold what each status promises, and packing accepts exactly a complete set of chunks."""

import shutil

import numpy as np
import pytest

from delssome_fm.train.analytic import (ARRAYS, STATUS_CODE, analytic_models, pack_analytic,
                                        patch_analytic, patch_targets, write_chunk)


def test_rows_do_not_depend_on_the_chunking_and_pack_round_trips(tiny_store_sinh, tmp_path):
    cfg, corpus_cfg, sim_cfg = tiny_store_sinh
    store = tmp_path / "store"
    shutil.copytree(cfg.store_dir, store)
    with np.load(store / "train" / "models.npz") as m:
        n_models, n_rows = len(m["index"]), int(m["n_rows"].sum())
    assert n_models >= 2
    whole = analytic_models(corpus_cfg, sim_cfg, store, "train", 0, n_models, 2)
    cut = n_models // 2
    parts = [analytic_models(corpus_cfg, sim_cfg, store, "train", a, b, 2)
             for a, b in ((0, cut), (cut, n_models))]
    for k in ("row", *ARRAYS):
        np.testing.assert_array_equal(whole[k], np.concatenate([p[k] for p in parts]))
    np.testing.assert_array_equal(whole["row"], np.arange(n_rows))

    status = whole["analytic_status"]
    assert set(status.tolist()) <= set(STATUS_CODE.values())
    stable = status == STATUS_CODE["stable"]
    assert np.isfinite(whole["analytic_pairwise"][stable]).all()
    assert np.isnan(whole["analytic_pairwise"][~stable]).all()
    assert np.isnan(whole["analytic_fcd"][~stable]).all()
    fcd = whole["analytic_fcd"][stable]
    fcd = fcd[np.isfinite(fcd).all(axis=1)]
    assert (np.diff(fcd, axis=1) >= 0).all() and np.allclose(fcd[:, -1], 1.0)
    assert (whole["analytic_decay"][stable] > 0).all()

    write_chunk(store, "train", 0, cut, parts[0], 2)
    with pytest.raises(ValueError, match="without gaps"):
        pack_analytic(store, "train")
    write_chunk(store, "train", cut, n_models, parts[1], 2)
    summary = pack_analytic(store, "train")
    assert summary["rows"] == n_rows and summary["n_surrogates"] == 2
    for k in ARRAYS:
        np.testing.assert_array_equal(np.load(store / "train" / f"{k}.npy"), whole[k])


def test_patch_recomputes_rows_without_an_answer_and_keeps_the_rest(tiny_store_sinh, tmp_path):
    """A packed split whose rows without a stable answer are recomputed and patched equals the
    split computed in one go; the stable rows are untouched; patching twice is refused."""
    cfg, corpus_cfg, sim_cfg = tiny_store_sinh
    store = tmp_path / "store"
    shutil.copytree(cfg.store_dir, store)
    with np.load(store / "train" / "models.npz") as m:
        n_models = len(m["index"])
    whole = analytic_models(corpus_cfg, sim_cfg, store, "train", 0, n_models, 2)
    write_chunk(store, "train", 0, n_models, whole, 2)
    pack_analytic(store, "train")
    # pretend some stable rows had failed before (status none, no values)
    stable = np.flatnonzero(whole["analytic_status"] == STATUS_CODE["stable"])
    assert len(stable) >= 2
    failed = stable[::2]
    root = store / "train"
    for k, fill in (("analytic_status", STATUS_CODE["none"]), ("analytic_decay", np.nan),
                    ("analytic_pairwise", np.nan), ("analytic_fcd", np.nan)):
        a = np.load(root / f"{k}.npy", mmap_mode="r+")
        a[failed] = fill
        a.flush()
    targets = patch_targets(store, "train")
    expected = np.zeros(len(targets), bool)
    expected[failed] = True
    expected[whole["analytic_status"] != STATUS_CODE["stable"]] = True
    np.testing.assert_array_equal(targets, expected)
    cut = n_models // 2
    for a, b in ((0, cut), (cut, n_models)):
        part = analytic_models(corpus_cfg, sim_cfg, store, "train", a, b, 2, targets)
        assert set(part["row"].tolist()) <= set(np.flatnonzero(targets).tolist())
        write_chunk(store, "train", a, b, part, 2, patch=True)
    summary = patch_analytic(store, "train")
    assert summary["patched_rows"] == int(targets.sum())
    for k in ARRAYS:
        np.testing.assert_array_equal(np.load(root / f"{k}.npy"), whole[k])
    assert (np.load(root / "analytic_status_v1.npy")[failed] == STATUS_CODE["none"]).all()
    with pytest.raises(ValueError, match="patched already"):
        patch_analytic(store, "train")
