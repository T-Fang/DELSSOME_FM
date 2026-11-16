"""train/analytic.py on the tiny sinh store: the rows do not depend on the chunking, the
arrays hold what each status promises, and packing accepts exactly a complete set of chunks."""

import shutil

import numpy as np
import pytest

from delssome_fm.train.analytic import (ARRAYS, STATUS_CODE, analytic_models, pack_analytic,
                                        write_chunk)


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
