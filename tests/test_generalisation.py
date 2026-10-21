"""sim/generalisation.py: SC choice per cell, and that the two SC cells of a model share its
draws (a paired comparison)."""

import numpy as np

from conftest import TINY_SIM
from delssome_fm.config import CorpusConfig, GeneralisationConfig, SimConfig
from delssome_fm.sim.corpus import sc_group
from delssome_fm.sim.generalisation import CELLS, cell_sc_group, simulate_cell


def _cfgs(tmp_path):
    corpus = CorpusConfig(seed=11, n_param_sets=3, first_draw=0, batch_size=100, screen_draws=2,
                          screen_frames=40, screen_burn_in_frames=5, flat_rel_sd=1e-6,
                          initial_state_scale=0.1, fcd_store_stride=10,
                          parameter_policy="multiplier", output_dir=tmp_path)
    gen = GeneralisationConfig(output_dir=tmp_path / "gen", seed=3, n_models=2, first_draw=100,
                               n_draws=2)
    return corpus, gen


def test_own_split_keeps_the_corpus_sc_and_the_other_split_is_seeded(tmp_path):
    corpus, _ = _cfgs(tmp_path)
    for i in range(50):
        assert cell_sc_group(corpus, "seen_model_seen_sc", i, 64) == sc_group(corpus, "train", i, 64)
        assert cell_sc_group(corpus, "unseen_model_unseen_sc", i, 14) == sc_group(corpus, "val", i, 14)
    swapped = [cell_sc_group(corpus, "seen_model_unseen_sc", i, 14) for i in range(300)]
    assert swapped == [cell_sc_group(corpus, "seen_model_unseen_sc", i, 14) for i in range(300)]
    assert set(swapped) == set(range(14))


def test_both_sc_cells_of_a_model_share_its_draws(tmp_path):
    corpus, gen = _cfgs(tmp_path)
    sim = SimConfig(**TINY_SIM)
    rng = np.random.default_rng(0)
    scs = {"train": rng.uniform(0, 0.02, (64, 68, 68)), "val": rng.uniform(0, 0.02, (14, 68, 68))}
    out = {}
    for cell in ("seen_model_seen_sc", "seen_model_unseen_sc"):
        path = simulate_cell(corpus, sim, gen, cell, [0, 1], scs[CELLS[cell][1]],
                             tmp_path / cell / "chunk")
        out[cell] = np.load(path)
    a, b = out["seen_model_seen_sc"], out["seen_model_unseen_sc"]
    assert a["draw"].tolist() == [100, 101, 100, 101] and a["model"].tolist() == [0, 0, 1, 1]
    np.testing.assert_array_equal(a["theta"], b["theta"])
    assert (tmp_path / "seen_model_unseen_sc" / "chunk.cards.jsonl").is_file()
