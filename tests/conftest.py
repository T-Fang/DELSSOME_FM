from pathlib import Path

import pytest

from delssome_fm.config import DataConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"


@pytest.fixture(scope="session")
def data_cfg() -> DataConfig:
    return load_config(DATA_CONFIG_PATH, DataConfig)


def data_available() -> bool:
    """True when the real HCP-YA data is reachable (on the lab cluster or its NAS)."""
    cfg = load_config(DATA_CONFIG_PATH, DataConfig)
    return cfg.subject_list.is_file() and cfg.reference_fc_dir.is_dir()


# A tiny corpus (12 candidates, short simulations) packed into a training store, shared by
# tests/test_pack.py and tests/test_network.py. Building it takes a few seconds.
TINY_SIM = dict(dt=0.001, tr=0.72, n_frames=40, burn_in_frames=5, divergence_bound=1e6,
                fcd_window=10, fcd_bins=100)
TINY_CANDIDATES = 12


@pytest.fixture(scope="session")
def tiny_store(tmp_path_factory):
    """(TrainingDataConfig, CorpusConfig, SimConfig) of a packed tiny train split; the val
    split is a copy of it, so a data module can be built."""
    import shutil

    import numpy as np

    from delssome_fm.config import CorpusConfig, SimConfig, SplitSizes, TrainingDataConfig
    from delssome_fm.sim.corpus import run_candidates, write_chunk
    from delssome_fm.train.pack import normalisation, pack_network_fc, pack_split

    root = tmp_path_factory.mktemp("tiny")
    corpus_cfg = CorpusConfig(seed=11, n_param_sets=3, first_draw=0, batch_size=2, screen_draws=2,
                              screen_frames=40, screen_burn_in_frames=5, flat_rel_sd=1e-6,
                              initial_state_scale=0.1, fcd_store_stride=10,
                              parameter_policy="multiplier",
                              output_dir=root / "corpus")
    sim_cfg = SimConfig(**TINY_SIM)
    scs = np.random.default_rng(0).uniform(0, 0.02, (64, 68, 68))
    logs, arrays, cards = run_candidates(corpus_cfg, sim_cfg, "train", scs, 0, TINY_CANDIDATES)
    write_chunk(corpus_cfg, "train", 0, TINY_CANDIDATES, logs, arrays, cards)
    n_valid = sum(r.valid for r in logs)
    cfg = TrainingDataConfig(store_dir=root / "store",
                             n_models=SplitSizes(train=n_valid, val=n_valid, test=n_valid))
    pack_split(corpus_cfg, sim_cfg, cfg, "train", scs)
    pack_network_fc(cfg.store_dir / "train")
    normalisation(cfg.store_dir)
    shutil.copytree(cfg.store_dir / "train", cfg.store_dir / "val")
    shutil.copytree(cfg.store_dir / "train", cfg.store_dir / "test")
    return cfg, corpus_cfg, sim_cfg
