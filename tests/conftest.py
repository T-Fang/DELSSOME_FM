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
    return _tiny_store(tmp_path_factory, "multiplier")


@pytest.fixture(scope="session")
def tiny_store_sinh(tmp_path_factory):
    """As tiny_store, from a sinh-policy corpus (the parameter-token backbone needs one)."""
    return _tiny_store(tmp_path_factory, "sinh")


def _tiny_store(tmp_path_factory, policy: str):
    import shutil

    import numpy as np

    from delssome_fm.config import CorpusConfig, SimConfig, SplitSizes, TrainingDataConfig
    from delssome_fm.sim.corpus import run_candidates, write_chunk
    from delssome_fm.train.pack import normalisation, pack_network_fc, pack_split

    root = tmp_path_factory.mktemp(f"tiny_{policy}")
    corpus_cfg = CorpusConfig(seed=11, n_param_sets=3, first_draw=0, batch_size=2, screen_draws=2,
                              screen_frames=40, screen_burn_in_frames=5, flat_rel_sd=1e-6,
                              initial_state_scale=0.1, fcd_store_stride=10,
                              parameter_policy=policy, gain_scope="draw", rate_floor=0.0,
                              gain_prior="absolute",
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


@pytest.fixture(scope="session")
def tiny_cma_store(tiny_store, tmp_path_factory):
    """(store_dir, CmaesConfig, CorpusConfig, SimConfig): a copy of tiny_store with CMA-ES
    draws packed from fake chunks (train/pack.pack_cma_split, 7 per model, the first 2 of each
    generation; with the network FC). The last model was skipped by CMA-ES (no candidates, a
    log row saying so). Each other model's candidates are its base draws repeated as
    3 generations of 4,
    candidate 0 marked diverged, in two chunks. val_cma and test_cma are copies of train_cma,
    as val and test are copies of train."""
    import json
    import shutil

    import numpy as np

    from delssome_fm.config import CmaesConfig, SplitSizes
    from delssome_fm.sim.cmaes import write_plan
    from delssome_fm.train.pack import pack_cma_split, pack_network_fc

    _, corpus_cfg, sim_cfg = tiny_store
    root = tmp_path_factory.mktemp("tiny_cma")
    store = root / "store"
    shutil.copytree(tiny_store[0].store_dir, store)
    members = np.load(store / "train" / "models.npz")["index"].tolist()
    assert len(members) >= 2, "the tiny corpus needs two valid models"
    cfg = CmaesConfig(corpus_config=root / "corpus.yaml", n_models=SplitSizes(len(members), 1, 1),
                      coupled_threshold=0.05, max_coupled_share=1.0, max_runs=2,
                      min_coupled_share=0.5, population=4, generations=3, sigma0=0.05,
                      failed_cost=5.0, output_dir=root / "cma")
    write_plan(cfg, "train", [{"index": i, "usable": 1, "coupled": 1} for i in members])
    (chunk,) = (corpus_cfg.output_dir / "train").glob("candidates*_draws0-*.npz")
    d = dict(np.load(chunk))
    per_draw = [k for k in d if k != "draw" and d[k].shape[:1] == d["model"].shape]

    def fake(i):
        at = np.resize(np.flatnonzero(d["model"] == i), 12)
        out = {k: d[k][at].copy() for k in per_draw}
        out["diverged"][0] = True
        out.update(run=np.zeros(12, np.int64), generation=np.repeat(np.arange(3), 4),
                   member=np.tile(np.arange(4), 3), level=np.zeros((12, 5), np.float32),
                   spread=np.zeros((12, 5), np.float32), cost=np.zeros((12, 3), np.float32))
        return out

    for first, stop in ((0, 1), (1, len(members))):
        ran = [i for i in members[first:stop] if i != members[-1]]
        path = cfg.output_dir / "train" / f"plan{first:06d}-{stop:06d}.npz"
        parts = [fake(i) for i in ran]
        np.savez(path, **({k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                          if parts else {}))
        path.with_suffix(".log.jsonl").write_text("".join(
            json.dumps({"index": i, "kept": i in ran} | ({} if i in ran else {"skipped": "test"}))
            + "\n" for i in members[first:stop]))
    pack_cma_split(cfg, corpus_cfg, sim_cfg, store, "train", per_model=7, per_generation=2)
    pack_network_fc(store / "train_cma")           # the base splits have the network FC too
    for split in ("val", "test"):
        shutil.copytree(store / "train_cma", store / f"{split}_cma")
        summary = json.loads((store / f"{split}_cma" / "summary.json").read_text())
        summary.update(split=f"{split}_cma", base_split=split)
        (store / f"{split}_cma" / "summary.json").write_text(json.dumps(summary))
    return store, cfg, corpus_cfg, sim_cfg
