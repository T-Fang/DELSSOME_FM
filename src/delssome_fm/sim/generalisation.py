"""Generalisation diagnostic (2026-10-16): is stage 1's train/val gap caused by new equations
or by new connectomes? Simulates the 2 x 2 cells

                         SC of a train group      SC of a val group
    train corpus model   seen_model_seen_sc       seen_model_unseen_sc
    val corpus model     unseen_model_seen_sc     unseen_model_unseen_sc

for a seeded sample of models from each split, with new parameter draws (first_draw onwards,
which stage 1 never saw). A model's own split gives its own SC group; the other split's SC
is a seeded random group of that split. The draws (parameters, initial state, noise) of a
model are the same in both of its cells, so the SC effect is measured within pairs.

Each chunk is written like a corpus chunk (arrays .npz + .cards.jsonl, `sc_group` indexing
the cell's SC split), so train/pack.pack_chunks packs the cells into a training store.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from delssome_fm.config import CorpusConfig, GeneralisationConfig, SimConfig, SplitSizes
from delssome_fm.sim.corpus import (SPLIT_STREAM, card_for, first_valid, sc_group,
                                    simulate_draws)
from delssome_fm.spec.card import card_to_dict
from delssome_fm.spec.compile import compile_card

# cell -> (split of the models, split of the SC)
CELLS: dict[str, tuple[str, str]] = {
    "seen_model_seen_sc": ("train", "train"),
    "seen_model_unseen_sc": ("train", "val"),
    "unseen_model_seen_sc": ("val", "train"),
    "unseen_model_unseen_sc": ("val", "val"),
}
_SWAP = 4   # stream tag of the other-split SC choice (corpus.py uses 1-3)


def sample_models(cfg: GeneralisationConfig, members: list[int], split: str) -> list[int]:
    """`cfg.n_models` of a split's corpus models, seeded, in index order."""
    rng = np.random.default_rng([cfg.seed, SPLIT_STREAM[split]])
    return sorted(int(i) for i in rng.choice(members, cfg.n_models, replace=False))


def write_sample(corpus_cfg: CorpusConfig, corpus_sizes: SplitSizes,
                 cfg: GeneralisationConfig) -> dict[str, list[int]]:
    """Sample the train and val models once and write <output_dir>/models.json, so that
    every simulation job and the packing use the same models. corpus_sizes: the corpus size
    of each split (configs/training_data.yaml)."""
    sample = {s: sample_models(cfg, first_valid(corpus_cfg, s, getattr(corpus_sizes, s)), s)
              for s in ("train", "val")}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "models.json").write_text(json.dumps(sample))
    return sample


def read_sample(cfg: GeneralisationConfig) -> dict[str, list[int]]:
    path = cfg.output_dir / "models.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run scripts/submit_generalisation.py first")
    return json.loads(path.read_text())


def cell_sc_group(corpus_cfg: CorpusConfig, cell: str, index: int, n_groups: int) -> int:
    """The SC group a model is simulated with in `cell`, as an index into the SC split."""
    model_split, sc_split = CELLS[cell]
    if model_split == sc_split:
        return sc_group(corpus_cfg, model_split, index, n_groups)
    rng = np.random.default_rng([corpus_cfg.seed, SPLIT_STREAM[model_split], index, _SWAP])
    return int(rng.integers(n_groups))


def simulate_cell(corpus_cfg: CorpusConfig, sim_cfg: SimConfig, cfg: GeneralisationConfig,
                  cell: str, indices: list[int], scs: np.ndarray, path: Path) -> Path:
    """Simulate draws first_draw .. first_draw + n_draws - 1 of `indices` in `cell` and write
    <path>.npz (per-draw arrays with `model`) and <path>.cards.jsonl. scs: (G, N, N), the
    SC split of the cell (corpus.split_scs)."""
    model_split, _ = CELLS[cell]
    one_batch = dataclasses.replace(corpus_cfg, batch_size=cfg.n_draws)
    rows, cards = [], []
    for i in indices:
        card = card_for(corpus_cfg, model_split, i)
        g = cell_sc_group(corpus_cfg, cell, i, len(scs))
        res = simulate_draws(one_batch, sim_cfg, compile_card(card), model_split, i, scs[g],
                             cfg.first_draw, cfg.n_draws)
        res["model"] = np.full(cfg.n_draws, i)
        rows.append(res)
        cards.append({"split": model_split, "index": i, "sc_group": g,
                      "card": card_to_dict(card)})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
    path.with_suffix(".cards.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cards))
    return path.with_suffix(".npz")


def chunk_path(cfg: GeneralisationConfig, cell: str, start: int, stop: int) -> Path:
    """Chunk of the cell's sampled models start..stop-1 (positions in the sample)."""
    return cfg.output_dir / "chunks" / cell / f"models{start:04d}-{stop:04d}"
