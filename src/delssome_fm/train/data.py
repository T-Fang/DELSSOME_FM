"""Stage-1 data: read the training store written by train/pack.py and serve batches.

`TrainingStore` opens one split's arrays (memory-mapped, so many loader workers can share
them) and builds a whole batch from a list of row indices: inputs, the models' equation
graphs padded to the batch's largest graph, and the z-scored targets.

`Stage1DataModule` gives Lightning a shuffled train loader and a fixed validation subset.
The loader samples batches of row indices and each worker builds one batch at a time, so a
batch is read with one sorted fancy-index per array rather than B separate reads.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import (BatchSampler, DataLoader, Dataset, SequentialSampler,
                              SubsetRandomSampler)

from delssome_fm.nn.template_encoder import GraphBatch
from delssome_fm.train.pack import TARGETS

Z_SCORED = ("regional", "pairwise", "fc_moments")   # fcd is a raw CDF (pack.normalisation)


@dataclass
class Stage1Batch:
    """theta (B, N, P) and psi (B, K): log10 multipliers, 0 where the model has no such
    parameter; sc (B, N, N); graph: the rows' DAGs; targets: name -> tensor (pack.TARGETS)."""

    theta: torch.Tensor
    psi: torch.Tensor
    sc: torch.Tensor
    graph: GraphBatch
    targets: dict[str, torch.Tensor]

    def pin_memory(self) -> "Stage1Batch":
        """Called by DataLoader(pin_memory=True) for faster copies to the GPU."""
        pin = lambda x: x.pin_memory()                                    # noqa: E731
        graph = GraphBatch(**{f.name: pin(getattr(self.graph, f.name))
                              for f in dataclasses.fields(GraphBatch)})
        return Stage1Batch(pin(self.theta), pin(self.psi), pin(self.sc), graph,
                           {k: pin(v) for k, v in self.targets.items()})


class TrainingStore(Dataset):
    """One split of the store. Indexing with a list of row indices returns a Stage1Batch."""

    def __init__(self, store_dir: Path, split: str):
        root = Path(store_dir) / split
        if not (root / "summary.json").is_file():
            raise FileNotFoundError(f"{root}: no packed split (run scripts/pack_training_data.py)")
        self.rows = {k: np.load(root / f"{k}.npy", mmap_mode="r")
                     for k in ("model", "theta", "psi", *TARGETS)}
        with np.load(root / "models.npz") as m:
            self.models = {k: m[k] for k in m.files}
        self.node_features = np.load(root / "node_features.npy", mmap_mode="r")
        self.node_degree = np.load(root / "node_degree.npy", mmap_mode="r")
        self.distances = np.load(root / "distances.npy", mmap_mode="r")
        self.sc = np.load(root / "sc.npy")
        with np.load(Path(store_dir) / "normalisation.npz") as n:
            self.norm = {k: n[k] for k in n.files}

    def __len__(self) -> int:
        return len(self.rows["model"])

    @property
    def n_fcd_levels(self) -> int:
        return self.rows["fcd"].shape[1]

    def __getitem__(self, rows: list[int]) -> Stage1Batch:
        idx = np.sort(np.asarray(rows))
        model = self.rows["model"][idx]
        targets = {k: torch.from_numpy(self._target(k, self.rows[k][idx])) for k in TARGETS}
        return Stage1Batch(theta=_log10_or_zero(self.rows["theta"][idx]),
                           psi=_log10_or_zero(self.rows["psi"][idx]),
                           sc=torch.from_numpy(self.sc[self.models["sc_group"][model]]),
                           graph=self._graphs(model), targets=targets)

    def _target(self, name: str, x: np.ndarray) -> np.ndarray:
        if name not in Z_SCORED:
            return np.asarray(x, np.float32)
        return ((x - self.norm[f"{name}_mean"]) / self.norm[f"{name}_sd"]).astype(np.float32)

    def _graphs(self, model: np.ndarray) -> GraphBatch:
        """Each row's DAG, padded to the largest one in the batch."""
        offsets, dist_offsets = self.models["node_offset"], self.models["distance_offset"]
        sizes = offsets[model + 1] - offsets[model]
        B, M = len(model), int(sizes.max())
        features = np.zeros((B, M, self.node_features.shape[1]), np.float32)
        degree = np.zeros((B, M, 2), np.int16)            # int16 keeps batches small;
        distances = np.full((B, M, M), -1, np.int16)      # the encoder casts to long
        for b, (m, n) in enumerate(zip(model, sizes)):
            features[b, :n] = self.node_features[offsets[m]:offsets[m + 1]]
            degree[b, :n] = self.node_degree[offsets[m]:offsets[m + 1]]
            distances[b, :n, :n] = self.distances[dist_offsets[m]:dist_offsets[m + 1]].reshape(n, n)
        roles = {k: torch.from_numpy(self.models[k][model].astype(np.int64))
                 for k in ("regional_par", "global_par", "deriv_roots", "agg", "out_root")}
        return GraphBatch(features=torch.from_numpy(features), degree=torch.from_numpy(degree),
                          distances=torch.from_numpy(distances),
                          node_mask=torch.from_numpy(np.arange(M)[None, :] < sizes[:, None]),
                          **roles)


def _log10_or_zero(x: np.ndarray) -> torch.Tensor:
    """log10 of the parameter multipliers; NaN padding (absent parameters) becomes 0."""
    with np.errstate(invalid="ignore"):
        out = np.log10(np.asarray(x, np.float32))
    return torch.from_numpy(np.nan_to_num(out, nan=0.0))


def subset_rows(store: TrainingStore, n_models: int, draws_per_model: int) -> np.ndarray:
    """Rows of the first `n_models` models of the store, at most `draws_per_model` of each
    (their first usable draws). Rows are ordered by model, then draw (train/pack.py)."""
    n_rows = store.models["n_rows"]
    if not 0 < n_models <= len(n_rows):
        raise ValueError(f"train_models must be in 1..{len(n_rows)}, found {n_models}")
    start = np.concatenate([[0], np.cumsum(n_rows)])[:n_models]
    take = np.minimum(n_rows[:n_models], draws_per_model)
    return np.concatenate([np.arange(s, s + t) for s, t in zip(start, take)])


class Stage1DataModule(L.LightningDataModule):
    """Shuffled train batches (seeded) over a subset of models and draws (`subset_rows`); a
    fixed random subset of `val_rows` val rows."""

    def __init__(self, store_dir: Path, batch_size: int, val_rows: int, num_workers: int,
                 seed: int, train_models: int, draws_per_model: int):
        super().__init__()
        self.store_dir, self.batch_size, self.val_rows = Path(store_dir), batch_size, val_rows
        self.num_workers, self.seed = num_workers, seed
        self.train_models, self.draws_per_model = train_models, draws_per_model

    def setup(self, stage: str | None = None) -> None:
        self.train = TrainingStore(self.store_dir, "train")
        self.train_rows = subset_rows(self.train, self.train_models, self.draws_per_model)
        self.val = TrainingStore(self.store_dir, "val")
        rng = np.random.default_rng(self.seed)
        self.val_subset = np.sort(rng.choice(len(self.val), min(self.val_rows, len(self.val)),
                                             replace=False))

    def train_dataloader(self) -> DataLoader:
        sampler = SubsetRandomSampler(self.train_rows.tolist(),
                                      generator=torch.Generator().manual_seed(self.seed))
        return self._loader(self.train, BatchSampler(sampler, self.batch_size, drop_last=True))

    def val_dataloader(self) -> DataLoader:
        order = SequentialSampler(self.val_subset.tolist())
        batches = [[int(self.val_subset[i]) for i in b]
                   for b in BatchSampler(order, self.batch_size, drop_last=False)]
        return self._loader(self.val, batches)

    def _loader(self, store: TrainingStore, batches) -> DataLoader:
        # batch_size=None: the sampler yields whole batches of row indices, and
        # TrainingStore.__getitem__ turns each into a Stage1Batch
        return DataLoader(store, sampler=batches, batch_size=None, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=self.num_workers > 0)
