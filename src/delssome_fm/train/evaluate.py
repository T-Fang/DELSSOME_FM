"""Evaluate a trained stage-1 model on any packed split of a training store: the per-group
mean squared errors that validation logs (z-scored targets; raw CDF for the FCD), and R^2 of
the pairwise targets. Used for the generalisation diagnostic (sim/generalisation.py).

`per_model_errors` gives each model's error with every model weighted equally, with its mean
FC, so results can be split by regime (coupled vs uncoupled models) and compared between
checkpoints on the same rows; `bootstrap_mean` gives intervals over models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchmetrics
from lightning.fabric.utilities.apply_func import move_data_to_device
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from delssome_fm.config import Stage1Config, config_from_dict
from delssome_fm.train.data import TrainingStore, model_mean_fc
from delssome_fm.train.losses import GROUPS
from delssome_fm.train.stage1 import Stage1Module


# Stage1Config fields added after some checkpoints were written, with the value every
# earlier run had in effect. None of them changes the network's weights or shape except
# fc_offset, which did not exist before it was added (so False).
LEGACY_DEFAULTS = {
    "fc_offset": False,              # added 2026-10-16
    "train_models": 100_000,         # added 2026-10-16: earlier runs used the whole corpus
    "draws_per_model": 100,
    "oversample_coupled": False,     # added 2026-10-17
    "coupled_share": 0.5,            # unused when oversample_coupled is False
    "coupled_threshold": 0.05,
}


def load_stage1(checkpoint: Path) -> Stage1Module:
    """The Stage1Module of a Lightning checkpoint, rebuilt from the config it recorded."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    n_fcd_levels = hp.pop("n_fcd_levels")
    hp.pop("train_rows", None)           # logged for information, not a config field
    hp = {**LEGACY_DEFAULTS, **hp}
    module = Stage1Module(config_from_dict(Stage1Config, hp, str(checkpoint)), n_fcd_levels)
    module.load_state_dict(ckpt["state_dict"])
    return module.eval()


@torch.no_grad()
def evaluate(module: Stage1Module, store_dir: Path, split: str, batch_size: int = 512,
             max_rows: int | None = None, seed: int = 0) -> dict[str, float]:
    """Metrics of `module` on store_dir/split (all rows, or a seeded subset of max_rows)."""
    store = TrainingStore(store_dir, split)
    rows = np.arange(len(store))
    if max_rows is not None and max_rows < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))
    batches = [rows[b].tolist() for b in BatchSampler(SequentialSampler(rows), batch_size, False)]
    device = next(module.parameters()).device
    mse = {g: torchmetrics.MeanSquaredError().to(device) for g in GROUPS}
    r2 = torchmetrics.R2Score().to(device)
    all_edges = torch.arange(len(module.edges), device=device)
    for batch in DataLoader(store, sampler=batches, batch_size=None):
        batch = move_data_to_device(batch, device)
        pred = module.predict_statistics(batch, all_edges)
        for g in GROUPS:
            mse[g].update(getattr(pred, g).float().flatten(), batch.targets[g].flatten())
        r2.update(pred.pairwise.float().flatten(), batch.targets["pairwise"].flatten())
    out = {g: float(m.compute()) for g, m in mse.items()}
    out.update(total=sum(out.values()), r2_pairwise=float(r2.compute()), rows=len(rows))
    return out



# ---------------------------------------------------------------- by model and regime


@torch.no_grad()
def per_model_errors(module: Stage1Module, store_dir: Path, split: str,
                     max_rows: int | None = None, seed: int = 0, batch_size: int = 512,
                     ) -> dict[str, np.ndarray]:
    """Each model's mean squared error per group (and their sum, `total`) over its rows in the
    split, or in a seeded subset of max_rows rows; `model` (store row of each model) and
    `mean_fc` (from model_mean_fc) say which models these are. Every model weighs the same."""
    store = TrainingStore(store_dir, split)
    rows = np.arange(len(store))
    if max_rows is not None and max_rows < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))
    device = next(module.parameters()).device
    all_edges = torch.arange(len(module.edges), device=device)
    errors = {g: [] for g in GROUPS}
    batches = [rows[i:i + batch_size].tolist() for i in range(0, len(rows), batch_size)]
    for batch in DataLoader(store, sampler=batches, batch_size=None):
        batch = move_data_to_device(batch, device)
        pred = module.predict_statistics(batch, all_edges)
        for g in GROUPS:
            d = (getattr(pred, g).float() - batch.targets[g]) ** 2
            errors[g].append(d.reshape(len(d), -1).mean(1).cpu().numpy())
    row_model = np.asarray(store.rows["model"][rows])
    models, inverse = np.unique(row_model, return_inverse=True)
    counts = np.bincount(inverse)
    out = {g: np.bincount(inverse, np.concatenate(v)) / counts for g, v in errors.items()}
    out["total"] = sum(out[g] for g in GROUPS)
    out["model"] = models
    out["mean_fc"] = model_mean_fc(store)[models]
    return out


def bootstrap_mean(x: np.ndarray, n_boot: int = 2000, seed: int = 1) -> tuple[float, float, float]:
    """Mean of x and its 95% bootstrap interval (resampling the entries, here models)."""
    idx = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    boot = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
