"""Evaluate a trained stage-1 model on any packed split of a training store: the per-group
mean squared errors that validation logs (z-scored targets; raw CDF for the FCD), and R^2 of
the pairwise targets. Used for the generalisation diagnostic (sim/generalisation.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchmetrics
from lightning.fabric.utilities.apply_func import move_data_to_device
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from delssome_fm.config import Stage1Config, config_from_dict
from delssome_fm.train.data import TrainingStore
from delssome_fm.train.losses import GROUPS
from delssome_fm.train.stage1 import Stage1Module


def load_stage1(checkpoint: Path) -> Stage1Module:
    """The Stage1Module of a Lightning checkpoint, rebuilt from the config it recorded."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    n_fcd_levels = hp.pop("n_fcd_levels")
    hp.setdefault("fc_offset", False)   # checkpoints written before 2026-10-16 had no offset
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

