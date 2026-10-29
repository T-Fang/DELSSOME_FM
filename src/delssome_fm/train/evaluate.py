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
from delssome_fm.train.losses import GROUPS, OPTIONAL_GROUPS
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
    "pair_head": "bilinear",         # added 2026-10-19: the only head before
    "pair_hidden": 128,              # unused by the bilinear head
    "final_eval_rows": 200_000,      # added 2026-10-19; evaluation only
    "cost_loss": False,              # added 2026-10-19
    "fc_autoencoder": "none",        # added 2026-10-25, with the "latent" pair head
    "stat_loss": True,               # added 2026-10-25: every earlier run trained on them
    "catch24_loss": False,           # added 2026-10-25
    "fc_net_loss": False,            # added 2026-10-26
    "coupled_draws_only": False,     # added 2026-10-26
    "token_scheme": "region",        # added 2026-10-29: the only scheme before
}


def evaluation_rows(pool: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """The rows of `pool` that a run is evaluated on after training: a subset of
    cfg.final_eval_rows drawn with the run's seed, sorted. The final evaluation and the report
    figures both use it, so their numbers come from the same rows; for val and test the pool
    is the whole split, so every run with the same seed sees the same rows."""
    pool = np.asarray(pool)
    rng = np.random.default_rng(cfg.seed)
    return np.sort(rng.choice(pool, min(cfg.final_eval_rows, len(pool)), replace=False))


def load_stage1(checkpoint: Path) -> Stage1Module:
    """The Stage1Module of a Lightning checkpoint, rebuilt from the config it recorded."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    n_fcd_levels = hp.pop("n_fcd_levels")
    hp.pop("train_rows", None)           # logged for information, not a config field
    norm = (hp.pop("pairwise_mean", None), hp.pop("pairwise_sd", None))
    cfg = config_from_dict(Stage1Config, {**LEGACY_DEFAULTS, **hp}, str(checkpoint))
    if norm[0] is None:                  # checkpoints before 2026-10-19: from the store
        with np.load(cfg.store_dir / "normalisation.npz") as n:
            norm = (float(n["pairwise_mean"]), float(n["pairwise_sd"]))
    module = Stage1Module(cfg, n_fcd_levels, norm)
    module.load_state_dict(ckpt["state_dict"])
    return module.eval()


@torch.no_grad()
def evaluate(module: Stage1Module, store_dir: Path, split: str, batch_size: int = 512,
             max_rows: int | None = None, seed: int = 0,
             candidates: np.ndarray | None = None) -> dict[str, float]:
    """Metrics of `module` on store_dir/split: on all rows (or `candidates`, e.g. a run's own
    training rows), or on a seeded subset of max_rows of them."""
    store = TrainingStore(store_dir, split)
    rows = np.arange(len(store)) if candidates is None else np.asarray(candidates)
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


# ---------------------------------------------------------------- one pass, every number


def check_parameter_policy(module: Stage1Module, store: TrainingStore) -> None:
    """Raise unless `store` encodes parameters as the module's training store does: a
    log10-multiplier input means nothing to a network trained on sinh-encoded values."""
    trained = TrainingStore(module.cfg.store_dir, "train").parameter_policy
    if store.parameter_policy != trained:
        raise ValueError(f"the module was trained on {trained!r}-policy parameters, the "
                         f"evaluated store holds {store.parameter_policy!r}-policy ones")


@torch.no_grad()
def row_metrics(module: Stage1Module, store: TrainingStore, rows: np.ndarray,
                batch_size: int = 512, num_workers: int = 0) -> dict[str, np.ndarray]:
    """One pass over `rows` of `store`. Per row: the MSE of each group (z-scored units; raw
    CDF for the FCD), the pairwise sums needed for R^2 over all rows and edges (squared error,
    target sum and sum of squares), the raw-unit cost terms between prediction and simulation
    (train/costs.py), whether the simulated draw is coupled, and its model."""
    check_parameter_policy(module, store)
    rows = np.sort(np.asarray(rows))     # a batch is read sorted; keep rows in that order
    device = next(module.parameters()).device
    edges = torch.arange(len(module.edges), device=device)
    keys = (*GROUPS, "sse_pairwise", "sum_pairwise", "sumsq_pairwise", "corr", "mean", "ks",
            "coupled")
    extra = tuple(g for g in OPTIONAL_GROUPS
                  if getattr(module.heads, g) is not None and g in store.targets)
    out = {k: [] for k in keys + extra}
    batches = [rows[i:i + batch_size].tolist() for i in range(0, len(rows), batch_size)]
    for batch in DataLoader(store, sampler=batches, batch_size=None, num_workers=num_workers):
        batch = move_data_to_device(batch, device)
        pred = module.predict_statistics(batch, edges)
        for g in GROUPS + extra:
            d = (getattr(pred, g).float() - batch.targets[g].float()) ** 2
            out[g].append(d.reshape(len(d), -1).mean(1))
        t = batch.targets["pairwise"].float()
        out["sse_pairwise"].append(((pred.pairwise.float() - t) ** 2).sum(1))
        out["sum_pairwise"].append(t.sum(1))
        out["sumsq_pairwise"].append((t ** 2).sum(1))
        for k, v in module.raw_costs(pred, batch).items():
            out[k].append(v)
    res = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    res["model"] = np.asarray(store.rows["model"][rows])
    return res


def summarise_rows(r: dict[str, np.ndarray], n_edges: int = 2278) -> dict[str, float]:
    """Row metrics -> the logged numbers: MSE per group and their total (the four summary
    statistic groups, so totals compare across models), catch24 MSE when the model has the
    head, pairwise R^2, and the median cost terms over uncoupled and coupled rows."""
    out = {g: float(r[g].mean()) for g in GROUPS + tuple(g for g in OPTIONAL_GROUPS if g in r)}
    out["total"] = sum(out[g] for g in GROUPS)
    n = len(r["sse_pairwise"]) * n_edges
    sst = r["sumsq_pairwise"].sum() - r["sum_pairwise"].sum() ** 2 / n
    out["r2_pairwise"] = float(1 - r["sse_pairwise"].sum() / sst)
    for regime, mask in (("uncoupled", ~r["coupled"]), ("coupled", r["coupled"])):
        out[f"rows_{regime}"] = int(mask.sum())
        for t in ("corr", "mean", "ks"):
            if mask.any():
                out[f"cost_{t}_{regime}"] = float(np.nanmedian(r[t][mask]))
    return out


def per_model(r: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Row metrics -> each model's mean total MSE and mean coupledness (models weigh the same)."""
    models, inverse = np.unique(r["model"], return_inverse=True)
    counts = np.bincount(inverse)
    total = sum(r[g] for g in GROUPS)
    return {"model": models, "total": np.bincount(inverse, total) / counts,
            "coupled_share": np.bincount(inverse, r["coupled"].astype(float)) / counts}
