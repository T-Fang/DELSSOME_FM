"""Numbers behind the stage-1 report figures (scripts/stage1_report_data.py makes them,
scripts/plot_stage1_report.py draws them).

`training_composition` recovers which rows a run trained on, so its in-sample error can be
measured. `predict_fc_fcd` maps a model's predictions back to the units of a simulation: raw
FC (tanh of the de-z-scored arctanh FC) and the FCD CDF at the stored levels. `cost_terms` is
the original DELSSOME cost between two such sets of statistics (docs/data.md §7): 1 - r of
the FC upper triangles, d = |difference of their means|, and KS = max |CDF difference|, here
over the 100 stored FCD levels (data.md §8: within 0.0004 of the 10,000-bin KS).
"""

from __future__ import annotations

import numpy as np
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from torch.utils.data import DataLoader

from delssome_fm.train.data import Composition, TrainingStore, subset_rows
from delssome_fm.train.stage1 import Stage1Module


def training_composition(module: Stage1Module) -> Composition:
    """The models and draws `module` was trained on, from its recorded config."""
    c = module.cfg
    return Composition(c.train_models, c.draws_per_model,
                       c.coupled_share if c.oversample_coupled else None, c.coupled_threshold)


def training_rows(module: Stage1Module) -> np.ndarray:
    """Row indices of the train split of module.cfg.store_dir that `module` trained on."""
    return subset_rows(TrainingStore(module.cfg.store_dir, "train"), training_composition(module))


@torch.no_grad()
def predict_fc_fcd(module: Stage1Module, store: TrainingStore, rows: np.ndarray,
                   batch_size: int = 512) -> dict[str, np.ndarray]:
    """For each row: predicted and simulated FC upper triangles (raw r, (R, E)) and FCD CDFs
    ((R, L)). The store must be one z-scored like the module's training store."""
    device = next(module.parameters()).device
    edges = torch.arange(len(module.edges), device=device)
    mean, sd = store.norm["pairwise_mean"], store.norm["pairwise_sd"]
    out = {k: [] for k in ("fc_pred", "fc_true", "cdf_pred", "cdf_true")}
    batches = [rows[i:i + batch_size].tolist() for i in range(0, len(rows), batch_size)]
    for batch in DataLoader(store, sampler=batches, batch_size=None):
        batch = move_data_to_device(batch, device)
        pred = module.predict_statistics(batch, edges)
        for name, z in (("fc_pred", pred.pairwise), ("fc_true", batch.targets["pairwise"])):
            out[name].append(np.tanh(z.float().cpu().numpy() * sd + mean))
        out["cdf_pred"].append(pred.fcd.float().cpu().numpy())
        out["cdf_true"].append(batch.targets["fcd"].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def cost_terms(fc_a: np.ndarray, fc_b: np.ndarray, cdf_a: np.ndarray,
               cdf_b: np.ndarray) -> dict[str, np.ndarray]:
    """Row-wise DELSSOME cost terms between two sets of statistics: fc (R, E) raw FC upper
    triangles, cdf (R, L) CDFs. Returns corr (1 - r), mean (|mean difference|), ks."""
    a = fc_a - fc_a.mean(1, keepdims=True)
    b = fc_b - fc_b.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return {"corr": 1 - r, "mean": np.abs(fc_a.mean(1) - fc_b.mean(1)),
            "ks": np.abs(cdf_a - cdf_b).max(1)}


# ---------------------------------------------------------------- the report's data


def scaling_metrics(module: Stage1Module, max_rows: int, seed: int = 0) -> dict[str, dict]:
    """evaluate() of one run on its own training rows, on val and on test (max_rows each)."""
    from delssome_fm.train.evaluate import evaluate
    store = module.cfg.store_dir
    return {"train": evaluate(module, store, "train", max_rows=max_rows, seed=seed,
                              candidates=training_rows(module)),
            "val": evaluate(module, store, "val", max_rows=max_rows, seed=seed),
            "test": evaluate(module, store, "test", max_rows=max_rows, seed=seed)}


def prediction_vs_simulation(module: Stage1Module, split: str, max_rows: int, seed: int = 0,
                             coupled_threshold: float = 0.05) -> dict[str, np.ndarray]:
    """Cost terms between the model's predictions and the simulations on `split`, and between
    a reference prediction (the average FC and CDF of the training rows) and the simulations.
    Also keeps four examples: coupled draws at the 10th, 50th and 90th percentile of the
    model's 1 - r, and the median uncoupled draw."""
    store = TrainingStore(module.cfg.store_dir, split)
    rows = np.sort(np.random.default_rng(seed).choice(len(store), min(max_rows, len(store)),
                                                      replace=False))
    p = predict_fc_fcd(module, store, rows)
    train = TrainingStore(module.cfg.store_dir, "train")
    ref_rows = np.sort(np.random.default_rng(seed).choice(training_rows(module), max_rows,
                                                          replace=False))
    ref = predict_fc_fcd(module, train, ref_rows)
    fc_ref = np.broadcast_to(ref["fc_true"].mean(0), p["fc_true"].shape)
    cdf_ref = np.broadcast_to(ref["cdf_true"].mean(0), p["cdf_true"].shape)
    model = cost_terms(p["fc_pred"], p["fc_true"], p["cdf_pred"], p["cdf_true"])
    reference = cost_terms(fc_ref, p["fc_true"], cdf_ref, p["cdf_true"])
    true_mean_fc = p["fc_true"].mean(1)
    coupled = true_mean_fc > coupled_threshold
    order = np.flatnonzero(coupled)[np.argsort(model["corr"][coupled])]
    uncoupled = np.flatnonzero(~coupled)[np.argsort(model["corr"][~coupled])]
    picks = [order[int(q * (len(order) - 1))] for q in (0.1, 0.5, 0.9)]
    picks.append(uncoupled[len(uncoupled) // 2])
    out = {f"model_{k}": v for k, v in model.items()}
    out.update({f"reference_{k}": v for k, v in reference.items()})
    out.update(true_mean_fc=true_mean_fc, example_rows=rows[picks],
               **{f"example_{k}": v[picks] for k, v in p.items()})
    return out
