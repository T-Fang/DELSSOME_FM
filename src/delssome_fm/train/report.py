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

from pathlib import Path

import numpy as np
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from torch.utils.data import DataLoader

from delssome_fm.train import costs
from delssome_fm.train.data import Composition, TrainingStore, subset_rows
from delssome_fm.train.evaluate import evaluation_rows
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
    ((R, L)). Predictions are de-z-scored with the statistics of the module's training store,
    simulations with those of `store` (they differ slightly between stores)."""
    device = next(module.parameters()).device
    edges = torch.arange(len(module.edges), device=device)
    with np.load(Path(module.cfg.store_dir) / "normalisation.npz") as n:
        own = {"fc_pred": (n["pairwise_mean"], n["pairwise_sd"])}
    own["fc_true"] = (store.norm["pairwise_mean"], store.norm["pairwise_sd"])
    out = {k: [] for k in ("fc_pred", "fc_true", "cdf_pred", "cdf_true")}
    batches = [rows[i:i + batch_size].tolist() for i in range(0, len(rows), batch_size)]
    for batch in DataLoader(store, sampler=batches, batch_size=None):
        batch = move_data_to_device(batch, device)
        pred = module.predict_statistics(batch, edges)
        for name, z in (("fc_pred", pred.pairwise), ("fc_true", batch.targets["pairwise"])):
            mean, sd = own[name]
            out[name].append(np.tanh(z.float().cpu().numpy() * sd + mean))
        out["cdf_pred"].append(pred.fcd.float().cpu().numpy())
        out["cdf_true"].append(batch.targets["fcd"].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def cost_terms(fc_a: np.ndarray, fc_b: np.ndarray, cdf_a: np.ndarray,
               cdf_b: np.ndarray) -> dict[str, np.ndarray]:
    """Row-wise DELSSOME cost terms between two sets of statistics: fc (R, E) raw FC upper
    triangles, cdf (R, L) CDFs, as numpy. Returns corr (1 - r), mean (|mean difference|), ks
    (train/costs.py)."""
    t = [torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64)
         for x in (fc_a, fc_b, cdf_a, cdf_b)]
    return {k: v.numpy() for k, v in costs.cost_terms(*t).items()}


# ---------------------------------------------------------------- the report's data


def scaling_metrics(module: Stage1Module, max_rows: int, seed: int = 0) -> dict[str, dict]:
    """evaluate() of one run on its own training rows, on val and on test (max_rows each)."""
    from delssome_fm.train.evaluate import evaluate
    store = module.cfg.store_dir
    return {"train": evaluate(module, store, "train", max_rows=max_rows, seed=seed,
                              candidates=training_rows(module)),
            "val": evaluate(module, store, "val", max_rows=max_rows, seed=seed),
            "test": evaluate(module, store, "test", max_rows=max_rows, seed=seed)}


def training_average(module: Stage1Module, max_rows: int, seed: int = 0,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """The average simulated FC (E,) and FCD CDF (L,) over a seeded sample of the rows
    `module` trained on: the reference prediction that ignores the model and parameters."""
    rows = training_rows(module)
    rows = np.sort(np.random.default_rng(seed).choice(rows, min(max_rows, len(rows)),
                                                      replace=False))
    ref = predict_fc_fcd(module, TrainingStore(module.cfg.store_dir, "train"), rows)
    return ref["fc_true"].mean(0), ref["cdf_true"].mean(0)


def reference_comparison(module: Stage1Module, store_dir: Path, name: str,
                         fc_avg: np.ndarray, cdf_avg: np.ndarray) -> dict[str, np.ndarray]:
    """Cost terms on one reference model's fitted sets (sim/reference_eval.py): the model's
    prediction vs simulation repeat 0, repeat 1 vs repeat 0 (the noise floor: the same
    parameters, other noise), and the training average vs repeat 0. Only sets usable in both
    repeats count. Also keeps a typical set as an example: the one with the median simulated
    mean FC (so a model's rare degenerate fit, e.g. FIC's set with FC ~ 0, is not shown)."""
    stores = [TrainingStore(store_dir, f"{name}_r{k}") for k in (0, 1)]
    draws = [np.load(Path(store_dir) / f"{name}_r{k}" / "draw.npy") for k in (0, 1)]
    common = np.intersect1d(*draws)
    rows = [np.searchsorted(d, common) for d in draws]   # draws are stored in increasing order
    a = predict_fc_fcd(module, stores[0], rows[0])
    b = predict_fc_fcd(module, stores[1], rows[1])
    model = cost_terms(a["fc_pred"], a["fc_true"], a["cdf_pred"], a["cdf_true"])
    floor = cost_terms(b["fc_true"], a["fc_true"], b["cdf_true"], a["cdf_true"])
    ref = cost_terms(np.broadcast_to(fc_avg, a["fc_true"].shape), a["fc_true"],
                     np.broadcast_to(cdf_avg, a["cdf_true"].shape), a["cdf_true"])
    pick = int(np.argsort(a["fc_true"].mean(1))[len(common) // 2])
    out = {f"model_{k}": v for k, v in model.items()}
    out.update({f"floor_{k}": v for k, v in floor.items()})
    out.update({f"reference_{k}": v for k, v in ref.items()})
    out.update(sets=common, example_set=common[pick], true_mean_fc=a["fc_true"].mean(1),
               **{f"example_{k}": v[pick] for k, v in a.items()})
    return out


def prediction_vs_simulation(module: Stage1Module, split: str, reference_rows: int,
                             seed: int = 0, coupled_threshold: float = 0.05,
                             ) -> dict[str, np.ndarray]:
    """Cost terms between the model's predictions and the simulations on `split`, on the rows
    of the final evaluation (evaluate.evaluation_rows, so they match final/<split>/*), and
    between a reference prediction (the average FC and CDF of `reference_rows` training rows)
    and the simulations.
    Also keeps four examples: draws in the realistic coupled range (coupled_threshold < mean
    FC < 0.9; above it FC is near-uniform synchronisation) at the 10th, 50th and 90th
    percentile of the model's 1 - r, and the median uncoupled draw."""
    store = TrainingStore(module.cfg.store_dir, split)
    rows = evaluation_rows(np.arange(len(store)), module.cfg)
    p = predict_fc_fcd(module, store, rows)
    fc_avg, cdf_avg = training_average(module, reference_rows, seed)
    fc_ref = np.broadcast_to(fc_avg, p["fc_true"].shape)
    cdf_ref = np.broadcast_to(cdf_avg, p["cdf_true"].shape)
    model = cost_terms(p["fc_pred"], p["fc_true"], p["cdf_pred"], p["cdf_true"])
    reference = cost_terms(fc_ref, p["fc_true"], cdf_ref, p["cdf_true"])
    true_mean_fc = p["fc_true"].mean(1)
    coupled = true_mean_fc > coupled_threshold
    realistic = coupled & (true_mean_fc < 0.9)
    order = np.flatnonzero(realistic)[np.argsort(model["corr"][realistic])]
    uncoupled = np.flatnonzero(~coupled)[np.argsort(model["corr"][~coupled])]
    picks = [order[int(q * (len(order) - 1))] for q in (0.1, 0.5, 0.9)]
    picks.append(uncoupled[len(uncoupled) // 2])
    out = {f"model_{k}": v for k, v in model.items()}
    out.update({f"reference_{k}": v for k, v in reference.items()})
    out.update(true_mean_fc=true_mean_fc, example_rows=rows[picks],
               **{f"example_{k}": v[picks] for k, v in p.items()})
    return out
