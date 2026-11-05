"""Train stage 1 and evaluate the result, in one job (scripts/train_stage1.py calls `fit`).

`fit` builds the data module, the Stage1Module, the MLflow logger and checkpointing, trains,
then runs `final_evaluation` on the best checkpoint (lowest val/selection: the summed val
MSE of the groups the model trains on, which is val/total unless catch24 is trained) on the
same GPU, so
the report needs no separate evaluation job:

    final/<split>/<metric>   for split = train (the run's own training rows), val, test, each
                             a seeded subset of final_eval_rows rows (all rows if null): MSE
                             per group, total, r2_pairwise, and the median raw-unit cost
                             terms by regime. Base draws only, so runs with and without
                             CMA-ES draws compare directly.
    final/<split>_cma/...    with cma_draws: the same on the split's CMA-ES draws, and
    final/<split>_pooled/... on base and CMA-ES draws together (2026-11-05)
    final_eval.npz           (artifact) per-model total MSE and coupled share for val and
                             test (and val_cma, test_cma); a few test examples' predicted and
                             simulated FC and CDF

Test is evaluated only here, once, on the checkpoint that val selected; it never steers
training or checkpoint choice.

With task: coupling the module is train/coupling.CouplingModule (val/selection is its val
BCE) and `final_classification` evaluates it: final/<split>/<classification metric>, and
final_eval.npz with each evaluated row's logit, label and mean FC.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from delssome_fm.config import Stage1Config, to_yaml
from delssome_fm.train.coupling import (CouplingModule, check_task, classification_metrics,
                                        classify_rows, load_coupling)
from delssome_fm.train.data import Composition, Stage1DataModule, TrainingStore, coupled_rows
from delssome_fm.train.evaluate import (evaluation_rows, load_stage1, per_model, row_metrics,
                                       summarise_rows)
from delssome_fm.train.report import predict_fc_fcd
from delssome_fm.train.stage1 import Stage1Module


def fit(cfg: Stage1Config, config_path: Path, run_name: str | None = None) -> Path:
    """Train stage 1 as configured and evaluate the best checkpoint; returns the run's
    checkpoint directory."""
    check_task(cfg)
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")   # TF32 tensor cores for the float32 parts
    run_name = run_name or datetime.now().strftime("stage1_%Y%m%d_%H%M%S")
    composition = Composition(cfg.train_models, cfg.draws_per_model,
                              cfg.coupled_share if cfg.oversample_coupled else None,
                              cfg.coupled_threshold, cfg.coupled_draws_only)
    data = Stage1DataModule(cfg.store_dir, cfg.batch_size, cfg.val_rows, cfg.num_workers,
                            cfg.seed, composition, cma=cfg.cma_draws)
    data.setup()
    norm = data.train.norm
    pairwise_norm = (float(norm["pairwise_mean"]), float(norm["pairwise_sd"]))
    module = (CouplingModule(cfg, pairwise_norm) if cfg.task == "coupling"
              else Stage1Module(cfg, data.train.n_fcd_levels, pairwise_norm))
    logger = MLFlowLogger(experiment_name=cfg.mlflow_experiment, run_name=run_name,
                          tracking_uri=cfg.mlflow_tracking_uri,
                          artifact_location=cfg.mlflow_artifact_dir.as_uri())
    checkpoints = cfg.output_dir / run_name
    checkpoints.mkdir(parents=True, exist_ok=True)
    effective = checkpoints / "config.yaml"     # the config file plus any --set overrides
    effective.write_text(f"# from {config_path}\n" + to_yaml(cfg))
    logger.experiment.log_artifact(logger.run_id, str(effective))
    logger.log_hyperparams({"train_rows": len(data.train_rows)})
    if cfg.token_scheme == "parameter" and (data.train.parameter_policy != "sinh"
                                            or data.train.node_value is None):
        raise ValueError(f"the parameter-token backbone reads constants as values on the "
                         f"parameters' scale: it needs a sinh-policy store with node_value.npy "
                         f"(scripts/pack_node_values.py); {cfg.store_dir} has policy "
                         f"{data.train.parameter_policy!r}, node values "
                         f"{'present' if data.train.node_value is not None else 'absent'}")
    for flag, target in ((cfg.catch24_loss, "catch24"), (cfg.fc_net_loss, "fc_net")):
        if flag and target not in data.train.targets:
            raise ValueError(f"{target} is to be trained, but {cfg.store_dir} has no {target} "
                             f"targets")
    best = ModelCheckpoint(dirpath=checkpoints, monitor="val/selection", save_top_k=3,
                           save_last=True, filename="step{step}", auto_insert_metric_name=False)
    trainer = L.Trainer(
        max_steps=cfg.max_steps, val_check_interval=cfg.val_every_steps,
        check_val_every_n_epoch=None, precision=cfg.precision,
        gradient_clip_val=cfg.gradient_clip, logger=logger,
        log_every_n_steps=cfg.log_every_steps,
        callbacks=[best, LearningRateMonitor(logging_interval="step")])
    trainer.fit(module, datamodule=data)
    # Lightning moves the module back to the CPU when fit ends, so module.device is no longer
    # the training device: choose the GPU explicitly (else this runs on 1 CPU thread for hours)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.task == "coupling":
        final_classification(load_coupling(Path(best.best_model_path)).to(device), data,
                             logger, checkpoints)
        return checkpoints
    final = load_stage1(Path(best.best_model_path)).to(device)
    final_evaluation(final, data, logger, checkpoints)
    final_evaluation(final, data, logger, checkpoints, coupled=True)
    return checkpoints


def final_evaluation(module: Stage1Module, data: Stage1DataModule, logger: MLFlowLogger,
                     out: Path, coupled: bool = False) -> dict[str, dict[str, float]]:
    """Evaluate `module` on train (its own rows), val and test; log final/<split>/<metric>
    and the final_eval.npz artifact (module docstring). coupled=True: only the coupled draws
    of each pool (draw mean FC > coupled_threshold), logged as final_coupled/<split>/<metric>
    and final_eval_coupled.npz, so that a model trained on all draws and one trained on
    coupled draws only are compared on the same rows. A coupled-only model's final/* rows are
    already coupled: val and test are restricted to coupled draws for it either way."""
    cfg = module.cfg
    prefix = "final_coupled" if coupled else "final"
    splits = {"train": (data.train, data.train_rows), "val": (data.val, None),
              "test": (TrainingStore(cfg.store_dir, "test", cma=cfg.cma_draws), None)}
    summary, arrays = {}, {}
    for split, (store, candidates) in splits.items():
        pool = np.arange(len(store)) if candidates is None else candidates
        if coupled or cfg.coupled_draws_only:
            pool = coupled_rows(store, pool, cfg.coupled_threshold)
        rows = evaluation_rows(pool, cfg)        # sorted: row_metrics keeps this order
        r = row_metrics(module, store, rows, num_workers=cfg.num_workers)
        cma = rows >= store.n_base
        parts = {split: ~cma}
        if cma.any():
            parts.update({f"{split}_cma": cma, f"{split}_pooled": np.ones_like(cma)})
        for name, mask in parts.items():
            if not mask.any():
                continue
            part = {k: v[mask] for k, v in r.items()}
            summary[name] = summarise_rows(part)
            logger.log_metrics({f"{prefix}/{name}/{k}": v for k, v in summary[name].items()})
            if split != "train" and not name.endswith("_pooled"):
                arrays.update({f"{name}_{k}": v for k, v in per_model(part).items()})
        if split == "test" and (~cma).any():
            base_rows = rows[~cma]
            picks = _examples({k: v[~cma] for k, v in r.items()})
            arrays["test_example_rows"] = base_rows[picks]
            p = predict_fc_fcd(module, store, base_rows[picks])
            arrays.update({f"test_example_{k}": v for k, v in p.items()})
    path = out / f"{prefix.replace('final', 'final_eval')}.npz"
    np.savez(path, **arrays)
    logger.experiment.log_artifact(logger.run_id, str(path))
    return summary


def final_classification(module: CouplingModule, data: Stage1DataModule, logger: MLFlowLogger,
                         out: Path) -> dict[str, dict[str, float]]:
    """Evaluate a coupling classifier on train (its own rows), val and test, each a seeded
    subset of final_eval_rows rows (evaluation_rows: the same val and test rows as a
    statistics run with the same seed); log final/<split>/<metric> and final_eval.npz."""
    cfg = module.cfg
    splits = {"train": (data.train, data.train_rows), "val": (data.val, None),
              "test": (TrainingStore(cfg.store_dir, "test"), None)}
    summary, arrays = {}, {}
    for split, (store, candidates) in splits.items():
        pool = np.arange(len(store)) if candidates is None else candidates
        r = classify_rows(module, store, evaluation_rows(pool, cfg), num_workers=cfg.num_workers)
        summary[split] = classification_metrics(torch.from_numpy(r["logit"]),
                                                torch.from_numpy(r["label"]))
        summary[split]["rows"] = len(r["logit"])
        logger.log_metrics({f"final/{split}/{k}": v for k, v in summary[split].items()})
        arrays.update({f"{split}_{k}": v for k, v in r.items()})
    path = out / "final_eval.npz"
    np.savez(path, **arrays)
    logger.experiment.log_artifact(logger.run_id, str(path))
    return summary


def _examples(r: dict[str, np.ndarray]) -> list[int]:
    """Coupled rows at the 10th, 50th and 90th percentile of 1 - r, and the median uncoupled
    row by 1 - r (positions into the evaluated rows)."""
    picks = []
    coupled = np.flatnonzero(r["coupled"] & np.isfinite(r["corr"]))
    if len(coupled):
        order = coupled[np.argsort(r["corr"][coupled])]
        picks += [int(order[int(q * (len(order) - 1))]) for q in (0.1, 0.5, 0.9)]
    uncoupled = np.flatnonzero(~r["coupled"] & np.isfinite(r["corr"]))
    if len(uncoupled):
        picks.append(int(uncoupled[np.argsort(r["corr"][uncoupled])][len(uncoupled) // 2]))
    return picks
