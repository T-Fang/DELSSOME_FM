"""Train stage 1 and evaluate the result, in one job (scripts/train_stage1.py calls `fit`).

`fit` builds the data module, the Stage1Module, the MLflow logger and checkpointing, trains,
then runs `final_evaluation` on the best checkpoint (lowest val/total) on the same GPU, so
the report needs no separate evaluation job:

    final/<split>/<metric>   for split = train (the run's own training rows), val, test, each
                             a seeded subset of final_eval_rows rows: MSE per group, total,
                             r2_pairwise, and the median raw-unit cost terms by regime
    final_eval.npz           (artifact) per-model total MSE and coupled share for val and
                             test; a few test examples' predicted and simulated FC and CDF

Test is evaluated only here, once, on the checkpoint that val selected; it never steers
training or checkpoint choice.
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
from delssome_fm.train.data import Composition, Stage1DataModule, TrainingStore
from delssome_fm.train.evaluate import (evaluation_rows, load_stage1, per_model, row_metrics,
                                       summarise_rows)
from delssome_fm.train.report import predict_fc_fcd
from delssome_fm.train.stage1 import Stage1Module


def fit(cfg: Stage1Config, config_path: Path, run_name: str | None = None) -> Path:
    """Train stage 1 as configured and evaluate the best checkpoint; returns the run's
    checkpoint directory."""
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")   # TF32 tensor cores for the float32 parts
    run_name = run_name or datetime.now().strftime("stage1_%Y%m%d_%H%M%S")
    composition = Composition(cfg.train_models, cfg.draws_per_model,
                              cfg.coupled_share if cfg.oversample_coupled else None,
                              cfg.coupled_threshold)
    data = Stage1DataModule(cfg.store_dir, cfg.batch_size, cfg.val_rows, cfg.num_workers,
                            cfg.seed, composition)
    data.setup()
    norm = data.train.norm
    module = Stage1Module(cfg, data.train.n_fcd_levels,
                          (float(norm["pairwise_mean"]), float(norm["pairwise_sd"])))
    logger = MLFlowLogger(experiment_name=cfg.mlflow_experiment, run_name=run_name,
                          tracking_uri=cfg.mlflow_tracking_uri,
                          artifact_location=cfg.mlflow_artifact_dir.as_uri())
    checkpoints = cfg.output_dir / run_name
    checkpoints.mkdir(parents=True, exist_ok=True)
    effective = checkpoints / "config.yaml"     # the config file plus any --set overrides
    effective.write_text(f"# from {config_path}\n" + to_yaml(cfg))
    logger.experiment.log_artifact(logger.run_id, str(effective))
    logger.log_hyperparams({"train_rows": len(data.train_rows)})
    best = ModelCheckpoint(dirpath=checkpoints, monitor="val/total", save_top_k=3,
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
    final_evaluation(load_stage1(Path(best.best_model_path)).to(device), data, logger, checkpoints)
    return checkpoints


def final_evaluation(module: Stage1Module, data: Stage1DataModule, logger: MLFlowLogger,
                     out: Path) -> dict[str, dict[str, float]]:
    """Evaluate `module` on train (its own rows), val and test; log final/<split>/<metric>
    and the final_eval.npz artifact (module docstring)."""
    cfg = module.cfg
    splits = {"train": (data.train, data.train_rows), "val": (data.val, None),
              "test": (TrainingStore(cfg.store_dir, "test"), None)}
    summary, arrays = {}, {}
    for split, (store, candidates) in splits.items():
        rows = evaluation_rows(np.arange(len(store)) if candidates is None else candidates, cfg)
        r = row_metrics(module, store, rows, num_workers=cfg.num_workers)
        summary[split] = summarise_rows(r)
        logger.log_metrics({f"final/{split}/{k}": v for k, v in summary[split].items()})
        if split != "train":
            arrays.update({f"{split}_{k}": v for k, v in per_model(r).items()})
        if split == "test":
            picks = _examples(r)
            arrays["test_example_rows"] = rows[picks]
            p = predict_fc_fcd(module, store, rows[picks])
            arrays.update({f"test_example_{k}": v for k, v in p.items()})
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
