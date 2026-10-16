"""Stage 1 (architecture.md §5): pretrain the backbone to predict simulated summary
statistics of the synthetic corpus.

`Stage1Module` is the LightningModule: backbone + stage-1 heads, the uncertainty-weighted
loss, AdamW with linear warm-up and cosine decay. `fit` wires it to the data, MLflow and
checkpointing; scripts/train_stage1.py only parses arguments and calls it.

Logged to MLflow, per step: train/<group> MSE, train/total, train/s_<group> (the learned
loss scales), the learning rate. Per validation: val/<group> MSE over the fixed val subset,
val/total (their sum), val/r2_pairwise (R^2 of the z-scored arctanh FC over all edges).
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from delssome_fm.config import Stage1Config
from delssome_fm.nn.backbone import Backbone, BackboneSize
from delssome_fm.nn.heads import Stage1Heads
from delssome_fm.train.data import Stage1Batch, Stage1DataModule
from delssome_fm.train.losses import GROUPS, UncertaintyWeighting, group_losses

N_REGIONS = 68


def upper_triangle(n: int = N_REGIONS) -> torch.Tensor:
    """(n(n-1)/2, 2) the (i, j), i < j, pairs in the order of the stored pairwise targets."""
    return torch.triu_indices(n, n, offset=1).T


class Stage1Module(L.LightningModule):
    def __init__(self, cfg: Stage1Config, n_fcd_levels: int):
        super().__init__()
        self.save_hyperparameters({**_flat(cfg), "n_fcd_levels": n_fcd_levels})
        self.cfg = cfg
        self.backbone = Backbone(BackboneSize(
            n_regions=N_REGIONS, d_model=cfg.d_model, d_template=cfg.d_template,
            template_layers=cfg.template_layers, backbone_layers=cfg.backbone_layers,
            n_heads=cfg.n_heads, ffn_multiplier=cfg.ffn_multiplier, dropout=cfg.dropout,
            max_distance=cfg.max_distance, max_degree=cfg.max_degree))
        self.heads = Stage1Heads(cfg.d_model, cfg.fc_rank, n_fcd_levels, cfg.fc_offset)
        self.weighting = UncertaintyWeighting(GROUPS)
        self.register_buffer("edges", upper_triangle(), persistent=False)
        self.val_mse = torch.nn.ModuleDict({g: torchmetrics.MeanSquaredError() for g in GROUPS})
        self.val_r2_pairwise = torchmetrics.R2Score()

    def predict_statistics(self, batch: Stage1Batch, edge_index: torch.Tensor):
        out = self.backbone(batch.theta, batch.psi, batch.sc, batch.graph)
        return self.heads(out, self.edges[edge_index])

    def training_step(self, batch: Stage1Batch, batch_idx: int) -> torch.Tensor:
        # a fresh random subset of FC edges each step (architecture.md §5.2)
        edge_index = torch.randperm(len(self.edges), device=self.device)[:self.cfg.edges_per_step]
        pred = self.predict_statistics(batch, edge_index)
        losses = group_losses(pred, batch.targets, edge_index)
        total = self.weighting(losses)
        B = len(batch.theta)
        self.log_dict({f"train/{g}": v for g, v in losses.items()}, batch_size=B)
        self.log("train/total", total, prog_bar=True, batch_size=B)
        self.log_dict({f"train/s_{g}": s for g, s in self.weighting.scales().items()}, batch_size=B)
        return total

    def validation_step(self, batch: Stage1Batch, batch_idx: int) -> None:
        all_edges = torch.arange(len(self.edges), device=self.device)
        pred = self.predict_statistics(batch, all_edges)
        for g in GROUPS:
            self.val_mse[g].update(getattr(pred, g).float().flatten(), batch.targets[g].flatten())
        self.val_r2_pairwise.update(pred.pairwise.float().flatten(),
                                    batch.targets["pairwise"].flatten())

    def on_validation_epoch_end(self) -> None:
        mse = {g: self.val_mse[g].compute() for g in GROUPS}
        self.log_dict({f"val/{g}": v for g, v in mse.items()})
        self.log("val/total", torch.stack(list(mse.values())).sum(), prog_bar=True)
        self.log("val/r2_pairwise", self.val_r2_pairwise.compute())
        for metric in [*self.val_mse.values(), self.val_r2_pairwise]:
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate,
                                      weight_decay=self.cfg.weight_decay)
        warmup, total = self.cfg.warmup_steps, self.cfg.max_steps

        def factor(step: int) -> float:
            """Linear warm-up to the full rate, then cosine decay to 0 at max_steps."""
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def fit(cfg: Stage1Config, config_path: Path, run_name: str | None = None) -> Path:
    """Train stage 1 as configured; returns the run's checkpoint directory."""
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")   # TF32 tensor cores for the float32 parts
    run_name = run_name or datetime.now().strftime("stage1_%Y%m%d_%H%M%S")
    data = Stage1DataModule(cfg.store_dir, cfg.batch_size, cfg.val_rows, cfg.num_workers,
                            cfg.seed)
    data.setup()
    module = Stage1Module(cfg, data.train.n_fcd_levels)
    logger = MLFlowLogger(experiment_name=cfg.mlflow_experiment, run_name=run_name,
                          tracking_uri=cfg.mlflow_tracking_uri,
                          artifact_location=cfg.mlflow_artifact_dir.as_uri())
    logger.experiment.log_artifact(logger.run_id, str(config_path))
    checkpoints = cfg.output_dir / run_name
    trainer = L.Trainer(
        max_steps=cfg.max_steps, val_check_interval=cfg.val_every_steps,
        check_val_every_n_epoch=None, precision=cfg.precision,
        gradient_clip_val=cfg.gradient_clip, logger=logger,
        log_every_n_steps=cfg.log_every_steps,
        callbacks=[ModelCheckpoint(dirpath=checkpoints, monitor="val/total", save_top_k=3,
                                   save_last=True, filename="step{step}",
                                   auto_insert_metric_name=False),
                   LearningRateMonitor(logging_interval="step")])
    trainer.fit(module, datamodule=data)
    return checkpoints


def _flat(cfg: Stage1Config) -> dict:
    """The config as plain values, for Lightning's hyperparameters and MLflow params."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()}
