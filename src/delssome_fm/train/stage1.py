"""Stage 1 (architecture.md §5): pretrain the backbone to predict simulated summary
statistics of the synthetic corpus.

`Stage1Module` is the LightningModule: backbone + stage-1 heads, the uncertainty-weighted
loss, AdamW with linear warm-up and cosine decay. train/fit.py wires it to the data, MLflow
and checkpointing, and evaluates the best checkpoint after training.

Logged to MLflow, per step: train/<group> MSE, train/total, train/s_<group> (the learned
loss scales), the learning rate. Per validation, over the fixed val subset: val/<group> MSE,
val/total (their sum), val/r2_pairwise (R^2 of the z-scored arctanh FC over all edges), and
the DELSSOME cost terms between prediction and simulation in raw units (train/costs.py), as
medians over uncoupled and coupled draws: val/cost_<corr|mean|ks>_<uncoupled|coupled>.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import lightning as L
import torch
import torchmetrics

from delssome_fm.config import Stage1Config
from delssome_fm.nn.backbone import Backbone, BackboneSize
from delssome_fm.nn.heads import Stage1Heads
from delssome_fm.train.costs import TERMS, cost_terms, raw_fc
from delssome_fm.train.data import Stage1Batch
from delssome_fm.train.losses import COST_GROUPS, GROUPS, UncertaintyWeighting, group_losses

N_REGIONS = 68


def upper_triangle(n: int = N_REGIONS) -> torch.Tensor:
    """(n(n-1)/2, 2) the (i, j), i < j, pairs in the order of the stored pairwise targets."""
    return torch.triu_indices(n, n, offset=1).T


class Stage1Module(L.LightningModule):
    def __init__(self, cfg: Stage1Config, n_fcd_levels: int, pairwise_norm: tuple[float, float]):
        """pairwise_norm: (mean, SD) of arctanh FC in the training store, to map predictions
        back to raw FC (kept with the checkpoint)."""
        super().__init__()
        self.save_hyperparameters({**_flat(cfg), "n_fcd_levels": n_fcd_levels,
                                   "pairwise_mean": float(pairwise_norm[0]),
                                   "pairwise_sd": float(pairwise_norm[1])})
        self.cfg = cfg
        self.pairwise_norm = (float(pairwise_norm[0]), float(pairwise_norm[1]))
        self.backbone = Backbone(BackboneSize(
            n_regions=N_REGIONS, d_model=cfg.d_model, d_template=cfg.d_template,
            template_layers=cfg.template_layers, backbone_layers=cfg.backbone_layers,
            n_heads=cfg.n_heads, ffn_multiplier=cfg.ffn_multiplier, dropout=cfg.dropout,
            max_distance=cfg.max_distance, max_degree=cfg.max_degree))
        self.heads = Stage1Heads(cfg.d_model, cfg.fc_rank, n_fcd_levels, cfg.fc_offset,
                                 cfg.pair_head, cfg.pair_hidden)
        self.weighting = UncertaintyWeighting(GROUPS + (COST_GROUPS if cfg.cost_loss else ()))
        self.register_buffer("edges", upper_triangle(), persistent=False)
        self.val_mse = torch.nn.ModuleDict({g: torchmetrics.MeanSquaredError() for g in GROUPS})
        self.val_r2_pairwise = torchmetrics.R2Score()
        self._val_costs: list[dict[str, torch.Tensor]] = []

    def predict_statistics(self, batch: Stage1Batch, edge_index: torch.Tensor):
        out = self.backbone(batch.theta, batch.psi, batch.sc, batch.graph)
        return self.heads(out, self.edges[edge_index])

    def training_step(self, batch: Stage1Batch, batch_idx: int) -> torch.Tensor:
        if self.cfg.cost_loss:
            # 1 - r and d are defined on the whole FC upper triangle: predict every edge
            edge_index = torch.arange(len(self.edges), device=self.device)
        else:
            # a fresh random subset of FC edges each step (architecture.md §5.2)
            edge_index = torch.randperm(len(self.edges), device=self.device)[:self.cfg.edges_per_step]
        pred = self.predict_statistics(batch, edge_index)
        losses = group_losses(pred, batch.targets, edge_index)
        if self.cfg.cost_loss:
            costs = self.raw_costs(pred, batch, eps=1e-8)
            losses.update({f"cost_{t}": costs[t].mean() for t in TERMS})
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
        self._val_costs.append(self.raw_costs(pred, batch))

    def raw_costs(self, pred, batch: Stage1Batch, eps: float = 0.0) -> dict[str, torch.Tensor]:
        """Per row: the cost terms between prediction and simulation in raw units (all edges
        predicted), and whether the simulated draw is coupled (mean FC > coupled_threshold)."""
        fc_pred = raw_fc(pred.pairwise, *self.pairwise_norm)
        fc_true = raw_fc(batch.targets["pairwise"], *self.pairwise_norm)
        out = cost_terms(fc_pred, fc_true, pred.fcd.float(), batch.targets["fcd"].float(), eps)
        out["coupled"] = fc_true.mean(1) > self.cfg.coupled_threshold
        return out

    def on_validation_epoch_end(self) -> None:
        costs = {k: torch.cat([c[k] for c in self._val_costs]) for k in (*TERMS, "coupled")}
        self._val_costs.clear()
        for regime, mask in (("uncoupled", ~costs["coupled"]), ("coupled", costs["coupled"])):
            if mask.any():
                self.log_dict({f"val/cost_{t}_{regime}": torch.nanmedian(costs[t][mask])
                               for t in TERMS})
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


def _flat(cfg: Stage1Config) -> dict:
    """The config as plain values, for Lightning's hyperparameters and MLflow params."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()}
