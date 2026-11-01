"""Coupling classification (architecture.md §7.8, 2026-11-01): can the backbone tell a
coupled draw from an uncoupled one? A draw is coupled when its simulated mean off-diagonal
FC exceeds coupled_threshold (0.05), the regime split used throughout §7.

`CouplingModule` is the stage-1 backbone (any token scheme; the test uses raw-theta parameter
tokens) with a single head, MLP([d, d, 1])(h_CLS) -> the logit of "coupled", trained on binary
cross-entropy alone: no summary-statistic, cost or other loss (config task: coupling, with
stat_loss, cost_loss, catch24_loss and fc_net_loss false; the FC-head fields are unused).
The label is computed from the batch's pairwise target (raw FC = tanh of the de-z-scored
arctanh FC, averaged over the upper triangle), as Stage1Module.raw_costs labels the regimes.

Logged to MLflow: train/bce and train/accuracy per step; per validation, over the fixed val
subset, val/<metric> for the metrics of `classification_metrics`, and val/selection = val/bce,
which picks the checkpoint. After training (train/fit.py), the best checkpoint is evaluated
on train (its own rows), val and test as final/<split>/<metric>; final_eval.npz keeps each
evaluated row's logit, label, simulated mean FC and model, for analysis by mean FC.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric.utilities.apply_func import move_data_to_device
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_auroc

from delssome_fm.config import Stage1Config, config_from_dict
from delssome_fm.nn.backbone import Backbone
from delssome_fm.nn.layers import MLP
from delssome_fm.train.costs import raw_fc
from delssome_fm.train.data import Stage1Batch, TrainingStore
from delssome_fm.train.stage1 import _flat, adamw_cosine, backbone_size

TASKS = ("statistics", "coupling")
OTHER_LOSSES = ("stat_loss", "cost_loss", "catch24_loss", "fc_net_loss")


def check_task(cfg: Stage1Config) -> None:
    """Raise unless cfg.task is known and, for coupling, no other loss is switched on."""
    if cfg.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, found {cfg.task!r}")
    if cfg.task == "coupling":
        on = [f for f in OTHER_LOSSES if getattr(cfg, f)]
        if on:
            raise ValueError(f"task coupling trains the classifier alone: set {on} to false")
        if cfg.coupled_draws_only:
            raise ValueError("task coupling needs both classes: coupled_draws_only must be false")


class CouplingModule(L.LightningModule):
    def __init__(self, cfg: Stage1Config, pairwise_norm: tuple[float, float]):
        """pairwise_norm: (mean, SD) of arctanh FC in the training store, to label rows from
        their pairwise target (kept with the checkpoint)."""
        super().__init__()
        check_task(cfg)
        if cfg.task != "coupling":
            raise ValueError(f"CouplingModule is the coupling task, the config has {cfg.task!r}")
        self.save_hyperparameters({**_flat(cfg), "pairwise_mean": float(pairwise_norm[0]),
                                   "pairwise_sd": float(pairwise_norm[1])})
        self.cfg = cfg
        self.pairwise_norm = (float(pairwise_norm[0]), float(pairwise_norm[1]))
        self.backbone = Backbone(backbone_size(cfg))
        self.head = MLP([cfg.d_model, cfg.d_model, 1])
        self._val: list[tuple[torch.Tensor, torch.Tensor]] = []

    def logits(self, batch: Stage1Batch) -> torch.Tensor:
        """(B,) the logit of "coupled" for each row."""
        out = self.backbone(batch.theta, batch.psi, batch.sc, batch.graph)
        return self.head(out.h_cls).squeeze(-1).float()

    def mean_fc(self, batch: Stage1Batch) -> torch.Tensor:
        """(B,) each row's simulated mean off-diagonal FC, in raw units."""
        return raw_fc(batch.targets["pairwise"], *self.pairwise_norm).mean(1)

    def labels(self, batch: Stage1Batch) -> torch.Tensor:
        """(B,) True where the simulated draw is coupled."""
        return self.mean_fc(batch) > self.cfg.coupled_threshold

    def training_step(self, batch: Stage1Batch, batch_idx: int) -> torch.Tensor:
        logit, label = self.logits(batch), self.labels(batch)
        loss = F.binary_cross_entropy_with_logits(logit, label.float())
        B = len(logit)
        self.log("train/bce", loss, prog_bar=True, batch_size=B)
        self.log("train/accuracy", ((logit > 0) == label).float().mean(), batch_size=B)
        return loss

    def validation_step(self, batch: Stage1Batch, batch_idx: int) -> None:
        self._val.append((self.logits(batch), self.labels(batch)))

    def on_validation_epoch_end(self) -> None:
        logit = torch.cat([v[0] for v in self._val])
        label = torch.cat([v[1] for v in self._val])
        self._val.clear()
        metrics = classification_metrics(logit, label)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})
        self.log("val/selection", metrics["bce"])

    def configure_optimizers(self):
        return adamw_cosine(self.parameters(), self.cfg)


def classification_metrics(logit: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
    """logit (R,), label (R,) bool -> BCE, accuracy, the recall of each class (coupled draws
    called coupled; uncoupled called uncoupled), balanced accuracy (their mean), AUROC, and
    the coupled share. Recalls, balanced accuracy and AUROC need both classes: NaN without."""
    logit, label = logit.float(), label.bool()
    called = logit > 0
    n_coupled, n_uncoupled = int(label.sum()), int((~label).sum())
    nan = float("nan")
    recall_c = float((called & label).sum()) / n_coupled if n_coupled else nan
    recall_u = float((~called & ~label).sum()) / n_uncoupled if n_uncoupled else nan
    both = n_coupled > 0 and n_uncoupled > 0
    return {"bce": float(F.binary_cross_entropy_with_logits(logit, label.float())),
            "accuracy": float((called == label).float().mean()),
            "recall_coupled": recall_c, "recall_uncoupled": recall_u,
            "balanced_accuracy": (recall_c + recall_u) / 2,
            "auroc": float(binary_auroc(logit, label.long())) if both else nan,
            "coupled_share": n_coupled / len(label)}


def load_coupling(checkpoint: Path) -> CouplingModule:
    """The CouplingModule of a Lightning checkpoint, rebuilt from the config it recorded."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    hp.pop("train_rows", None)           # logged for information, not a config field
    norm = (hp.pop("pairwise_mean"), hp.pop("pairwise_sd"))
    module = CouplingModule(config_from_dict(Stage1Config, hp, str(checkpoint)), norm)
    module.load_state_dict(ckpt["state_dict"])
    return module.eval()


@torch.no_grad()
def classify_rows(module: CouplingModule, store: TrainingStore, rows: np.ndarray,
                  batch_size: int = 512, num_workers: int = 0) -> dict[str, np.ndarray]:
    """One pass over `rows` (sorted) of `store`: per row the logit, the label, the simulated
    mean FC and the model."""
    rows = np.sort(np.asarray(rows))
    device = next(module.parameters()).device
    out = {k: [] for k in ("logit", "label", "mean_fc")}
    batches = [rows[i:i + batch_size].tolist() for i in range(0, len(rows), batch_size)]
    for batch in DataLoader(store, sampler=batches, batch_size=None, num_workers=num_workers):
        batch = move_data_to_device(batch, device)
        out["logit"].append(module.logits(batch))
        out["label"].append(module.labels(batch))
        out["mean_fc"].append(module.mean_fc(batch))
    res = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    res["row"] = rows
    res["model"] = np.asarray(store.rows["model"][rows])
    return res
