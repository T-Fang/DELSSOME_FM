"""Train the FC autoencoder (nn/fc_autoencoder.py) on the FC of a training store, and load it
back for stage 1's "latent" FC head (2026-10-25).

Data: the pairwise target of the stage-1 store, the stored arctanh FC z-scored with the
store's pooled statistics (pack.normalisation), so the autoencoder works in exactly the units
stage 1 predicts. Training uses every row of the train split (the autoencoder sees only FC);
validation a fixed seeded subset of `val_rows` val rows. Loss: mean squared error.

Logged to MLflow: train/mse; per validation val/mse, val/r2 (over all rows and edges) and the
DELSSOME FC cost terms between reconstruction and simulation in raw units (train/costs.py),
as medians over uncoupled and coupled draws: val/cost_<corr|mean>_<regime>. The checkpoint
with the lowest val/mse is kept.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from delssome_fm.config import FCAutoencoderConfig, to_yaml
from delssome_fm.nn.fc_autoencoder import N_EDGES, FCAutoencoder
from delssome_fm.train.costs import cost_terms, raw_fc


class PairwiseRows(Dataset):
    """One split's z-scored pairwise targets; indexing with a list of rows gives (B, 2278)."""

    def __init__(self, store_dir: Path, split: str):
        self.x = np.load(Path(store_dir) / split / "pairwise.npy", mmap_mode="r")
        with np.load(Path(store_dir) / "normalisation.npz") as n:
            self.mean, self.sd = float(n["pairwise_mean"]), float(n["pairwise_sd"])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, rows: list[int]) -> torch.Tensor:
        x = np.asarray(self.x[np.sort(np.asarray(rows))], np.float32)
        return torch.from_numpy((x - self.mean) / self.sd)


class FCAutoencoderModule(L.LightningModule):
    def __init__(self, cfg: FCAutoencoderConfig, pairwise_norm: tuple[float, float],
                 coupled_threshold: float = 0.05):
        super().__init__()
        self.save_hyperparameters({"latent_dim": cfg.latent_dim, "hidden": list(cfg.hidden),
                                   "pairwise_mean": pairwise_norm[0],
                                   "pairwise_sd": pairwise_norm[1],
                                   "store_dir": str(cfg.store_dir)})
        self.cfg, self.pairwise_norm = cfg, pairwise_norm
        self.coupled_threshold = coupled_threshold
        self.model = FCAutoencoder(cfg.latent_dim, cfg.hidden)
        self._val: list[dict[str, torch.Tensor]] = []

    def training_step(self, x: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = torch.mean((self.model(x).float() - x) ** 2)
        self.log("train/mse", loss, prog_bar=True, batch_size=len(x))
        return loss

    def validation_step(self, x: torch.Tensor, batch_idx: int) -> None:
        y = self.model(x).float()
        fc_true, fc_pred = raw_fc(x, *self.pairwise_norm), raw_fc(y, *self.pairwise_norm)
        zero = torch.zeros(len(x), 1, device=x.device)
        c = cost_terms(fc_pred, fc_true, zero, zero)
        self._val.append({"sse": ((y - x) ** 2).sum(1), "sum": x.sum(1), "sumsq": (x ** 2).sum(1),
                          "corr": c["corr"], "mean": c["mean"],
                          "coupled": fc_true.mean(1) > self.coupled_threshold})

    def on_validation_epoch_end(self) -> None:
        v = {k: torch.cat([b[k] for b in self._val]) for k in self._val[0]}
        self._val.clear()
        n = len(v["sse"]) * N_EDGES                     # values over all rows and edges
        mean = v["sum"].sum() / n
        self.log("val/mse", v["sse"].sum() / n, prog_bar=True)
        self.log("val/r2", 1 - v["sse"].sum() / (v["sumsq"].sum() - n * mean ** 2))
        for regime, mask in (("uncoupled", ~v["coupled"]), ("coupled", v["coupled"])):
            if mask.any():
                self.log_dict({f"val/cost_{t}_{regime}": torch.nanmedian(v[t][mask])
                               for t in ("corr", "mean")})

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


def fit_autoencoder(cfg: FCAutoencoderConfig, config_path: Path, run_name: str | None = None
                    ) -> Path:
    """Train as configured; returns the path of the best checkpoint (lowest val/mse)."""
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    run_name = run_name or datetime.now().strftime("fc_ae_%Y%m%d_%H%M%S")
    train, val = PairwiseRows(cfg.store_dir, "train"), PairwiseRows(cfg.store_dir, "val")
    rng = np.random.default_rng(cfg.seed)
    val_rows = np.sort(rng.choice(len(val), min(cfg.val_rows, len(val)), replace=False))
    loader = lambda ds, batches: DataLoader(ds, sampler=batches, batch_size=None,   # noqa: E731
                                            num_workers=cfg.num_workers,
                                            pin_memory=torch.cuda.is_available(),
                                            persistent_workers=cfg.num_workers > 0)
    train_batches = BatchSampler(RandomSampler(range(len(train)), generator=torch.Generator()
                                               .manual_seed(cfg.seed)), cfg.batch_size, True)
    val_batches = [[int(val_rows[i]) for i in b] for b in
                   BatchSampler(SequentialSampler(range(len(val_rows))), cfg.batch_size, False)]
    module = FCAutoencoderModule(cfg, (train.mean, train.sd))
    logger = MLFlowLogger(experiment_name=cfg.mlflow_experiment, run_name=run_name,
                          tracking_uri=cfg.mlflow_tracking_uri,
                          artifact_location=cfg.mlflow_artifact_dir.as_uri())
    out = cfg.output_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    effective = out / "config.yaml"
    effective.write_text(f"# from {config_path}\n" + to_yaml(cfg))
    logger.experiment.log_artifact(logger.run_id, str(effective))
    best = ModelCheckpoint(dirpath=out, monitor="val/mse", save_top_k=1, save_last=True,
                           filename="step{step}", auto_insert_metric_name=False)
    trainer = L.Trainer(max_steps=cfg.max_steps, val_check_interval=cfg.val_every_steps,
                        check_val_every_n_epoch=None, precision=cfg.precision, logger=logger,
                        gradient_clip_val=cfg.gradient_clip,
                        log_every_n_steps=cfg.log_every_steps, callbacks=[best])
    trainer.fit(module, loader(train, train_batches), loader(val, val_batches))
    return Path(best.best_model_path)


def load_fc_autoencoder(path: Path) -> tuple[FCAutoencoder, tuple[float, float]]:
    """The trained autoencoder of a checkpoint, and the (mean, SD) of arctanh FC it was
    trained in (its store's pooled statistics)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    h = ckpt["hyper_parameters"]
    model = FCAutoencoder(h["latent_dim"], tuple(h["hidden"]))
    model.load_state_dict({k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
                           if k.startswith("model.")})
    return model, (float(h["pairwise_mean"]), float(h["pairwise_sd"]))
