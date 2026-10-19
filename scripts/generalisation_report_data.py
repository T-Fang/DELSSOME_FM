"""Per-model errors of one stage-1 checkpoint in each cell of the generalisation diagnostic
(sim/generalisation.py), for the 2 x 2 figure of scripts/plot_stage1_report.py. Writes
<out> (.npz): <cell>_total (per-model total MSE, each model weighted equally) and <cell>_index
(corpus index). In the seen-model cells only models the checkpoint actually trained on are
kept (a model sampled from the 100-draw corpus may have no usable draw among a 10-draw run's
draws 0-9).
The store must be z-scored with the checkpoint's own training-store statistics:
outputs/generalisation/store (original store) or store_d10 (10-draw store).

    python scripts/generalisation_report_data.py --checkpoint outputs/stage1/stage1_v1/step170000.ckpt \\
        --store-dir outputs/generalisation/store --out outputs/report/generalisation_stage1_v1.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from delssome_fm.sim.generalisation import CELLS
from delssome_fm.train.data import TrainingStore
from delssome_fm.train.evaluate import load_stage1, per_model_errors
from delssome_fm.train.report import training_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    module = load_stage1(args.checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
    train = TrainingStore(module.cfg.store_dir, "train")
    trained_rows = np.unique(np.asarray(train.rows["model"])[training_rows(module)])
    trained = set(train.models["index"][trained_rows].tolist())
    arrays = {}
    for cell, (model_split, _) in CELLS.items():
        e = per_model_errors(module, args.store_dir, cell)
        index = np.load(args.store_dir / cell / "models.npz")["index"][e["model"]]
        keep = np.array([i in trained for i in index]) if model_split == "train" else \
            np.ones(len(index), bool)
        arrays.update({f"{cell}_total": e["total"][keep], f"{cell}_index": index[keep]})
        print(f"{cell}: {keep.sum()} models (of {len(keep)}), mean total {e['total'][keep].mean():.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, checkpoint=str(args.checkpoint), **arrays)


if __name__ == "__main__":
    main()
