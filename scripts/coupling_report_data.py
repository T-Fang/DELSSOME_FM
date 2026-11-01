"""The regime call of statistics models next to the coupling classifier's (train/coupling.py):
on the classifier's evaluated test rows (its final_eval.npz), each statistics checkpoint's
predicted mean FC (train/report.predicted_mean_fc), called coupled when > the classifier's
coupled_threshold. Writes <out>/coupling_regression.npz (rows, simulated mean FC, label,
classifier logit, one pred_mean_fc_<name> per checkpoint) and prints the classification
metrics of each. Run on a GPU (L40S) job:

    python scripts/coupling_report_data.py --classifier outputs/stage1/sinh_ptok_raw_coupling_cls \\
        --checkpoint sinh_ptok_raw_cost=outputs/stage1/sinh_ptok_raw_cost/step185000.ckpt \\
        --out outputs/report_sinh/sinh_ptok_raw_coupling_cls
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from delssome_fm.train.coupling import classification_metrics, load_coupling
from delssome_fm.train.data import TrainingStore
from delssome_fm.train.evaluate import load_stage1
from delssome_fm.train.report import predicted_mean_fc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--classifier", type=Path, required=True, help="the classifier's run directory")
    p.add_argument("--checkpoint", nargs="+", required=True, metavar="NAME=PATH")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_coupling(args.classifier / "last.ckpt").cfg
    with np.load(args.classifier / "final_eval.npz") as f:
        rows, mean_fc, label, logit = (f[f"test_{k}"] for k in ("row", "mean_fc", "label", "logit"))
    store = TrainingStore(cfg.store_dir, "test")
    out = {"rows": rows, "mean_fc": mean_fc, "label": label, "logit": logit}
    scores = {"classifier": logit}
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        module = load_stage1(Path(path)).to(device)
        pred = predicted_mean_fc(module, store, rows)
        out[f"pred_mean_fc_{name}"] = pred
        scores[name] = pred - cfg.coupled_threshold          # > 0: called coupled
    np.savez(args.out / "coupling_regression.npz", **out)
    for name, s in scores.items():
        m = classification_metrics(torch.from_numpy(s), torch.from_numpy(label))
        print(f"{name:28s}", " ".join(f"{k} {v:.3f}" for k, v in m.items() if k != "bce"))


if __name__ == "__main__":
    main()
