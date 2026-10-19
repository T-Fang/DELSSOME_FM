"""The data behind one model's report figures: cost terms between its predictions and the
simulations on test draws, with the training-average reference and examples
(train/report.prediction_vs_simulation), and the reference-model comparison
(train/report.reference_comparison). Writes <out>/report_data.npz and <out>/comparison.npz,
which scripts/plot_stage1_report.py draws:

    python scripts/model_report_data.py --checkpoint outputs/stage1/d10_n300k_mlp_cat/step150000.ckpt \\
        --out outputs/report/d10_n300k_mlp_cat
    python scripts/plot_stage1_report.py --data outputs/report/d10_n300k_mlp_cat \\
        --reference outputs/report/d10_n300k_mlp_cat/comparison.npz --label "pair-MLP head (concat)"
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from delssome_fm.config import ReferenceEvalConfig, load_config
from delssome_fm.train.evaluate import load_stage1
from delssome_fm.train.report import (prediction_vs_simulation, reference_comparison,
                                      training_average)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=20_000, help="test rows and training-average rows")
    p.add_argument("--reference-config", type=Path, default=Path("configs/reference_eval.yaml"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = load_stage1(args.checkpoint).to(device)
    test = prediction_vs_simulation(module, "test", args.rows)
    np.savez(args.out / "report_data.npz", **{f"best_{k}": v for k, v in test.items()})
    ref_cfg = load_config(args.reference_config, ReferenceEvalConfig)
    fc_avg, cdf_avg = training_average(module, args.rows)
    arrays = {}
    for name in ref_cfg.models:
        r = reference_comparison(module, ref_cfg.output_dir / "store", name, fc_avg, cdf_avg)
        arrays.update({f"{name}_{k}": v for k, v in r.items()})
    np.savez(args.out / "comparison.npz", checkpoint=str(args.checkpoint), **arrays)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
