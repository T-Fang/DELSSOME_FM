"""Compare stage-1 checkpoints on a split, by regime: each model's error weighted equally,
for uncoupled models (mean FC <= threshold) and coupled ones, with 95% intervals over models,
and the paired difference of each checkpoint from the first (same rows for all).

    python scripts/compare_by_regime.py --store-dir outputs/training_store --split val \\
        --max-rows 20000 --checkpoints outputs/stage1/a/step1.ckpt outputs/stage1/b/step2.ckpt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from delssome_fm.train.evaluate import bootstrap_mean, load_stage1, per_model_errors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--threshold", type=float, default=0.05, help="coupled: model mean FC > this")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    errs = {c: per_model_errors(load_stage1(c).to(device), args.store_dir, args.split,
                                max_rows=args.max_rows) for c in args.checkpoints}
    first = errs[args.checkpoints[0]]
    for c, e in errs.items():          # same seeded rows, so the same models, for every one
        if not np.array_equal(e["model"], first["model"]):
            raise ValueError(f"{c}: evaluated on different models than {args.checkpoints[0]}")
    regimes = {"uncoupled": first["mean_fc"] <= args.threshold,
               "coupled": first["mean_fc"] > args.threshold}
    for name, mask in regimes.items():
        print(f"== {name}: {mask.sum()} models")
        for c, e in errs.items():
            m, lo, hi = bootstrap_mean(e["total"][mask])
            line = f"  {c.parent.name:22s} total {m:.3f} [{lo:.3f}, {hi:.3f}]  " + "  ".join(
                f"{g} {e[g][mask].mean():.3f}" for g in ("regional", "pairwise", "fcd", "fc_moments"))
            if c != args.checkpoints[0]:
                d, dlo, dhi = bootstrap_mean(e["total"][mask] - first["total"][mask])
                line += f"   vs first: {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
            print(line)


if __name__ == "__main__":
    main()
