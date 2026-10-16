"""Generalisation diagnostic, step 3 of 3, or any evaluation: metrics of stage-1 checkpoints
on packed splits (train/evaluate.py). Prints one row per checkpoint and split.

    python scripts/evaluate_stage1.py --store-dir outputs/generalisation/store \\
        --splits seen_model_seen_sc seen_model_unseen_sc unseen_model_seen_sc \\
                 unseen_model_unseen_sc --checkpoints outputs/stage1/stage1_v1/last.ckpt
"""

import argparse
import json
from pathlib import Path

import torch

from delssome_fm.train.evaluate import evaluate, load_stage1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--max-rows", type=int, default=None, help="seeded subset per split")
    p.add_argument("--out", type=Path, default=None, help="also write the results as JSON")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for ckpt in args.checkpoints:
        module = load_stage1(ckpt).to(device)
        for split in args.splits:
            m = evaluate(module, args.store_dir, split, max_rows=args.max_rows)
            results.append({"checkpoint": str(ckpt), "split": split, **m})
            print(f"{ckpt.parent.name}/{ckpt.name}  {split:24s} " + "  ".join(
                f"{k} {m[k]:.4f}" for k in ("total", "regional", "pairwise", "fcd",
                                            "fc_moments", "r2_pairwise")) + f"  rows {m['rows']}")
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
