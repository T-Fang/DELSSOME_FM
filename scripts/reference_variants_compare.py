"""Reference-model variants, step 3 of 3 (sim/reference_eval.py, 2026-11-04): a stage-1 model
on each reference model's fitted sets (split <model>_r0), the fitted sets with i.i.d. regional
maps (<model>_scrambled) and corpus draws of the card (<model>_random), against a seeded
subset of its own synthetic test draws. Per split: the regime call (predicted mean FC above
coupled_threshold) and the median cost terms per simulated regime (train/report.regime_summary).
Writes <out>/<name>.npz (per-row arrays) and prints the table. A GPU job:

    python scripts/reference_variants_compare.py \\
        --checkpoint outputs/stage1/sinh_ptok_raw_m300k_s400k/step385000.ckpt \\
        --out outputs/report_sinh/sinh_ptok_raw_m300k_s400k
    ... --models fic --variants r0 scrambled shuffled scrambled_solved --test-rows 0 \
        --name reference_variants_fic
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from delssome_fm.config import ReferenceEvalConfig, load_config
from delssome_fm.train.data import TrainingStore
from delssome_fm.train.evaluate import evaluation_rows, load_stage1
from delssome_fm.train.report import regime_rows, regime_summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/reference_eval_sinh.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--test-rows", type=int, default=20_000, help="0: no synthetic baseline")
    p.add_argument("--models", nargs="+", default=None, help="default: the config's models")
    p.add_argument("--variants", nargs="+", default=["r0", "scrambled", "random"],
                   help="store splits <model>_<variant> (r0: the fitted sets)")
    p.add_argument("--name", default="reference_variants")
    args = p.parse_args()

    cfg = load_config(args.config, ReferenceEvalConfig)
    module = load_stage1(args.checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
    threshold = module.cfg.coupled_threshold
    stores = {"synthetic_test": TrainingStore(module.cfg.store_dir, "test")} if args.test_rows else {}
    stores.update({f"{n}_{v}": TrainingStore(cfg.output_dir / "store", f"{n}_{v}")
                   for n in args.models or cfg.models for v in args.variants})
    arrays, table = {}, {}
    for split, store in stores.items():
        if split == "synthetic_test":
            pool = evaluation_rows(np.arange(len(store)), module.cfg)
            rows = np.sort(np.random.default_rng(0).choice(pool, args.test_rows, replace=False))
        else:
            rows = np.arange(len(store))
        r = regime_rows(module, store, rows)
        arrays.update({f"{split}_{k}": v for k, v in r.items()})
        table[split] = regime_summary(r, threshold)
        print(split, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                 for k, v in table[split].items()}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"{args.name}.npz", checkpoint=str(args.checkpoint),
             summary=json.dumps(table), **arrays)
    print(f"wrote {args.out / args.name}.npz")


if __name__ == "__main__":
    main()
