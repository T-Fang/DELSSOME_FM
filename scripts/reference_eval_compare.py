"""Stage-1 test on the reference models, step 2 of 2: the best stage-1 model's predictions vs
the simulations of scripts/reference_eval_data.py (train/report.reference_comparison). Writes
<output_dir>/comparison.npz, which scripts/plot_stage1_report.py draws. Small; runs locally:

    OMP_NUM_THREADS=8 python scripts/reference_eval_compare.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from delssome_fm.config import ReferenceEvalConfig, load_config
from delssome_fm.train.evaluate import load_stage1
from delssome_fm.train.report import reference_comparison, training_average

BEST = Path("outputs/stage1/d10_n300k_natural/step80000.ckpt")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/reference_eval.yaml"))
    p.add_argument("--checkpoint", type=Path, default=BEST)
    args = p.parse_args()

    torch.set_num_threads(8)
    cfg = load_config(args.config, ReferenceEvalConfig)
    module = load_stage1(args.checkpoint)
    fc_avg, cdf_avg = training_average(module, 20_000)
    arrays = {}
    for name in cfg.models:
        r = reference_comparison(module, cfg.output_dir / "store", name, fc_avg, cdf_avg)
        arrays.update({f"{name}_{k}": v for k, v in r.items()})
        print(name, f"{len(r['sets'])} sets;", ", ".join(
            f"{k} median: model {np.median(r[f'model_{k}']):.3f} floor "
            f"{np.median(r[f'floor_{k}']):.3f} reference {np.median(r[f'reference_{k}']):.3f}"
            for k in ("corr", "mean", "ks")))
    np.savez(cfg.output_dir / "comparison.npz", checkpoint=str(args.checkpoint), **arrays)


if __name__ == "__main__":
    main()
