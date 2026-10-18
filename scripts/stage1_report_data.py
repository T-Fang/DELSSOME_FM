"""Compute the numbers behind the stage-1 report figures (train/report.py) and save them to
<out>/report_data.npz and <out>/scaling.json. Evaluates many checkpoints on 20k rows each, so
run it as a CPU job:

    python scripts/submit_job.py --name report_data --walltime 06:00:00 --mem 32G --ncpus 16 \\
        --conda-env delssome_fm_train --submit -- env OMP_NUM_THREADS=16 \\
        python scripts/stage1_report_data.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from delssome_fm.train.evaluate import load_stage1, per_model_errors
from delssome_fm.train.report import prediction_vs_simulation, scaling_metrics

S = Path("outputs/stage1")
# run name -> (best checkpoint by val/total, training models, draws per model)
SCALING = {
    "sweep_m10k_d10": ("step50000.ckpt", 10_000, "10 draws"),
    "sweep_m30k_d10": ("step85000.ckpt", 30_000, "10 draws"),
    "sweep_m100k_d10": ("step80000.ckpt", 100_000, "10 draws"),
    "d10_n150k_c36": ("step140000.ckpt", 150_000, "10 draws"),
    "d10_n300k_natural": ("step80000.ckpt", 300_000, "10 draws"),
    "sweep_m10k_dall": ("step115000.ckpt", 10_000, "all draws (~80)"),
    "sweep_m30k_dall": ("step105000.ckpt", 30_000, "all draws (~80)"),
    "stage1_v1": ("step170000.ckpt", 100_000, "all draws (~80)"),
}
COMPOSITION = {"d10_n150k_c36": 0.364, "d10_n150k_c50": 0.5, "d10_n150k_c70": 0.7,
               "d10_n150k_c85": 0.85}
COMPOSITION_CKPT = {"d10_n150k_c36": "step140000.ckpt", "d10_n150k_c50": "step115000.ckpt",
                    "d10_n150k_c70": "step90000.ckpt", "d10_n150k_c85": "step70000.ckpt"}
BEST = "d10_n300k_natural"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("outputs/report"))
    p.add_argument("--rows", type=int, default=20_000, help="rows per evaluation")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scaling = {}
    for run, (ckpt, n_models, draws) in SCALING.items():
        scaling[run] = {"n_models": n_models, "draws": draws,
                        **scaling_metrics(load_stage1(S / run / ckpt), args.rows)}
        print(run, {k: round(v["total"], 4) for k, v in scaling[run].items() if k in ("train", "val", "test")})
    (args.out / "scaling.json").write_text(json.dumps(scaling, indent=2))

    arrays = {f"best_{k}": v for k, v in prediction_vs_simulation(
        load_stage1(S / BEST / SCALING[BEST][0]), "test", args.rows).items()}
    for run, ckpt in [*COMPOSITION_CKPT.items(), (BEST, SCALING[BEST][0])]:
        e = per_model_errors(load_stage1(S / run / ckpt), Path("outputs/training_store_d10"),
                             "val", max_rows=args.rows)
        arrays.update({f"comp_{run}_total": e["total"], f"comp_{run}_mean_fc": e["mean_fc"]})
    np.savez(args.out / "report_data.npz", **arrays)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
