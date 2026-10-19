"""Comparison table of stage-1 models for the report: held-out test metrics (MLflow
final/test/*, from train/fit.final_evaluation or scripts/evaluate_final.py) and the reference
models' cost terms (scripts/model_report_data.py's comparison.npz). Writes
<out>/comparison_table.md and .csv.

    python scripts/comparison_table.py --out outputs/report
"""

import argparse
import csv
from pathlib import Path

import mlflow
import numpy as np

# run -> (description, reference-model comparison file)
RUNS = {
    "stage1_v1": ("100k models × ~80 draws, bilinear head", "outputs/report/stage1_v1/comparison.npz"),
    "d10_n300k_natural": ("300k models × 10 draws, bilinear head",
                          "outputs/report/d10_n300k_natural/comparison.npz"),
    "d10_n300k_costloss": ("300k × 10, bilinear head + cost loss",
                           "outputs/report/d10_n300k_costloss/comparison.npz"),
    "d10_n300k_mlp_sym": ("300k × 10, pair-MLP head (symmetric)",
                          "outputs/report/d10_n300k_mlp_sym/comparison.npz"),
    "d10_n300k_mlp_cat": ("300k × 10, pair-MLP head (concat)",
                          "outputs/report/d10_n300k_mlp_cat/comparison.npz"),
}
TEST = [("total", "MSE total"), ("r2_pairwise", "R² FC"),
        ("cost_corr_coupled", "1−r coupled"), ("cost_mean_coupled", "d coupled"),
        ("cost_ks_coupled", "KS coupled"), ("cost_mean_uncoupled", "d uncoupled"),
        ("cost_ks_uncoupled", "KS uncoupled")]
MODELS = [("mfm", "MFM"), ("fic", "FIC"), ("hopf", "Hopf")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("outputs/report"))
    p.add_argument("--mlflow", default="sqlite:////home/ftian/storage/projects/DELSSOME-FM/"
                   "outputs/mlflow/mlflow.db")
    args = p.parse_args()

    client = mlflow.MlflowClient(tracking_uri=args.mlflow)
    runs = {r.info.run_name: r for r in
            client.search_runs([client.get_experiment_by_name("stage1").experiment_id])}
    rows = []
    for name, (desc, ref_path) in RUNS.items():
        m = runs[name].data.metrics
        row = {"run": name, "model": desc}
        row.update({label: m.get(f"final/test/{k}", np.nan) for k, label in TEST})
        ref = np.load(ref_path)
        for key, label in MODELS:
            for t, tl in (("corr", "1−r"), ("mean", "d"), ("ks", "KS")):
                row[f"{label} {tl}"] = float(np.median(ref[f"{key}_model_{t}"]))
        rows.append(row)
    floors = np.load(RUNS["d10_n300k_natural"][1])
    floor = {"run": "noise floor", "model": "a 2nd simulation of the same parameters"}
    for key, label in MODELS:
        for t, tl in (("corr", "1−r"), ("mean", "d"), ("ks", "KS")):
            floor[f"{label} {tl}"] = float(np.median(floors[f"{key}_floor_{t}"]))
    rows.append(floor)

    args.out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with open(args.out / "comparison_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: v if isinstance(v, str) else ("—" if np.isnan(v) else f"{v:.3f}")  # noqa: E731
    test_keys = ["model"] + [label for _, label in TEST]
    ref_keys = ["model"] + [k for k in keys if k.split(" ")[0] in ("MFM", "FIC", "Hopf")]
    lines = ["## Held-out test models (synthetic; medians of cost terms over draws)", "",
             "| " + " | ".join(test_keys) + " |", "|" + "---|" * len(test_keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in test_keys) + " |" for r in rows[:-1]]
    lines += ["", "## Reference models at their published test fits (medians over sets)", "",
              "| " + " | ".join(ref_keys) + " |", "|" + "---|" * len(ref_keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in ref_keys) + " |" for r in rows]
    (args.out / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
