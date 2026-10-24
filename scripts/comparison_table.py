"""Comparison table of stage-1 models for the report: metrics on the rows each model trained
on, val and test (MLflow final/<split>/*, from train/fit.final_evaluation or
scripts/evaluate_final.py) and the reference models' cost terms (scripts/model_report_data.py's
comparison.npz, where it exists). Writes <out>/comparison_table.md and .csv.

    python scripts/comparison_table.py --runs d10 --out outputs/report
    python scripts/comparison_table.py --runs sinh --out outputs/report_sinh
"""

import argparse
import csv
from pathlib import Path

import mlflow
import numpy as np

# run set -> (MLflow experiment, {run: (description, reference-model comparison file)})
D10 = {
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
SINH = {   # 100k models x all draws (~40 usable each) of the sinh corpus, mlp_concat FC head
    "sinh_orig": ("original", "outputs/report_sinh/sinh_orig/comparison.npz"),
    "sinh_m100k_d100_cost": ("original + cost loss",
                             "outputs/report_sinh/sinh_m100k_d100_cost/comparison.npz"),
    "sinh_orig_c24": ("original + catch24 loss", "outputs/report_sinh/sinh_orig_c24/comparison.npz"),
    "sinh_cost_c24": ("original + cost + catch24 loss",
                      "outputs/report_sinh/sinh_cost_c24/comparison.npz"),
    "sinh_c24_only": ("catch24 loss only", "outputs/report_sinh/sinh_c24_only/comparison.npz"),
    "sinh_latent_cost": ("latent FC head (autoencoder) + cost loss",
                         "outputs/report_sinh/sinh_latent_cost/comparison.npz"),
}
RUN_SETS = {"d10": ("stage1", D10), "sinh": ("stage1_sinh", SINH)}
SPLITS = [("train", "trained rows"), ("val", "val"), ("test", "test")]
TEST = [("total", "MSE total"), ("r2_pairwise", "R² FC"),
        ("cost_corr_coupled", "1−r coupled"), ("cost_mean_coupled", "d coupled"),
        ("cost_ks_coupled", "KS coupled"), ("cost_mean_uncoupled", "d uncoupled"),
        ("cost_ks_uncoupled", "KS uncoupled"), ("catch24", "catch24 MSE")]
MODELS = [("mfm", "MFM"), ("fic", "FIC"), ("hopf", "Hopf")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", choices=sorted(RUN_SETS), default="d10")
    p.add_argument("--out", type=Path, default=Path("outputs/report"))
    p.add_argument("--mlflow", default="sqlite:////home/ftian/storage/projects/DELSSOME-FM/"
                   "outputs/mlflow/mlflow.db")
    args = p.parse_args()

    experiment, runs_spec = RUN_SETS[args.runs]
    client = mlflow.MlflowClient(tracking_uri=args.mlflow)
    runs = {r.info.run_name: r for r in
            client.search_runs([client.get_experiment_by_name(experiment).experiment_id])}
    rows = []
    for name, (desc, ref_path) in runs_spec.items():
        if name not in runs or "final/test/total" not in runs[name].data.metrics:
            print(f"skipping {name}: no final evaluation in MLflow yet")
            continue
        m = runs[name].data.metrics
        row = {"run": name, "model": desc}
        for split, _ in SPLITS:
            row.update({f"{split} {label}": m.get(f"final/{split}/{k}", np.nan)
                        for k, label in TEST})
        ref = np.load(ref_path) if Path(ref_path).is_file() else None
        for key, label in MODELS:
            for t, tl in (("corr", "1−r"), ("mean", "d"), ("ks", "KS")):
                row[f"{label} {tl}"] = (float(np.median(ref[f"{key}_model_{t}"]))
                                        if ref is not None else np.nan)
        rows.append(row)
    with_ref = [spec[1] for name, spec in runs_spec.items()
                if name in {r["run"] for r in rows} and Path(spec[1]).is_file()]
    if not with_ref:
        raise FileNotFoundError("no reference comparison yet: run scripts/model_report_data.py")
    floors = np.load(with_ref[0])
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
    ref_keys = ["model"] + [k for k in keys if k.split(" ")[0] in ("MFM", "FIC", "Hopf")]
    lines = []
    for split, title in SPLITS:
        split_keys = ["model"] + [f"{split} {label}" for _, label in TEST]
        lines += [f"## Synthetic models, {title} (medians of cost terms over draws)", "",
                  "| " + " | ".join(["model"] + [label for _, label in TEST]) + " |",
                  "|" + "---|" * len(split_keys)]
        lines += ["| " + " | ".join(fmt(r.get(k, np.nan)) for k in split_keys) + " |"
                  for r in rows[:-1]]
        lines += [""]
    lines += ["## Reference models at their published test fits (medians over sets)", "",
              "| " + " | ".join(ref_keys) + " |", "|" + "---|" * len(ref_keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in ref_keys) + " |" for r in rows]
    (args.out / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
