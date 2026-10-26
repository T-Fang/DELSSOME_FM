"""Figures for the stage-1 progress report, from scripts/stage1_report_data.py's output and the
generalisation diagnostic. Writes PNGs to <data>/figures/:

    scaling.png         total MSE vs number of training models (train / val / test; val as
                        logged in MLflow at the best step, 50k rows), pairwise R^2, and
                        train and val MSE for 10 draws per model vs all draws
    cost_terms.png      1 - r, d and KS between the best model's prediction and the numerical
                        simulation, test draws, by regime; vs a training-average reference
    fc_examples.png     simulated vs predicted FC for four test draws
    fcd_examples.png    simulated vs predicted FCD CDF for the same draws, with the KS gap
    cost_terms_train.png, fc_examples_train.png, fcd_examples_train.png
                        the same on rows the model trained on (from report_data_train.npz,
                        scripts/model_report_data.py --split train; skipped if absent)
    scaling_grid.png    models x draws-per-model grid (--scaling-grid: final/<split>/* of the
                        MLflow runs <prefix>_m<M>k_d<D>_cost), train and val
    catch24.png         per-feature catch24 R^2, trained rows vs test (models with the head)
    network_fc.png, network_fc_examples.png
                        Yeo-7 network FC: predicted by an fc_net head and/or downsampled from
                        the region-level prediction, trained rows vs test
                        (scripts/network_fc_report_data.py; skipped if absent)
    composition.png     per-model error vs the share of coupled models in training
    generalisation.png  the 2 x 2 diagnostic as a matrix: seen/new equations x seen/new SC
                        (from scripts/generalisation_report_data.py)
    reference_cost_terms.png, reference_examples.png
                        MFM, FIC, Hopf at their published test fits: prediction vs simulation
                        (from scripts/reference_eval_compare.py; skipped if not yet run)

Needs matplotlib (the delssome_fm_train env has it):
    python scripts/plot_stage1_report.py --data outputs/report
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

REPO = Path(__file__).resolve().parents[1]
# reference palette (dataviz skill, light mode): categorical slots 1-3, ink, chrome
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
FCD_LEVELS = np.linspace(-0.98, 1.0, 100)       # summary.fcd_levels with stride 100


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, out, name):
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}")


def mlflow_val(scaling: dict, tracking_uri: str) -> dict:
    """Replace each run's val metrics with those MLflow logged at its best step (val/total):
    the fixed 50k-row val subset used during training, the numbers quoted elsewhere."""
    import mlflow
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    runs = {r.info.run_name: r.info.run_id
            for r in client.search_runs([client.get_experiment_by_name("stage1").experiment_id])}
    for name, v in scaling.items():
        hist = {k: {m.step: m.value for m in client.get_metric_history(runs[name], f"val/{k}")}
                for k in ("total", "r2_pairwise")}
        best = min(hist["total"], key=hist["total"].get)
        v["val"] = {**v["val"], "total": hist["total"][best],
                    "r2_pairwise": hist["r2_pairwise"][best]}
    return scaling


def scaling_figure(scaling: dict, out: Path):
    ten = sorted((v["n_models"], v) for v in scaling.values() if v["draws"] == "10 draws")
    allr = sorted((v["n_models"], v) for v in scaling.values() if v["draws"] != "10 draws")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = [n for n, _ in ten]
    ax = axes[0]
    for split, color in (("train", BLUE), ("val", ORANGE), ("test", AQUA)):
        y = [v[split]["total"] for _, v in ten]
        ax.plot(x, y, "-o", color=color, lw=2, ms=7, label=split, mec=SURFACE, mew=1.5)
    ax.set_title("Total MSE vs training models (10 draws each)", loc="left")
    ax.set_ylabel("total MSE (sum of 4 groups)")
    ax = axes[1]
    for split, color in (("val", ORANGE), ("test", AQUA)):
        ax.plot(x, [v[split]["r2_pairwise"] for _, v in ten], "-o", color=color, lw=2, ms=7,
                label=split, mec=SURFACE, mew=1.5)
    ax.set_title("Pairwise FC: R² on held-out models", loc="left")
    ax.set_ylabel("R² of z-scored arctanh FC")
    ax = axes[2]
    for series, color, label in ((ten, ORANGE, "10 draws per model"),
                                 (allr, BLUE, "all draws (~80 per model)")):
        for split, ls, marker in (("val", "-", "o"), ("train", "--", "s")):
            ax.plot([n for n, _ in series], [v[split]["total"] for _, v in series], ls,
                    marker=marker, color=color, lw=2, ms=7, label=f"{split}, {label}",
                    mec=SURFACE, mew=1.5)
    ax.set_title("Train and val MSE: models vs draws per model", loc="left")
    ax.set_ylabel("total MSE (train: the model's own training rows)")
    for ax in axes:
        style(ax)
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{n // 1000}k" for n in x])
        ax.minorticks_off()
        ax.set_xlabel("training models")
        ax.legend(frameon=False, fontsize=9)
    axes[2].legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 0.2))
    fig.suptitle("Stage 1 scaling: more distinct models helps; more draws per model barely does",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, out, "scaling.png")


# split -> (how titles name the draws, file-name suffix)
SPLITS = {"test": ("held-out test draws", ""),
          "train": ("training draws (rows the model trained on)", "_train")}


GRID_MODELS, GRID_DRAWS = (10, 30, 100), (10, 30, 100)   # thousands of models; draws per model
GRID_COLOURS = (BLUE, ORANGE, AQUA)


def mlflow_grid(tracking_uri: str, experiment: str, prefix: str) -> dict:
    """{(models_k, draws): final metrics} of the grid runs that have finished."""
    import mlflow
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    runs = {r.info.run_name: r.data.metrics for r in
            client.search_runs([client.get_experiment_by_name(experiment).experiment_id])}
    return {(m, d): runs[f"{prefix}_m{m}k_d{d}_cost"] for m in GRID_MODELS for d in GRID_DRAWS
            if "final/val/total" in runs.get(f"{prefix}_m{m}k_d{d}_cost", {})}


def scaling_grid_figure(grid: dict, out: Path, model_label: str):
    """Total MSE and coupled 1 - r against training models (one line per draws per model) and
    against draws per model (one line per model count); val solid, trained rows dashed."""
    panels = [("total", "total MSE (sum of 4 groups)", "models"),
              ("total", "total MSE (sum of 4 groups)", "draws"),
              ("r2_pairwise", "R² of z-scored arctanh FC", "models"),
              ("cost_corr_coupled", "1 − r, coupled draws (median)", "models")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    for ax, (metric, ylabel, along) in zip(axes, panels):
        lines = GRID_DRAWS if along == "models" else GRID_MODELS
        xs = GRID_MODELS if along == "models" else GRID_DRAWS
        for colour, fixed in zip(GRID_COLOURS, lines):
            for split, ls, marker in (("val", "-", "o"), ("train", "--", "s")):
                pts = [(x, grid[(x, fixed) if along == "models" else (fixed, x)]
                        .get(f"final/{split}/{metric}")) for x in xs
                       if ((x, fixed) if along == "models" else (fixed, x)) in grid]
                pts = [(x, y) for x, y in pts if y is not None]
                if pts:
                    label = f"{split}, {fixed} draws" if along == "models" else f"{split}, {fixed}k models"
                    ax.plot(*zip(*pts), ls, marker=marker, color=colour, lw=2, ms=7,
                            mec=SURFACE, mew=1.5, label=label)
        style(ax)
        ax.set_xscale("log")
        ax.set_xticks(xs, [f"{x}k" if along == "models" else str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("training models" if along == "models" else "draws per model (at most)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Scaling on the sinh corpus: {model_label}; val (solid) vs the rows each "
                 "model trained on (dashed)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, out, "scaling_grid.png")


def catch24_figure(test: dict, train: dict, out: Path, model_label: str):
    """Per-feature R^2 of the catch24 head on the rows it trained on and on test models."""
    from delssome_fm.sim.summary import CATCH24_NAMES
    fig, ax = plt.subplots(figsize=(13, 4.6))
    x = np.arange(24)
    for k, (d, colour, label) in enumerate(((train, BLUE, "trained rows"),
                                            (test, ORANGE, "test models"))):
        ax.bar(x + (k - 0.5) * 0.4, d["best_catch24_r2"], width=0.38, color=colour, label=label)
    ax.set_xticks(x, [n.replace("_", " ")[:28] for n in CATCH24_NAMES], rotation=60,
                  ha="right", fontsize=8)
    ax.set_ylabel("R² (z-scored feature, all regions)")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylim(min(-0.1, float(np.nanmin(test["best_catch24_r2"])) - 0.05), 1)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"catch24 per feature: {model_label}", loc="left")
    style(ax)
    fig.tight_layout()
    save(fig, out, "catch24.png")


NETWORK_KINDS = (("direct", "network-FC head"), ("downsampled", "region FC, downsampled"))


def network_fc_figure(test: dict, train: dict, out: Path, model_label: str):
    """1 - r on coupled draws, and the arctanh-FC MSE and |mean difference| over all draws, of
    the network FC predictions, on trained rows and test."""
    kinds = [(k, name) for k, name in NETWORK_KINDS if f"{k}_corr" in test]
    panels = [("corr", "1 − r of the 21 network FC values\n(coupled draws)", True),
              ("mse", "MSE of arctanh network FC\n(all draws)", False),
              ("mean", "|Δ mean network FC|\n(all draws)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (metric, title, coupled_only) in zip(axes, panels):
        data, pos, colors, labels = [], [], [], []
        for g, (k, name) in enumerate(kinds):
            for j, (d, split, colour) in enumerate(((train, "trained rows", BLUE),
                                                    (test, "test", ORANGE))):
                v = d[f"{k}_{metric}"][d["coupled"]] if coupled_only else d[f"{k}_{metric}"]
                data.append(v[np.isfinite(v)])
                pos.append(g * 2.6 + j)
                colors.append(colour)
        bp = ax.boxplot(data, positions=pos, widths=0.7, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=INK2),
                        capprops=dict(color=INK2))
        for box, c in zip(bp["boxes"], colors):
            box.set(facecolor=c, alpha=0.35, edgecolor=c, lw=1.5)
        for p_, v in zip(pos, data):
            ax.annotate(f"{np.median(v):.2f}", (p_, np.median(v)), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_xticks([g * 2.6 + 0.5 for g in range(len(kinds))], [name for _, name in kinds])
        ax.set_title(title, loc="left", fontsize=10)
        style(ax)
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.35, ec=BLUE),
                    plt.Rectangle((0, 0), 1, 1, fc=ORANGE, alpha=0.35, ec=ORANGE)],
                   ["trained rows", "test"], frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"Yeo-7 network FC: {model_label} vs numerical simulation "
                 f"(coupled: region mean FC > 0.05; n test = {len(test['coupled'])})",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, out, "network_fc.png")


def network_fc_examples_figure(test: dict, train: dict, out: Path, model_label: str):
    """Simulated and predicted 7 x 7 network FC of the coupled draw at the median 1 - r, on
    trained rows (top) and test (bottom)."""
    from delssome_fm.data.networks import NETWORK_PAIRS, YEO7_NAMES
    kinds = [(k, name) for k, name in NETWORK_KINDS if f"{k}_corr" in test]
    cmap = ListedColormap(spio.loadmat(REPO / "assets/colormaps/corr_mat_colorscale.mat")["rbmap2"])
    fig, axes = plt.subplots(2, 1 + len(kinds), figsize=(4.6 * (1 + len(kinds)), 8.6))
    short = [n.split()[0][:4] for n in YEO7_NAMES]

    def matrix(v):
        m = np.eye(7)
        for (a, b), x in zip(NETWORK_PAIRS, v):
            m[a, b] = m[b, a] = x
        return m

    for row, (d, split) in enumerate(((train, "trained rows"), (test, "test"))):
        key = kinds[0][0]
        c = np.flatnonzero(d["coupled"] & np.isfinite(d[f"{key}_corr"]))
        pick = c[np.argsort(d[f"{key}_corr"][c])[len(c) // 2]]
        mats = [("simulated", d["true"][pick])] + [(name, d[k][pick]) for k, name in kinds]
        lo = min(float(m.min()) for _, m in mats)
        for col, (name, v) in enumerate(mats):
            ax = axes[row, col]
            im = ax.imshow(matrix(v), cmap=cmap, vmin=lo, vmax=1)
            ax.set_xticks(range(7), short, fontsize=8)
            ax.set_yticks(range(7), short, fontsize=8)
            extra = "" if col == 0 else (f"\n1 − r = {d[f'{kinds[col - 1][0]}_corr'][pick]:.2f}, "
                                          f"d = {d[f'{kinds[col - 1][0]}_mean'][pick]:.2f}")
            ax.set_title(f"{split}: {name}{extra}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Yeo-7 network FC, the coupled draw at the median 1 − r ({kinds[0][1]}): "
                 f"{model_label}", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, out, "network_fc_examples.png")


def cost_terms_figure(d: dict, out: Path, model_label: str, note: str, split: str = "test"):
    coupled = d["best_true_mean_fc"] > 0.05
    terms = [("corr", "1 − r  (FC pattern)"), ("mean", "d = |Δ mean FC|"), ("ks", "KS (FCD CDF)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (k, title) in zip(axes, terms):
        data, pos, colors = [], [], []
        for g, mask in enumerate((~coupled, coupled)):
            for j, (who, color) in enumerate((("model", BLUE), ("reference", MUTED))):
                v = d[f"best_{who}_{k}"][mask]
                data.append(v[np.isfinite(v)])
                pos.append(g * 2.6 + j)
                colors.append(color)
        bp = ax.boxplot(data, positions=pos, widths=0.7, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=INK2),
                        capprops=dict(color=INK2))
        for box, c in zip(bp["boxes"], colors):
            box.set(facecolor=c, alpha=0.35, edgecolor=c, lw=1.5)
        for p, v in zip(pos, data):
            ax.annotate(f"{np.median(v):.2f}", (p, np.median(v)), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_xticks([0.5, 3.1], [f"uncoupled\n(n={(~coupled).sum()})",
                                   f"coupled, mean FC > 0.05\n(n={coupled.sum()})"])
        ax.set_title(title, loc="left")
        style(ax)
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.35, ec=BLUE),
                    plt.Rectangle((0, 0), 1, 1, fc=MUTED, alpha=0.35, ec=MUTED)],
                   ["stage-1 prediction", "training-average reference"], frameon=False,
                   fontsize=9, loc="lower left")
    fig.suptitle(f"{model_label} vs numerical simulation, {SPLITS[split][0]}: the original "
                 "DELSSOME cost terms", x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, -0.03, "Uncoupled draws' simulated FC is mostly sampling noise, so 1 − r ≈ 1 for "
             "any predictor." + (f" {note}" if note else ""), color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, out, f"cost_terms{SPLITS[split][1]}.png")


def _matrix(upper: np.ndarray, n: int = 68) -> np.ndarray:
    m = np.eye(n)
    i, j = np.triu_indices(n, 1)
    m[i, j] = m[j, i] = upper
    return m


def fc_examples_figure(d: dict, out: Path, model_label: str, split: str = "test"):
    cmap = ListedColormap(spio.loadmat(REPO / "assets/colormaps/corr_mat_colorscale.mat")["rbmap2"])
    labels = ["coupled, good (10th pct of 1 − r)", "coupled, typical (median)",
              "coupled, poor (90th pct)", "uncoupled, typical"]   # coupled: 0.05 < mean FC < 0.9
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))
    for col, label in enumerate(labels):
        sim, pred = d["best_example_fc_true"][col], d["best_example_fc_pred"][col]
        terms = {k: d[f"best_model_{k}"] for k in ("corr", "mean")}
        r = np.corrcoef(sim, pred)[0, 1]
        lo, hi = min(sim.min(), pred.min()), 1.0
        for row, (mat, who) in enumerate(((sim, "simulated"), (pred, "predicted"))):
            ax = axes[row, col]
            im = ax.imshow(_matrix(mat), cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{who}" + (f"\n{label}" if row == 0 else ""), fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[1, col].set_xlabel(f"1 − r = {1 - r:.2f},  d = {abs(sim.mean() - pred.mean()):.2f}",
                                color=INK2)
    fig.suptitle(f"FC: numerical simulation (top) vs {model_label} (bottom), {SPLITS[split][0]}"
                 "\n(coupled: 0.05 < mean FC < 0.9; percentiles of 1 − r among those draws); "
                 "colour range shared per column",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, out, f"fc_examples{SPLITS[split][1]}.png")


def fcd_examples_figure(d: dict, out: Path, model_label: str, split: str = "test"):
    labels = ["coupled, good", "coupled, typical", "coupled, poor", "uncoupled, typical"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True)
    for col, (ax, label) in enumerate(zip(axes, labels)):
        sim, pred = d["best_example_cdf_true"][col], d["best_example_cdf_pred"][col]
        gap = np.abs(sim - pred)
        k = int(np.argmax(gap))
        ax.plot(FCD_LEVELS, sim, color=BLUE, lw=2, label="simulated")
        ax.plot(FCD_LEVELS, pred, color=ORANGE, lw=2, ls="--", label="predicted")
        ax.vlines(FCD_LEVELS[k], min(sim[k], pred[k]), max(sim[k], pred[k]), color=INK, lw=1.5)
        ax.annotate(f"KS = {gap[k]:.3f}", (FCD_LEVELS[k], (sim[k] + pred[k]) / 2),
                    xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=9, color=INK)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("FCD value")
        style(ax, "both")
    axes[0].set_ylabel("CDF")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"FCD CDF: numerical simulation vs {model_label}, {SPLITS[split][0]} (same "
                 "draws as the FC figure); the black bar marks the KS distance", x=0.01, ha="left", color=INK,
                 fontsize=12)
    fig.tight_layout()
    save(fig, out, f"fcd_examples{SPLITS[split][1]}.png")


def composition_figure(d: dict, out: Path):
    shares = {"d10_n150k_c36": 36.4, "d10_n150k_c50": 50, "d10_n150k_c70": 70,
              "d10_n150k_c85": 85}
    ref = d["comp_d10_n150k_c36_total"]
    coupled = d["comp_d10_n150k_c36_mean_fc"] > 0.05
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, mask) in zip(axes, (("uncoupled val models", ~coupled),
                                       ("coupled val models", coupled))):
        idx = rng.integers(0, mask.sum(), (2000, mask.sum()))
        def change(run):
            diff = (d[f"comp_{run}_total"] - ref)[mask]
            boot = diff[idx].mean(1)
            return diff.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)
        xs, ms, los, his = [], [], [], []
        for run, share in shares.items():
            m, lo, hi = change(run)
            xs.append(share), ms.append(m), los.append(m - lo), his.append(hi - m)
        ax.errorbar(xs, ms, yerr=[los, his], fmt="-o", color=ORANGE, lw=2, ms=7, capsize=4,
                    mec=SURFACE, mew=1.5, label="150k models, coupled share varied")
        m, lo, hi = change("d10_n300k_natural")
        ax.errorbar([36.4], [m], yerr=[[m - lo], [hi - m]], fmt="D", color=BLUE, ms=8, capsize=4,
                    mec=SURFACE, mew=1.5, label="300k models, natural share")
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(f"{name} (n={mask.sum()})", loc="left")
        ax.set_xlabel("coupled models in training (%)")
        style(ax)
    axes[0].set_ylabel("change in per-model MSE vs natural share\n(mean, 95% CI over models)")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Oversampling coupled models helps them a little and hurts the rest; "
                 "more models helps both", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, out, "composition.png")


# sequential blue ramp (dataviz reference palette), light -> dark
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6aa5ec", "#3d86e0", "#2a78d6", "#1f62b5", "#184f95"]
GENERALISATION_RANGE = (0.0, 1.8)    # fixed, so the matrices of different models compare


def generalisation_figure(path: Path, out: Path, model_label: str):
    """2 x 2 matrix: equations (seen in training / new) x connectome (seen / new). Each cell
    is the mean over models of each model's total MSE on draws 100-109."""
    d = dict(np.load(path))
    rows, cols = ("seen", "unseen"), ("seen", "unseen")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vmin, vmax = GENERALISATION_RANGE
    means = np.array([[d[f"{r}_model_{c}_sc_total"].mean() for c in cols] for r in rows])
    im = ax.imshow(means, cmap=cmap, vmin=vmin, vmax=vmax)
    for (i, j), m in np.ndenumerate(means):
        dark = (m - vmin) / (vmax - vmin) > 0.55
        ax.text(j, i, f"{m:.2f}", ha="center", va="center", fontsize=16,
                color="white" if dark else INK)
    ax.set_xticks([0, 1], ["seen in training", "new"])
    ax.set_yticks([0, 1], ["seen in training", "new"])
    ax.set_xlabel("connectome (SC group)")
    ax.set_ylabel("model equations")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("total MSE, mean over models")
    cb.outline.set_visible(False)
    ax.set_title(f"Generalisation gap: new equations, not new connectomes\n({model_label})\n"
                 "new parameter draws 100-109 in every cell",
                 loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, out, "generalisation.png")


REFERENCE_MODELS = [("mfm", "MFM"), ("fic", "FIC"), ("hopf", "Hopf")]


def reference_cost_figure(r: dict, out: Path, model_label: str):
    """Per reference model: prediction vs simulation, the noise floor (simulation vs simulation)
    and the training average, on the model's fitted test sets."""
    terms = [("corr", "1 − r  (FC pattern)"), ("mean", "d = |Δ mean FC|"), ("ks", "KS (FCD CDF)")]
    kinds = [("model", BLUE, "stage-1 prediction"), ("floor", AQUA, "noise floor (2nd simulation)"),
             ("reference", MUTED, "training-average reference")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (k, title) in zip(axes, terms):
        for g, (m, label) in enumerate(REFERENCE_MODELS):
            for j, (kind, color, _) in enumerate(kinds):
                v = r[f"{m}_{kind}_{k}"]
                x = g * 3.4 + j
                bp = ax.boxplot([v], positions=[x], widths=0.7, showfliers=False,
                                patch_artist=True, medianprops=dict(color=INK, lw=2),
                                whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
                bp["boxes"][0].set(facecolor=color, alpha=0.3, edgecolor=color, lw=1.5)
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=color,
                           edgecolor="none", zorder=3)
        ax.set_xticks([g * 3.4 + 1 for g in range(3)],
                      [f"{label}\n(n={len(r[f'{m}_sets'])} sets)" for m, label in REFERENCE_MODELS])
        ax.set_title(title, loc="left")
        style(ax)
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.3, ec=c) for _, c, _ in kinds],
                   [l for _, _, l in kinds], frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"Reference models at their published test fits: {model_label} vs numerical "
                 "simulation (corpus protocol)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, out, "reference_cost_terms.png")


def reference_examples_figure(r: dict, out: Path, model_label: str):
    """Per reference model, a typical fitted set (median simulated mean FC): FC simulated and
    predicted on a shared colour range, the prediction on its own range (its pattern), and the
    FCD CDFs with the KS gap."""
    cmap = ListedColormap(spio.loadmat(REPO / "assets/colormaps/corr_mat_colorscale.mat")["rbmap2"])
    fig, axes = plt.subplots(3, 4, figsize=(17, 12))
    for row, (m, label) in enumerate(REFERENCE_MODELS):
        sim, pred = r[f"{m}_example_fc_true"], r[f"{m}_example_fc_pred"]
        lo = min(sim.min(), pred.min())
        panels = ((sim, "simulated FC", lo, 1.0), (pred, "predicted FC", lo, 1.0),
                  (pred, "predicted FC, own colour range", pred.min(), pred.max()))
        for col, (mat, who, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            m_ = _matrix(mat)
            if col == 2:
                np.fill_diagonal(m_, np.nan)          # the diagonal (1) would set the range
            im = ax.imshow(m_, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{label}, set {int(r[f'{m}_example_set']) + 1}: {who}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[row, 1].set_xlabel(f"1 − r = {1 - np.corrcoef(sim, pred)[0, 1]:.2f},  "
                                f"d = {abs(sim.mean() - pred.mean()):.2f}", color=INK2)
        ax = axes[row, 3]
        cs, cp = r[f"{m}_example_cdf_true"], r[f"{m}_example_cdf_pred"]
        k = int(np.argmax(np.abs(cs - cp)))
        ax.plot(FCD_LEVELS, cs, color=BLUE, lw=2, label="simulated")
        ax.plot(FCD_LEVELS, cp, color=ORANGE, lw=2, ls="--", label="predicted")
        ax.vlines(FCD_LEVELS[k], min(cs[k], cp[k]), max(cs[k], cp[k]), color=INK, lw=1.5)
        ax.annotate(f"KS = {abs(cs[k] - cp[k]):.3f}", (FCD_LEVELS[k], (cs[k] + cp[k]) / 2),
                    xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=9, color=INK)
        ax.set_title(f"{label}: FCD CDF", loc="left", fontsize=10)
        ax.set_xlabel("FCD value")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        style(ax, "both")
    fig.suptitle(f"Reference models, a typical fitted set (median simulated mean FC): simulation "
                 f"vs {model_label}",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, out, "reference_examples.png")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("outputs/report"))
    p.add_argument("--generalisation", type=Path, default=None,
                   help="scripts/generalisation_report_data.py output, e.g. "
                   "outputs/report/generalisation_stage1_v1.npz")
    p.add_argument("--generalisation-label", default=None,
                   help="how the generalisation figure names its model (default: --label)")
    p.add_argument("--label", default="best stage-1 model (300k models)",
                   help="how titles name the model")
    p.add_argument("--note", default="d is biased for the bilinear head without the FC offset "
                   "(architecture.md §7.2).", help="extra footnote of the cost-terms figure")
    p.add_argument("--reference", type=Path,
                   default=Path("outputs/reference_eval/comparison.npz"),
                   help="scripts/reference_eval_compare.py output; skipped if absent")
    p.add_argument("--scaling-grid", default=None, metavar="EXPERIMENT:PREFIX",
                   help="draw scaling_grid.png from these MLflow runs, e.g. stage1_sinh:sinh")
    p.add_argument("--mlflow", default="sqlite:////home/ftian/storage/projects/DELSSOME-FM/"
                   "outputs/mlflow/mlflow.db", help="tracking URI holding the runs' val metrics")
    args = p.parse_args()
    out = args.data / "figures"
    out.mkdir(parents=True, exist_ok=True)
    # each figure is drawn when its data exists
    d = dict(np.load(args.data / "report_data.npz")) if (args.data / "report_data.npz").is_file() \
        else {}
    t = dict(np.load(args.data / "report_data_train.npz")) \
        if (args.data / "report_data_train.npz").is_file() else {}
    if args.scaling_grid is not None:
        experiment, prefix = args.scaling_grid.split(":")
        scaling_grid_figure(mlflow_grid(args.mlflow, experiment, prefix), out, args.label)
    if (args.data / "scaling.json").is_file():
        scaling = mlflow_val(json.loads((args.data / "scaling.json").read_text()), args.mlflow)
        scaling_figure(scaling, out)
    for data, split in ((d, "test"), (t, "train")):
        if "best_model_corr" in data:           # the model predicts FC and FCD
            cost_terms_figure(data, out, args.label, args.note, split)
            fc_examples_figure(data, out, args.label, split)
            fcd_examples_figure(data, out, args.label, split)
    if "best_catch24_r2" in d and "best_catch24_r2" in t:
        catch24_figure(d, t, out, args.label)
    if all((args.data / f"network_fc_{s}.npz").is_file() for s in ("test", "train")):
        nt, nr = (dict(np.load(args.data / f"network_fc_{s}.npz")) for s in ("test", "train"))
        network_fc_figure(nt, nr, out, args.label)
        network_fc_examples_figure(nt, nr, out, args.label)
    if "comp_d10_n150k_c36_total" in d:
        composition_figure(d, out)
    if args.generalisation is not None:
        generalisation_figure(args.generalisation, out, args.generalisation_label or args.label)
    if args.reference.is_file():
        r = dict(np.load(args.reference))
        reference_cost_figure(r, out, args.label)
        reference_examples_figure(r, out, args.label)


if __name__ == "__main__":
    main()
