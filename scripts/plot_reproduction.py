"""Box plots of the recorded original test costs vs ours: 50 seeds per model, one evaluation
each (3 simulations averaged), from scripts/reproduce_costs.py with configs/reproduce_once.yaml.

Needs matplotlib, which this project does not depend on; run it in an env that has it:
    /home/ftian/storage/miniconda/envs/lifespan_ei/bin/python scripts/plot_reproduction.py
Writes outputs/reproduce/once/comparison_total.png and comparison_components.png."""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/nas/CSC21/Yeolab/Users/ftian/projects/DELSSOME-FM/outputs/reproduce")
MODELS = [("fic", "FIC"), ("mfm", "MFM"), ("hopf", "Hopf")]
COMPONENTS = [("corr", "1 − r (FC correlation)"), ("mean", "d (FC mean difference)"), ("ks", "KS (FCD)")]
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1, 2 (reference palette)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(model):
    orig = np.load(ROOT / "original" / f"{model}.npz")
    parts = [np.load(f) for f in glob.glob(str(ROOT / "once" / f"{model}_sets*.npz"))]
    sets = np.concatenate([p["sets"] for p in parts])
    if sorted(sets.tolist()) != list(range(50)):
        raise ValueError(f"{model}: results cover sets {sorted(sets.tolist())}, expected 0..49")
    order = np.argsort(sets)
    ours = {c: np.concatenate([p[c][:, 0] for p in parts])[order] for c in ("corr", "mean", "ks")}
    ours["total"] = ours["corr"] + ours["mean"] + ours["ks"]
    rec = {c: orig[c] for c in ("corr", "mean", "ks")}
    rec["total"] = rec["corr"] + rec["mean"] + rec["ks"]
    n_valid = np.concatenate([p["n_valid"][:, 0] for p in parts])[order]
    return rec, ours, n_valid


def panel(ax, rec, ours, title, ylabel):
    rng = np.random.default_rng(0)
    x = {0: rng.uniform(-0.12, 0.12, len(rec)), 1: 1 + rng.uniform(-0.12, 0.12, len(ours))}
    for a, b, xa, xb in zip(rec, ours, x[0], x[1]):          # pair lines, recessive
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([xa, xb], [a, b], color=GRID, lw=0.8, zorder=1)
    bp = ax.boxplot([rec[np.isfinite(rec)], ours[np.isfinite(ours)]], positions=[0, 1],
                    widths=0.5, showfliers=False, patch_artist=True, zorder=2,
                    medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=INK2, lw=1),
                    capprops=dict(color=INK2, lw=1), boxprops=dict(lw=1, edgecolor=INK2))
    for patch, c in zip(bp["boxes"], (BLUE, ORANGE)):
        patch.set_facecolor(c)
        patch.set_alpha(0.18)
    ax.scatter(x[0], rec, s=22, color=BLUE, edgecolor=SURFACE, lw=0.8, zorder=3)
    ax.scatter(x[1], ours, s=22, color=ORANGE, edgecolor=SURFACE, lw=0.8, zorder=3)
    for xi, v in ((0, rec), (1, ours)):
        ax.annotate(f"median {np.nanmedian(v):.3f}", (xi, np.nanmax(v)), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xticks([0, 1], ["Tianchu\n(recorded)", "Ours"], color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.margins(y=0.12)


def main():
    data = {m: load(m) for m, _ in MODELS}
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, (m, name) in zip(axes, MODELS):
        ax.set_facecolor(SURFACE)
        rec, ours, n_valid = data[m]
        panel(ax, rec["total"], ours["total"], name, "Total cost (1 − r) + d + KS")
        print(f"{name}: recorded median {np.median(rec['total']):.3f}, ours median "
              f"{np.nanmedian(ours['total']):.3f}; ours diverged evaluations "
              f"{int(np.sum(n_valid == 0))}, partial {int(np.sum((n_valid > 0) & (n_valid < 3)))}")
    fig.suptitle("HCP-YA test cost of the 50 surviving parameter sets per model: original vs ours "
                 "(one evaluation each, 3 simulations averaged)", fontsize=10, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "once" / "comparison_total.png", dpi=150, facecolor=SURFACE)

    fig, axes = plt.subplots(3, 3, figsize=(11, 10), facecolor=SURFACE)
    for j, (m, name) in enumerate(MODELS):
        rec, ours, _ = data[m]
        for i, (c, label) in enumerate(COMPONENTS):
            axes[i, j].set_facecolor(SURFACE)
            panel(axes[i, j], rec[c], ours[c], f"{name}: {label}", label)
    fig.tight_layout()
    fig.savefig(ROOT / "once" / "comparison_components.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
