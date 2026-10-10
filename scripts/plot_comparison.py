"""Tianchu Zeng's original simulations vs ours, for the 50 surviving test parameter sets of
each model: the FC+FCD cost *between* the two simulations, and FC/FCD matrix figures.

Inputs: outputs/reproduce/original_sim/<model>_seed<k>.npz (scripts/run_original_test.py)
and outputs/reproduce/once/<model>_sets<k-1>-<k>.npz (reproduce_costs.py with
configs/reproduce_once.yaml: two independent evaluations of ours; the second gives our noise
floor, ours vs ours).

Cost between two simulations, as the original cost but with the second simulation in place
of the empirical data: 1 - r (raw upper-triangle FC), d = |mean difference|, KS between the
FCD CDFs. Matrices are drawn as the original analysis_utils.plot_time_series draws them:
imshow with the original colour tables (assets/colormaps) and a colorbar; the colour
range is shared by Tianchu's and our panel of each row (min and max over both).

Needs matplotlib; run it in the lifespan_ei env:
    /home/ftian/storage/miniconda/envs/lifespan_ei/bin/python scripts/plot_comparison.py
Writes outputs/reproduce/comparison/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from matplotlib.colors import ListedColormap

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs" / "reproduce"
OUT = ROOT / "comparison"
MODELS = [("fic", "FIC"), ("mfm", "MFM"), ("hopf", "Hopf")]
COMPONENTS = [("total", "Total (1 − r) + d + KS"), ("corr", "1 − r"), ("mean", "d"),
              ("ks", "KS")]
INK, INK2, GRID, SURFACE, DOT = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"


def cost(fc_a, cdf_a, fc_b, cdf_b):
    iu = np.triu_indices(fc_a.shape[0], 1)
    a, b = fc_a[iu], fc_b[iu]
    corr = 1 - np.corrcoef(a, b)[0, 1]
    mean = abs(a.mean() - b.mean())
    ks = np.max(np.abs(cdf_a / cdf_a[-1] - cdf_b / cdf_b[-1]))
    return {"corr": corr, "mean": mean, "ks": ks, "total": corr + mean + ks}


def load(model, k):
    t = np.load(ROOT / "original_sim" / f"{model}_seed{k}.npz")
    o = np.load(ROOT / "once" / f"{model}_sets{k - 1}-{k}.npz")
    if o["n_valid"][0].min() == 0:
        raise ValueError(f"{model} seed{k}: one of our evaluations had no valid simulation")
    return t, o


def costs(model):
    vs_orig, floor = [], []
    for k in range(1, 51):
        t, o = load(model, k)
        t_cdf = np.cumsum(t["fcd_hist"])
        vs_orig.append(cost(t["fc"], t_cdf, o["fc"][0, 0], o["fcd_cdf"][0, 0]))
        floor.append(cost(o["fc"][0, 1], o["fcd_cdf"][0, 1], o["fc"][0, 0], o["fcd_cdf"][0, 0]))
    return ({c: np.array([d[c] for d in vs_orig]) for c, _ in COMPONENTS},
            {c: np.array([d[c] for d in floor]) for c, _ in COMPONENTS})


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def cost_figure():
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for i, (m, name) in enumerate(MODELS):
        vs, floor = costs(m)
        for j, (c, label) in enumerate(COMPONENTS):
            ax = axes[i, j]
            style(ax)
            v = vs[c]
            ax.boxplot([v], positions=[0], widths=0.5, showfliers=False, patch_artist=True,
                       medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=INK2),
                       capprops=dict(color=INK2),
                       boxprops=dict(facecolor="#dbe8f8", edgecolor=INK2))
            ax.scatter(rng.uniform(-0.12, 0.12, len(v)), v, s=20, color=DOT,
                       edgecolor=SURFACE, lw=0.8, zorder=3)
            ax.axhline(np.median(floor[c]), color=INK2, lw=1, ls="--")
            ax.annotate(f"ours vs ours median {np.median(floor[c]):.3f}",
                        (0.98, np.median(floor[c])), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", ha="right", fontsize=7,
                        color=INK2)
            ax.set_xticks([0], [f"median {np.median(v):.3f}"], color=INK2, fontsize=8)
            ax.set_xlim(-0.6, 0.6)
            ax.set_title(f"{name}: {label}", color=INK, fontsize=10, loc="left")
            print(f"{name} {c}: Tianchu vs ours median {np.median(v):.4f}, "
                  f"ours vs ours median {np.median(floor[c]):.4f}")
    fig.suptitle("Cost between Tianchu's simulation and ours, per surviving test parameter set "
                 "(50 per model; 3 simulations averaged each). Dashed: our noise floor",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "cost_tianchu_vs_ours.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def colormaps():
    fc = ListedColormap(spio.loadmat(REPO / "assets/colormaps/corr_mat_colorscale.mat")["rbmap2"])
    c3 = np.asarray(spio.loadmat(REPO / "assets/colormaps/Rapaeh_color_table_FCD.mat")["c3"])
    fcd = ListedColormap(np.hstack((c3, np.ones((c3.shape[0], 1)))))
    return fc, fcd


def matrix_figures():
    fc_cmap, fcd_cmap = colormaps()
    (OUT / "matrices").mkdir(parents=True, exist_ok=True)
    for m, name in MODELS:
        for k in range(1, 51):
            t, o = load(m, k)
            fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
            rows = [("FC", fc_cmap, t["fc_first"], o["fc_first"][0]),
                    ("FCD", fcd_cmap, t["fcd_first"], o["fcd_first"][0])]
            for i, (kind, cmap, a, b) in enumerate(rows):
                lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
                for ax, mat, who in zip(axes[i], (a, b), ("Tianchu", "ours")):
                    im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi)
                    ax.set_title(f"{kind}, {who}")
                    fig.colorbar(im, ax=ax)
            fig.suptitle(f"{name}, test parameter set of seed {k}: one simulation each")
            fig.tight_layout()
            fig.savefig(OUT / "matrices" / f"{m}_seed{k}.png", dpi=110)
            plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cost_figure()
    matrix_figures()


if __name__ == "__main__":
    main()
