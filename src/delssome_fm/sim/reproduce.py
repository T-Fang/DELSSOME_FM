"""Build step 6: re-evaluate the original DELSSOME's fitted parameter sets with our cards and
simulator, and judge the result against their recorded test costs.

The recorded cost of a parameter set came from one noisy evaluation (3 simulations averaged),
so it cannot be matched exactly. Instead each set is evaluated `n_noise_repeats` times with
independent noise, and the gate (agreed 2026-10-05, configs/reproduce.yaml) asks, for each
model and each of 1 - r, d, KS and the total:

    no bias:   paired t-test of our per-set mean against the recorded values,
               p > bias_alpha / (n_models * 4)            (Bonferroni over all 12 tests)
    no misfit: |recorded - our mean| <= z_threshold * our SD for all but max_misses sets

Calibrated before any real data was run: a perfect reimplementation fails it about 4% of the
time. The first version (p > 0.05 per test, 2 misses) would have failed one about 53% of the
time.

The original vector layout is [p1 (N), p2 (N), G, sigma (N)], with p1, p2 = (w, I0) for MFM,
(wEE, wEI) for FIC and (a, omega) for Hopf (tzeng `models/CBIG_*.py`). `card_inputs` maps it
onto the reference cards, including MFM's shifted drive and FIC's solved wIE column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io
from scipy import stats

from delssome_fm.config import ReproduceConfig, ReproduceModel, SimConfig
from delssome_fm.data.empirical import FCD_BINS, N_REGIONS, read_csv_array
from delssome_fm.sim import fic
from delssome_fm.sim.cost import delssome_cost
from delssome_fm.sim.integrate import make_batch_simulate, rescale_sc
from delssome_fm.sim.summary import fcd_cdf, functional_connectivity
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.reference import load_reference

COMPONENTS: tuple[str, ...] = ("corr", "mean", "ks", "total")
MFM_DRIVE_SHIFT = 108.0 / 270.0   # card I = published I0 - b/a (mfm.yaml)


@dataclass(frozen=True)
class Original:
    """params: (S, 3N+1); corr, mean, ks: (S,) recorded test costs."""

    params: np.ndarray
    corr: np.ndarray
    mean: np.ndarray
    ks: np.ndarray

    def component(self, name: str) -> np.ndarray:
        return self.corr + self.mean + self.ks if name == "total" else getattr(self, name)


@dataclass(frozen=True)
class TestInputs:
    """sc: (N, N) rescaled to max 0.02; fc: (N, N); fcd_cdf: (B,) as in the original files."""

    sc: np.ndarray
    fc: np.ndarray
    fcd_cdf: np.ndarray


def load_original(cfg: ReproduceConfig, model: ReproduceModel) -> Original:
    path = cfg.export_dir / f"{model.name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run scripts/export_original_params.py first")
    d = np.load(path)
    if d["params"].shape != (cfg.n_sets, 3 * N_REGIONS + 1):
        raise ValueError(f"{path}: expected params {(cfg.n_sets, 3 * N_REGIONS + 1)}, "
                         f"found {d['params'].shape}")
    return Original(params=d["params"], corr=d["corr"], mean=d["mean"], ks=d["ks"])


def load_test_inputs(cfg: ReproduceConfig) -> TestInputs:
    """The exact files the original test used (5-significant-digit CSVs; docs/data.md §7)."""
    d = cfg.original_input_dir
    fcd = scipy.io.loadmat(d / "FCD_CDF_test.mat")["FCD_CDF"].reshape(-1)
    if fcd.shape != (FCD_BINS,):
        raise ValueError(f"FCD_CDF_test.mat: expected ({FCD_BINS},), found {fcd.shape}")
    sc = read_csv_array(d / "SC_test.csv", (N_REGIONS, N_REGIONS))
    return TestInputs(sc=np.asarray(rescale_sc(jnp.asarray(sc))),
                      fc=read_csv_array(d / "FC_test.csv", (N_REGIONS, N_REGIONS)), fcd_cdf=fcd)


def card_inputs(name: str, vector: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One original parameter vector (3N+1,) -> (theta (N, P), psi (K,)) for the card."""
    N = N_REGIONS
    p1, p2, G, sigma = vector[:N], vector[N:2 * N], float(vector[2 * N]), vector[2 * N + 1:]
    if name == "mfm":
        theta = np.stack([p1, p2 - MFM_DRIVE_SHIFT, sigma], axis=1)
    elif name == "fic":
        theta = np.stack([p1, p2, sigma, fic.solve_w_ie(p1, p2, G, sc)], axis=1)
    elif name == "hopf":
        theta = np.stack([p1, p2, sigma], axis=1)
    else:
        raise KeyError(f"no original layout for model {name!r}")
    return theta, np.array([G])


def initial_state(model: ReproduceModel) -> np.ndarray:
    x0 = np.array(model.initial_state, float)
    if model.name == "fic":
        x0[0] = fic.excitatory_steady_state()[0]  # the original starts S_E at S_E_ave
    return x0


def evaluate(cfg: ReproduceConfig, sim_cfg: SimConfig, model: ReproduceModel,
             original: Original, inputs: TestInputs, sets: range) -> dict[str, np.ndarray]:
    """Our costs for `sets`: arrays (len(sets), n_noise_repeats) per component, plus
    n_valid (same shape): how many of the n_dup simulations did not diverge."""
    card = load_reference(model.name)
    sim_cfg = SimConfig(dt=model.dt, tr=sim_cfg.tr, n_frames=sim_cfg.n_frames,
                        burn_in_frames=model.burn_in_frames,
                        divergence_bound=sim_cfg.divergence_bound,
                        fcd_window=sim_cfg.fcd_window, fcd_bins=sim_cfg.fcd_bins)
    simulate = make_batch_simulate(compile_card(card), sim_cfg)
    summarise = jax.jit(lambda y: jax.lax.map(
        lambda one: (functional_connectivity(one), fcd_cdf(one, sim_cfg.fcd_window,
                                                            sim_cfg.fcd_bins)), y, batch_size=8))
    inputs_per_set = [card_inputs(model.name, original.params[s], inputs.sc) for s in sets]
    x0 = initial_state(model)
    jobs = [(i, r, k) for i in range(len(sets)) for r in range(cfg.n_noise_repeats)
            for k in range(cfg.n_dup)]
    fc = np.zeros((len(sets), cfg.n_noise_repeats, N_REGIONS, N_REGIONS))
    hist = np.zeros((len(sets), cfg.n_noise_repeats, sim_cfg.fcd_bins))
    n_valid = np.zeros((len(sets), cfg.n_noise_repeats), dtype=int)
    root = jax.random.key(cfg.seed)
    for start in range(0, len(jobs), cfg.batch_size):
        chunk = jobs[start:start + cfg.batch_size]
        padded = chunk + [chunk[-1]] * (cfg.batch_size - len(chunk))  # keep shapes static
        keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(
            root, sets[i]), r), k) for i, r, k in padded])
        theta = jnp.asarray(np.stack([inputs_per_set[i][0] for i, _, _ in padded]))
        psi = jnp.asarray(np.stack([inputs_per_set[i][1] for i, _, _ in padded]))
        out = simulate(keys, jnp.asarray(np.tile(x0, (len(padded), N_REGIONS, 1))), theta, psi,
                       jnp.asarray(np.tile(inputs.sc, (len(padded), 1, 1))))
        fcs, cdfs = (np.asarray(a, np.float64) for a in summarise(out.observed))
        diverged = np.asarray(out.diverged)
        for j, (i, r, _) in enumerate(chunk):
            if not diverged[j]:
                fc[i, r] += fcs[j]
                hist[i, r] += cdfs[j]   # cumulative counts: summing them sums the histograms
                n_valid[i, r] += 1
    return _costs(fc, hist, n_valid, inputs)


def _costs(fc: np.ndarray, cdf: np.ndarray, n_valid: np.ndarray,
           inputs: TestInputs) -> dict[str, np.ndarray]:
    out = {c: np.full(n_valid.shape, np.nan) for c in COMPONENTS}
    with jax.enable_x64(True):  # costs in float64; simulations stay float32
        for idx in np.ndindex(*n_valid.shape):
            if n_valid[idx] == 0:
                continue
            cost = delssome_cost(jnp.asarray(fc[idx] / n_valid[idx]), jnp.asarray(cdf[idx]),
                                 jnp.asarray(inputs.fc), jnp.asarray(inputs.fcd_cdf))
            for c in COMPONENTS:
                out[c][idx] = float(getattr(cost, c))
    out["n_valid"] = n_valid
    return out


@dataclass(frozen=True)
class ComponentReport:
    component: str
    p_bias: float
    mean_difference: float          # our mean minus recorded, averaged over sets
    misses: int
    worst_set: int                  # index into the original seed1..seedN (0-based)
    worst_z: float
    worst_recorded: float
    worst_ours: float
    passed: bool

    def describe(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"{status} {self.component:5s} ours-recorded {self.mean_difference:+.4f} "
                f"(paired t p={self.p_bias:.3f}); misses {self.misses}; worst seed"
                f"{self.worst_set + 1}: recorded {self.worst_recorded:.4f}, ours "
                f"{self.worst_ours:.4f}, z={self.worst_z:+.2f}")


def judge(cfg: ReproduceConfig, original: Original,
          ours: dict[str, np.ndarray]) -> list[ComponentReport]:
    """Apply the agreed pass criterion to every component of one model. Sets with any repeat
    in which all n_dup runs diverged count as misses (they have no cost to compare)."""
    alpha = cfg.bias_alpha / (len(cfg.models) * len(COMPONENTS))
    reports = []
    for c in COMPONENTS:
        rec = original.component(c)
        mean, sd = np.nanmean(ours[c], axis=1), np.nanstd(ours[c], axis=1, ddof=1)
        missing = np.isnan(ours[c]).any(axis=1)
        z = np.where(missing, np.inf, (rec - mean) / sd)
        k = int(np.argmax(np.abs(z)))
        misses = int(np.sum(np.abs(z) > cfg.z_threshold))
        ok = ~missing
        p = float(stats.ttest_rel(mean[ok], rec[ok]).pvalue) if ok.sum() > 1 else 0.0
        reports.append(ComponentReport(
            component=c, p_bias=p, mean_difference=float(np.mean(mean[ok] - rec[ok])),
            misses=misses, worst_set=k, worst_z=float(z[k]), worst_recorded=float(rec[k]),
            worst_ours=float(mean[k]), passed=misses <= cfg.max_misses and p > alpha))
    return reports


def output_path(cfg: ReproduceConfig, model: str, sets: range) -> Path:
    return cfg.output_dir / f"{model}_sets{sets.start}-{sets.stop}.npz"


def load_results(cfg: ReproduceConfig, model: str) -> dict[str, np.ndarray]:
    """Concatenate every chunk written for `model`; all n_sets must be present exactly once."""
    files = sorted(cfg.output_dir.glob(f"{model}_sets*.npz"),
                   key=lambda p: int(p.stem.split("sets")[1].split("-")[0]))
    if not files:
        raise FileNotFoundError(f"no results for {model} in {cfg.output_dir}")
    parts: list[dict[str, Any]] = [dict(np.load(f)) for f in files]
    covered = np.concatenate([p["sets"] for p in parts])
    if sorted(covered.tolist()) != list(range(cfg.n_sets)):
        raise ValueError(f"{model}: result chunks cover sets {sorted(covered.tolist())}, "
                         f"expected 0..{cfg.n_sets - 1} exactly once")
    order = np.argsort(covered)
    return {k: np.concatenate([p[k] for p in parts])[order] for k in (*COMPONENTS, "n_valid")}
