"""CMA-ES around the coupled draws of rarely coupled models (decided 2026-11-04/05, project
lead; generation.md §9).

Why. Fitted MFM and FIC sit in a coupled regime that few random draws of their equations
reach, and the stage-1 network misses it (architecture.md §7.8). The corpus has the same
kind of model: 22% of models have 0 < coupled share <= 15% of their usable base draws. For
these models CMA-ES adds coupled draws near the empirical data, so the network sees the
regime it must predict there.

What is searched. The model (equations, constants and coupling gains: corpus gain_scope
"model") and its SC group stay fixed. CMA-ES searches the sinh policy's per-draw
hyperparameters of the P regional parameters (spec/sampler.py), the levels z_p and spreads
kappa_p, 2P numbers, in coordinates u in [0, 1] (pycma bounds):

    u_z = (z / Z + 1) / 2      (u_z = |z| / Z for a parameter whose sign is fixed: only |z|
                                matters there)
    u_kappa = kappa            (SPREAD_RANGE = [0, 1])

Every candidate gets fresh regional noise eps, initial state and simulation noise, from
[seed, split, index, CMA_TAG, run, generation, member] (numpy) and the same path folded into
the JAX key: its regional values are mu + kappa |mu| eps as for a base draw. CMA-ES itself
draws its normals from [seed, split, index, CMA_TAG, run] (pycma's `randn` option, so the
global numpy RNG is never used).

Objective. The original DELSSOME cost (sim/cost.py), total = (1 - r) + d + KS, of the
candidate's summary against the empirical FC and FCD of the model's SC group, from the
statistics the corpus stores: FC = tanh(pairwise), KS over the stored FCD levels
(summary.fcd_levels, also applied to the empirical CDF). Base draws are ranked by the same
function. A diverged, flat or non-finite candidate is told `failed_cost`.

Runs (decided 2026-11-05). Run 0 starts at the coupled base draw (usable, mean FC >
coupled_threshold) with the lowest total. A run fails when fewer than min_coupled_share of its
usable candidates are coupled (or none is usable): CMA-ES left the coupled regime. Only then
does the next run start, at the coupled base draw with the next lowest total, or at the lowest
again when there is none (then only CMA-ES's own randomness differs), up to max_runs runs.
The last run is kept, failed or not: its candidates are the model's data, and the earlier
(failed) runs' candidates are discarded.

Skipped (decided 2026-11-06, project lead). A model none of whose coupled base draws has a
finite cost gets no CMA-ES: there is no start. In practice its coupled draws all have an
empty FCD, which stage 1 drops too, so for training it is a never-coupled model. Its log row
says why (`skipped`); a chunk whose models were all skipped writes an arrays file with no
arrays.

Output. Every candidate of the kept run, with the per-draw arrays the corpus stores
(corpus.simulate_given), so training can use them as further draws of the model; plus model,
run, generation, member, level, spread and cost terms. One log row per run, kept or not.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from delssome_fm.config import CmaesConfig, CorpusConfig, DataConfig, SimConfig
from delssome_fm.data.empirical import N_REGIONS, load_reference_group
from delssome_fm.data.groups import groups_from_config
from delssome_fm.sim.corpus import (CMA_TAG, P_MAX, SPLIT_STREAM, base_level_spread, card_for,
                                    first_valid, jax_key, model_gains, sc_group, simulate_given)
from delssome_fm.sim.cost import ks_cost, upper_fc_cost
from delssome_fm.sim.integrate import make_batch_simulate
from delssome_fm.sim.summary import fcd_levels
from delssome_fm.spec.card import ModelCard
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.sampler import SINH_Z, regional_values, sign_fixed

with warnings.catch_warnings():  # pycma's plotting needs matplotlib, which we do not use
    warnings.filterwarnings("ignore", "Could not import matplotlib", UserWarning)
    import cma

COST_TERMS = ("corr", "mean", "ks")


# ---------------------------------------------------------------- which models


def usable_coupled(arrays: dict[str, np.ndarray], threshold: float) -> tuple[np.ndarray,
                                                                               np.ndarray]:
    """(usable, coupled) per draw: usable = neither diverged nor flat; coupled = usable and
    mean off-diagonal FC > threshold."""
    usable = ~arrays["diverged"] & ~arrays["flat"]
    with np.errstate(invalid="ignore"):
        return usable, usable & (arrays["fc_moments"][:, 0] > threshold)


def base_chunks(corpus_cfg: CorpusConfig, split: str) -> list[tuple[int, int, Path]]:
    """(start, stop, arrays path) of the split's screening chunks, sorted."""
    return list(_base_chunks(corpus_cfg.output_dir, split))


@lru_cache(maxsize=None)
def _base_chunks(output_dir: Path, split: str) -> tuple[tuple[int, int, Path], ...]:
    out = []
    for f in (output_dir / split).glob("candidates*_draws0-*.npz"):
        m = re.fullmatch(r"candidates(\d{7})-(\d{7})_draws0-\d+\.npz", f.name)
        if m is None:
            raise ValueError(f"unexpected corpus file name {f.name}")
        out.append((int(m[1]), int(m[2]), f))
    return tuple(sorted(out))


def plan_split(cfg: CmaesConfig, corpus_cfg: CorpusConfig, split: str) -> list[dict[str, int]]:
    """The split's corpus models (its first n_models valid) with 0 < coupled share of usable
    base draws <= max_coupled_share, in index order: {index, usable, coupled}."""
    keep = set(first_valid(corpus_cfg, split, getattr(cfg.n_models, split)))
    found, plan = set(), []
    for start, stop, path in base_chunks(corpus_cfg, split):
        if not keep & set(range(start, stop)):
            continue
        with np.load(path) as d:
            arrays = {k: d[k] for k in ("model", "diverged", "flat", "fc_moments")}
        usable, coupled = usable_coupled(arrays, cfg.coupled_threshold)
        for i in np.unique(arrays["model"]):
            if int(i) not in keep:
                continue
            found.add(int(i))
            sel = arrays["model"] == i
            n_usable, n_coupled = int(usable[sel].sum()), int(coupled[sel].sum())
            if 0 < n_coupled <= cfg.max_coupled_share * n_usable:
                plan.append({"index": int(i), "usable": n_usable, "coupled": n_coupled})
    if found != keep:
        raise ValueError(f"{split}: {len(keep - found)} corpus models have no arrays, e.g. "
                         f"{sorted(keep - found)[:5]}")
    return sorted(plan, key=lambda r: r["index"])


def plan_path(cfg: CmaesConfig, split: str) -> Path:
    return cfg.output_dir / split / "plan.json"


def _selection(cfg: CmaesConfig, split: str) -> dict[str, Any]:
    """The config fields that decide a split's plan."""
    return {"corpus_config": str(cfg.corpus_config), "n_models": getattr(cfg.n_models, split),
            "coupled_threshold": cfg.coupled_threshold,
            "max_coupled_share": cfg.max_coupled_share}


def write_plan(cfg: CmaesConfig, split: str, models: list[dict[str, int]]) -> Path:
    path = plan_path(cfg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"selection": _selection(cfg, split), "models": models}))
    return path


def read_plan(cfg: CmaesConfig, split: str) -> list[dict[str, int]]:
    """The plan `write_plan` wrote; raises if it was made with other selection fields."""
    path = plan_path(cfg, split)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run scripts/cmaes_data.py plan first")
    plan = json.loads(path.read_text())
    if plan["selection"] != _selection(cfg, split):
        raise ValueError(f"{path} was made with {plan['selection']}, the config has "
                         f"{_selection(cfg, split)}")
    return plan["models"]


def model_arrays(corpus_cfg: CorpusConfig, split: str, index: int) -> dict[str, np.ndarray]:
    """The stored base draws of one valid model, in draw order."""
    hits = [p for a, b, p in base_chunks(corpus_cfg, split) if a <= index < b]
    if len(hits) != 1:
        raise ValueError(f"{split} model {index}: {len(hits)} screening chunks hold it")
    with np.load(hits[0]) as d:
        sel = d["model"] == index
        if not sel.any():
            raise ValueError(f"{split} model {index} is not valid in {hits[0]}")
        out = {k: d[k][sel] for k in d.files}
    if not np.array_equal(out["draw"], np.arange(len(out["draw"]))):
        raise ValueError(f"{split} model {index}: stored draws are not 0..{len(out['draw']) - 1}")
    return out


# ---------------------------------------------------------------- cost


def empirical(data_cfg: DataConfig, split: str, group: int, stride: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """The split's group `group` (sc_group order = split_scs order): upper-triangle FC (E,)
    and the FCD CDF at the stored levels (L,)."""
    groups = [g for g in groups_from_config(data_cfg) if g.split == split]
    data = load_reference_group(data_cfg, groups[group])
    rows, cols = np.triu_indices(N_REGIONS, 1)
    return data.fc[rows, cols], np.asarray(fcd_levels(data.fcd_cdf, stride))


@jax.jit
def _costs(fc_upper, cdf, emp_fc, emp_cdf):
    corr, mean = jax.vmap(upper_fc_cost, (0, None))(fc_upper, emp_fc)
    return jnp.stack([corr, mean, jax.vmap(ks_cost, (0, None))(cdf, emp_cdf)], axis=1)


def draw_costs(arrays: dict[str, np.ndarray], emp: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """(B, 3) the cost terms (COST_TERMS) of stored draws against `emp` (`empirical`), in
    float64; NaN where the draw is diverged."""
    with jax.enable_x64(True):
        fc = np.tanh(arrays["pairwise"].astype(np.float64))
        out = _costs(jnp.asarray(fc), jnp.asarray(arrays["fcd_cdf"], jnp.float64),
                     jnp.asarray(emp[0], jnp.float64), jnp.asarray(emp[1], jnp.float64))
        return np.asarray(out)


def start_draws(total: np.ndarray, coupled: np.ndarray, runs: int) -> list[int]:
    """Run k's start: the coupled draw with the k-th lowest finite total (ties: lower draw
    index), or the lowest when there are fewer than k + 1 (module docstring)."""
    candidates = np.flatnonzero(coupled & np.isfinite(total))
    if not len(candidates):
        raise ValueError("no coupled draw with a finite cost")
    ranked = candidates[np.argsort(total[candidates], kind="stable")]
    return [int(ranked[k]) if k < len(ranked) else int(ranked[0]) for k in range(runs)]


# ---------------------------------------------------------------- coordinates


def encode(card: ModelCard, z: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """(P,), (P,) -> u (2P,) in [0, 1]: levels then spreads (module docstring)."""
    u_z = np.where(sign_fixed(card), np.abs(z) / SINH_Z, (z / SINH_Z + 1) / 2)
    return np.concatenate([u_z, kappa])


def decode(card: ModelCard, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u (..., 2P) -> z (..., P), kappa (..., P); the inverse of `encode` on [0, 1]."""
    u = np.asarray(u, np.float64)
    P = len(card.regional_params)
    if u.shape[-1] != 2 * P or np.any((u < 0) | (u > 1)):
        raise ValueError(f"u must have {2 * P} entries in [0, 1], found shape {u.shape}, "
                         f"range [{u.min()}, {u.max()}]")
    u_z = u[..., :P]
    z = np.where(sign_fixed(card), u_z * SINH_Z, (2 * u_z - 1) * SINH_Z)
    return z, u[..., P:]


# ---------------------------------------------------------------- one model


def candidates(corpus_cfg: CorpusConfig, card: ModelCard, split: str, index: int, run: int,
               generation: int, z: np.ndarray, kappa: np.ndarray):
    """theta (n, N, P), x0 (n, N, V), keys (n,) of one generation's candidates (levels z and
    spreads kappa (n, P)), fresh noise each (module docstring)."""
    thetas, x0s = [], []
    for k in range(len(z)):
        rng = np.random.default_rng([corpus_cfg.seed, SPLIT_STREAM[split], index, CMA_TAG,
                                     run, generation, k])
        thetas.append(regional_values(card, z[k:k + 1], kappa[k:k + 1], rng, N_REGIONS)[0])
        x0s.append(rng.uniform(-corpus_cfg.initial_state_scale, corpus_cfg.initial_state_scale,
                               (N_REGIONS, card.n_states)))
    base = jax_key(corpus_cfg, split, index, CMA_TAG, run, generation)
    keys = jnp.stack([jax.random.fold_in(base, k) for k in range(len(z))])
    return np.stack(thetas), np.stack(x0s), keys


def strategy(cfg: CmaesConfig, u0: np.ndarray, rng: np.random.Generator
             ) -> cma.CMAEvolutionStrategy:
    """pycma on [0, 1]^2P from u0, drawing its normals from rng only (randn option)."""
    return cma.CMAEvolutionStrategy(
        u0, cfg.sigma0, {"bounds": [0, 1], "popsize": cfg.population, "maxiter": cfg.generations,
                         "randn": lambda *shape: rng.standard_normal(shape), "verbose": -9})


def _pad(a: np.ndarray, width: int) -> np.ndarray:
    out = np.full((*a.shape[:-1], width), np.nan, np.float32)
    out[..., :a.shape[-1]] = a
    return out


def run_model(cfg: CmaesConfig, corpus_cfg: CorpusConfig, sim_cfg: SimConfig,
              data_cfg: DataConfig, split: str, index: int, scs: np.ndarray
              ) -> tuple[dict[str, np.ndarray] | None, list[dict[str, Any]]]:
    """The CMA-ES runs of one model (module docstring): (the kept run's per-candidate
    arrays, one log row per run), or (None, one log row) for a skipped model."""
    if corpus_cfg.gain_scope != "model" or corpus_cfg.parameter_policy != "sinh":
        raise ValueError("CMA-ES needs a corpus with gain_scope model and the sinh policy, "
                         f"found {corpus_cfg.gain_scope!r}, {corpus_cfg.parameter_policy!r}")
    card = card_for(corpus_cfg, split, index)
    model = compile_card(card)
    g = sc_group(corpus_cfg, split, index, len(scs))
    emp = empirical(data_cfg, split, g, corpus_cfg.fcd_store_stride)
    base = model_arrays(corpus_cfg, split, index)
    gains = model_gains(corpus_cfg, card, split, index)
    K = len(card.global_params)
    if not np.allclose(base["psi"][:, :K], gains.astype(np.float32), rtol=1e-6):
        raise ValueError(f"{split} model {index}: stored gains differ from model_gains")
    usable, coupled = usable_coupled(base, cfg.coupled_threshold)
    base_cost = draw_costs(base, emp)
    if not (coupled & np.isfinite(base_cost.sum(axis=1))).any():
        return None, [{"split": split, "index": index,
                       "skipped": "no coupled base draw with a finite cost (an empty FCD)",
                       "base_usable": int(usable.sum()), "base_coupled": int(coupled.sum()),
                       "kept": False}]
    starts = start_draws(base_cost.sum(axis=1), coupled, cfg.max_runs)

    sim_corpus = dataclasses.replace(corpus_cfg, batch_size=cfg.population)
    simulate = make_batch_simulate(model, sim_cfg)
    psi = np.broadcast_to(gains, (cfg.population, K))
    logs = []
    for run, j in enumerate(starts):
        t0 = time.time()
        parts = []
        z0, kappa0, theta_j = base_level_spread(corpus_cfg, card, split, index, j)
        if not np.allclose(_pad(theta_j, P_MAX), base["theta"][j], rtol=1e-5, equal_nan=True):
            raise ValueError(f"{split} model {index} draw {j}: stored theta differs from its "
                             f"replayed levels and spreads")
        es = strategy(cfg, encode(card, z0, kappa0), np.random.default_rng(
            [corpus_cfg.seed, SPLIT_STREAM[split], index, CMA_TAG, run]))
        history = []
        generation = 0
        while not es.stop():
            u = np.asarray(es.ask())
            z, kappa = decode(card, u)
            theta, x0, keys = candidates(corpus_cfg, card, split, index, run, generation, z,
                                         kappa)
            res = simulate_given(sim_corpus, sim_cfg, model, theta, psi, x0, keys, scs[g],
                                 simulate)
            cost = draw_costs(res, emp)
            ok, hot = usable_coupled(res, cfg.coupled_threshold)
            total = cost.sum(axis=1)
            told = np.where(ok & np.isfinite(total), total, cfg.failed_cost)
            es.tell(list(u), told.tolist())
            n = len(u)
            res.update(model=np.full(n, index), run=np.full(n, run),
                       generation=np.full(n, generation), member=np.arange(n),
                       level=_pad(z, P_MAX), spread=_pad(kappa, P_MAX),
                       cost=cost.astype(np.float32))
            parts.append(res)
            history.append({"best": float(told.min()), "median": float(np.median(told)),
                            "coupled": int(hot.sum()), "failed": int((told == cfg.failed_cost).sum()),
                            "sigma": float(es.sigma)})
            generation += 1
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        ok, hot = usable_coupled(arrays, cfg.coupled_threshold)
        succeeded = bool(ok.any() and hot.sum() >= cfg.min_coupled_share * ok.sum())
        logs.append({
            "split": split, "index": index, "run": run, "start_draw": j,
            "start_cost": dict(zip(COST_TERMS, base_cost[j].tolist())),
            "base_usable": int(usable.sum()), "base_coupled": int(coupled.sum()),
            "base_best_total": float(np.nanmin(np.where(usable, base_cost.sum(axis=1), np.nan))),
            "generations": generation, "stop": {k: str(v) for k, v in es.stop().items()},
            "usable": int(ok.sum()), "coupled": int(hot.sum()), "succeeded": succeeded,
            "kept": False, "history": history, "wall_time_s": time.time() - t0})
        if succeeded:
            break
    logs[-1]["kept"] = True
    return arrays, logs


# ---------------------------------------------------------------- files


def chunk_paths(cfg: CmaesConfig, split: str, first: int, stop: int) -> dict[str, Path]:
    """Files of plan positions first..stop-1 (not model indices)."""
    base = cfg.output_dir / split / f"plan{first:06d}-{stop:06d}"
    return {"arrays": base.with_suffix(".npz"), "log": base.with_suffix(".log.jsonl"),
            "manifest": base.with_suffix(".manifest.json")}


def run_chunk(cfg: CmaesConfig, corpus_cfg: CorpusConfig, sim_cfg: SimConfig,
              data_cfg: DataConfig, split: str, first: int, stop: int, scs: np.ndarray
              ) -> dict[str, Path]:
    """CMA-ES for plan positions first..stop-1; writes the arrays (none if every model was
    skipped) and the run log."""
    plan = read_plan(cfg, split)
    if not 0 <= first < stop <= len(plan):
        raise ValueError(f"plan positions {first}..{stop - 1} outside 0..{len(plan) - 1}")
    parts, logs = [], []
    for row in plan[first:stop]:
        arrays, rows = run_model(cfg, corpus_cfg, sim_cfg, data_cfg, split, row["index"], scs)
        if arrays is not None:
            parts.append(arrays)
        logs.extend(rows)
    paths = chunk_paths(cfg, split, first, stop)
    paths["arrays"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths["arrays"], **({k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                                 if parts else {}))
    paths["log"].write_text("".join(json.dumps(r) + "\n" for r in logs))
    return paths
