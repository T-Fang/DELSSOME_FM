"""The synthetic corpus: screen sampled cards and simulate the valid ones (generation.md §6,
§7, §9), for three disjoint model populations: train, val and test.

A model is identified by (split, index). Everything about it is a deterministic function of
(seed, split, index), and each parameter draw of (seed, split, index, draw):

    1. the card (spec/sampler.py, stream = the split's code) and its connectome: one of the
       split's HCP-YA group SCs (train 64, val 14, test 13), uniformly, rescaled to max 0.02
    2. the screen: `screen_draws` short random draws. Rejected if every draw diverges or is
       flat (generation.md §6)
    3. the base draws 0 .. n_param_sets - 1 at full length, each summarised by
       sim/summary.py. The model is **valid** if at least one base draw is neither diverged
       nor flat. Only valid models' arrays are stored; divergent and flat draws of a valid
       model are stored too, labelled (generation.md §7)

Every candidate gets one log row (`CandidateLog`): the acceptance rate and mean
off-diagonal FC per model that brief §8 asks to log, and its validity.

Scaling up. The corpus of a split is "the first N valid models by index". More models: run
more candidate indices. More draws per model: `extend_models` simulates draws
first_draw .. for models already found valid, without re-screening. Draw j of a model is the
same whichever job, chunk or extension computes it.

Diverged and flat, precisely:
    diverged  any state or recorded value non-finite, or |value| > divergence_bound, at any
              frame boundary (sim/integrate.py); sticky for the rest of the run
    flat      every region's temporal SD of the recorded signal <= flat_rel_sd * max(1, |mean|)

This module does not decide job granularity or submit anything (scripts/ do).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from delssome_fm.config import CorpusConfig, DataConfig, SimConfig
from delssome_fm.data.empirical import N_REGIONS, load_reference_group
from delssome_fm.data.groups import SPLIT_NAMES, groups_from_config
from delssome_fm.sim.integrate import make_batch_simulate, rescale_sc
from delssome_fm.sim.summary import fcd_levels, summarize
from delssome_fm.spec.card import ModelCard, card_to_dict
from delssome_fm.spec.compile import CompiledModel, compile_card
from delssome_fm.spec.sampler import draw_parameters, sample_card

P_MAX = 5   # regional parameters: 1 + up to 4 (generation.md §4)
K_MAX = 2   # global parameters: one per channel, at most 2
SPLIT_STREAM = {name: code for code, name in enumerate(SPLIT_NAMES)}  # train 0, val 1, test 2

# Stream tags for np.random.default_rng([seed, stream, index, tag, ...]); the card itself
# uses [seed, stream, index] (sampler.sample_card).
_SC, _DRAW, _SCREEN = 1, 2, 3


@dataclass(frozen=True)
class CandidateLog:
    split: str
    index: int
    name: str
    sc_group: int              # index into the split's groups
    n_states: int
    n_regional: int
    n_global: int
    n_nodes: int
    kept: bool                 # passed the screen
    screen_diverged: int       # of screen_draws
    screen_flat: int           # of screen_draws
    n_draws: int               # base draws simulated (0 if not kept)
    n_draws_diverged: int
    n_draws_flat: int
    valid: bool                # kept, and >= 1 base draw neither diverged nor flat
    mean_fc: float             # mean off-diagonal FC over usable draws (screen if not kept)


def split_scs(data_cfg: DataConfig, split: str) -> np.ndarray:
    """(G, N, N) the split's group SCs, rescaled to max 0.02 as the simulator sees them."""
    if split not in SPLIT_STREAM:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, found {split!r}")
    groups = [g for g in groups_from_config(data_cfg) if g.split == split]
    return np.stack([np.asarray(rescale_sc(jnp.asarray(load_reference_group(data_cfg, g).sc)))
                     for g in groups])


def _key(cfg: CorpusConfig, split: str, index: int, *more: int) -> list[int]:
    return [cfg.seed, SPLIT_STREAM[split], index, *more]


def card_for(cfg: CorpusConfig, split: str, index: int) -> ModelCard:
    return sample_card(cfg.seed, SPLIT_STREAM[split], index)


def sc_group(cfg: CorpusConfig, split: str, index: int, n_groups: int) -> int:
    return int(np.random.default_rng(_key(cfg, split, index, _SC)).integers(n_groups))


def draws(cfg: CorpusConfig, card: ModelCard, split: str, index: int, tag: int, first: int,
          n: int):
    """theta (n, N, P), psi (n, K), x0 (n, N, V), keys (n,) for draws first..first+n-1."""
    thetas, psis, x0s = [], [], []
    for j in range(first, first + n):
        rng = np.random.default_rng(_key(cfg, split, index, tag, j))
        theta, psi = draw_parameters(card, rng, 1, N_REGIONS)
        thetas.append(theta[0])
        psis.append(psi[0])
        x0s.append(rng.uniform(-cfg.initial_state_scale, cfg.initial_state_scale,
                               (N_REGIONS, card.n_states)))
    base = jax.random.key(cfg.seed)
    for v in (SPLIT_STREAM[split], index, tag):
        base = jax.random.fold_in(base, v)
    keys = jnp.stack([jax.random.fold_in(base, j) for j in range(first, first + n)])
    return np.stack(thetas), np.stack(psis), np.stack(x0s), keys


def is_flat(observed: np.ndarray, rel_sd: float) -> np.ndarray:
    """observed: (B, T, N) -> (B,) True where every region's temporal SD is below the floor."""
    return flat_from_moments(observed.mean(axis=1), observed.std(axis=1), rel_sd)


def flat_from_moments(mean: np.ndarray, sd: np.ndarray, rel_sd: float) -> np.ndarray:
    """mean, sd: (B, N) per-region temporal moments -> (B,) the flat rule of `is_flat`."""
    return np.all(sd <= rel_sd * np.maximum(1.0, np.abs(mean)), axis=1)


def screen(cfg: CorpusConfig, sim_cfg: SimConfig, model: CompiledModel, split: str,
           index: int, sc: np.ndarray) -> tuple[bool, int, int, float]:
    """(kept, n_diverged, n_flat, mean off-diagonal FC over usable screen draws)."""
    short = SimConfig(dt=sim_cfg.dt, tr=sim_cfg.tr, n_frames=cfg.screen_frames,
                      burn_in_frames=cfg.screen_burn_in_frames,
                      divergence_bound=sim_cfg.divergence_bound,
                      fcd_window=sim_cfg.fcd_window, fcd_bins=sim_cfg.fcd_bins)
    theta, psi, x0, keys = draws(cfg, model.card, split, index, _SCREEN, 0, cfg.screen_draws)
    out = make_batch_simulate(model, short)(keys, x0, theta, psi,
                                            np.broadcast_to(sc, (len(keys), *sc.shape)))
    y, diverged = np.asarray(out.observed, np.float64), np.asarray(out.diverged)
    flat = ~diverged & is_flat(np.nan_to_num(y), cfg.flat_rel_sd)
    usable = ~diverged & ~flat
    mean_fc = (float(np.mean([_mean_off_diagonal(np.corrcoef(y[b].T))
                              for b in np.flatnonzero(usable)]))
               if usable.any() else float("nan"))
    return bool(usable.any()), int(diverged.sum()), int(flat.sum()), mean_fc


def _mean_off_diagonal(fc: np.ndarray) -> float:
    return float(fc[np.triu_indices(fc.shape[0], 1)].mean())


@lru_cache(maxsize=None)
def _summariser(cfg: SimConfig, stride: int):
    """jit(map(summarize)) for one batch; cached so it compiles once per process."""
    def one(y):
        s = summarize(y, cfg)
        return s._replace(fcd_cdf=fcd_levels(s.fcd_cdf, stride))
    return jax.jit(lambda ys: jax.lax.map(one, ys, batch_size=8))


def simulate_draws(cfg: CorpusConfig, sim_cfg: SimConfig, model: CompiledModel, split: str,
                   index: int, sc: np.ndarray, first: int, n: int) -> dict[str, np.ndarray]:
    """Draws first..first+n-1 of one model at full length, summarised: per-draw arrays
    (float32 statistics; diverged and flat flags; NaN statistics where diverged)."""
    theta, psi, x0, keys = draws(cfg, model.card, split, index, _DRAW, first, n)
    simulate = make_batch_simulate(model, sim_cfg)
    summarise = _summariser(sim_cfg, cfg.fcd_store_stride)
    parts = []
    for start in range(0, n, cfg.batch_size):
        sl = slice(start, start + cfg.batch_size)
        m = len(keys[sl])
        pad = cfg.batch_size - m
        # pad the last batch with copies of its final draw so shapes stay static
        k = keys[sl] if not pad else jnp.concatenate([keys[sl], jnp.repeat(keys[sl][-1:], pad, 0)])
        x0_b, theta_b, psi_b = (np.concatenate([a[sl]] + [a[sl][-1:]] * pad) if pad else a[sl]
                                for a in (x0, theta, psi))
        out = simulate(k, x0_b, theta_b, psi_b, np.broadcast_to(sc, (cfg.batch_size, *sc.shape)))
        s = summarise(out.observed)
        parts.append({"diverged": np.asarray(out.diverged)[:m],
                      **{name: np.asarray(v, np.float32)[:m] for name, v in s._asdict().items()}})
    res = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    res["flat"] = ~res["diverged"] & flat_from_moments(
        np.nan_to_num(res["regional"][..., 0]), np.nan_to_num(res["regional"][..., 1]),
        cfg.flat_rel_sd)
    for name in ("fc", "fcd_cdf", "regional", "pairwise", "fc_moments"):
        res[name][res["diverged"]] = np.nan
    res["theta"], res["psi"] = _pad(theta, P_MAX), _pad(psi, K_MAX)
    res["draw"] = np.arange(first, first + n)
    del res["fc"]  # pairwise holds arctanh(FC) upper triangle; fc_moments its summary
    return res


def _pad(a: np.ndarray, width: int) -> np.ndarray:
    if a.shape[-1] > width:
        raise ValueError(f"parameter count {a.shape[-1]} exceeds the corpus maximum {width}")
    out = np.full((*a.shape[:-1], width), np.nan, np.float32)
    out[..., :a.shape[-1]] = a
    return out


def _usable_mean_fc(res: dict[str, np.ndarray]) -> float:
    ok = ~res["diverged"] & ~res["flat"]
    return float(np.mean(res["fc_moments"][ok, 0])) if ok.any() else float("nan")


def run_candidates(cfg: CorpusConfig, sim_cfg: SimConfig, split: str, scs: np.ndarray,
                   start: int, stop: int) -> tuple[list[CandidateLog], dict[str, Any], list[dict]]:
    """Screen candidates start..stop-1 of `split` and simulate the base draws of the kept
    ones. Requires first_draw == 0 (base draws define validity).

    returns: (one log row per candidate,
              per-draw arrays of the valid models, concatenated, with `model` index,
              the valid models' cards as plain dicts)
    """
    if cfg.first_draw != 0:
        raise ValueError("screening runs simulate the base draws: first_draw must be 0; "
                         "use extend_models for further draws")
    logs, rows, cards = [], [], []
    for i in range(start, stop):
        card = card_for(cfg, split, i)
        model = compile_card(card)
        g = sc_group(cfg, split, i, len(scs))
        kept, n_div, n_flat, mean_fc = screen(cfg, sim_cfg, model, split, i, scs[g])
        n, d_div, d_flat, valid = 0, 0, 0, False
        if kept:
            res = simulate_draws(cfg, sim_cfg, model, split, i, scs[g], 0, cfg.n_param_sets)
            n, d_div, d_flat = len(res["draw"]), int(res["diverged"].sum()), int(res["flat"].sum())
            valid = d_div + d_flat < n
            mean_fc = _usable_mean_fc(res)
            if valid:
                res["model"] = np.full(n, i)
                rows.append(res)
                cards.append({"split": split, "index": i, "sc_group": g,
                              "card": card_to_dict(card)})
        logs.append(CandidateLog(split, i, card.name, g, card.n_states,
                                 len(card.regional_params), len(card.global_params),
                                 model.dag.n_nodes, kept, n_div, n_flat, n, d_div, d_flat, valid,
                                 mean_fc))
    arrays = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]} if rows else {}
    return logs, arrays, cards


def extend_models(cfg: CorpusConfig, sim_cfg: SimConfig, split: str, scs: np.ndarray,
                  indices: list[int]) -> dict[str, Any]:
    """Draws first_draw .. first_draw + n_param_sets - 1 of models already found valid (no
    screening): per-draw arrays concatenated, with `model` index."""
    if cfg.first_draw == 0:
        raise ValueError("extensions add draws after the base draws: first_draw must be > 0")
    rows = []
    for i in indices:
        card = card_for(cfg, split, i)
        g = sc_group(cfg, split, i, len(scs))
        res = simulate_draws(cfg, sim_cfg, compile_card(card), split, i, scs[g], cfg.first_draw,
                             cfg.n_param_sets)
        res["model"] = np.full(len(res["draw"]), i)
        rows.append(res)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]} if rows else {}


# ---------------------------------------------------------------- files


def chunk_paths(cfg: CorpusConfig, split: str, start: int, stop: int) -> dict[str, Path]:
    stem = (f"candidates{start:07d}-{stop:07d}_draws{cfg.first_draw}-"
            f"{cfg.first_draw + cfg.n_param_sets}")
    base = cfg.output_dir / split / stem
    return {"arrays": base.with_suffix(".npz"), "cards": base.with_suffix(".cards.jsonl"),
            "log": base.with_suffix(".log.jsonl"), "manifest": base.with_suffix(".manifest.json")}


def write_chunk(cfg: CorpusConfig, split: str, start: int, stop: int,
                logs: list[CandidateLog] | None, arrays: dict[str, Any],
                cards: list[dict] | None) -> dict[str, Path]:
    """Arrays always; the cards and per-candidate log for screening runs (None for
    extensions, whose models and logs exist already)."""
    paths = chunk_paths(cfg, split, start, stop)
    paths["arrays"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths["arrays"], **arrays)
    if cards is not None:
        paths["cards"].write_text("".join(json.dumps(c) + "\n" for c in cards))
    if logs is not None:
        paths["log"].write_text("".join(json.dumps(asdict(r)) + "\n" for r in logs))
    return paths


def read_logs(cfg: CorpusConfig, split: str) -> list[CandidateLog]:
    """All base-run log rows of a split, sorted by index; raises on a duplicated index."""
    rows = []
    for f in sorted((cfg.output_dir / split).glob("candidates*_draws0-*.log.jsonl")):
        rows.extend(CandidateLog(**json.loads(line)) for line in f.read_text().splitlines())
    rows.sort(key=lambda r: r.index)
    idx = [r.index for r in rows]
    if len(set(idx)) != len(idx):
        raise ValueError(f"{split}: some candidate indices are logged more than once")
    return rows


def corpus_status(cfg: CorpusConfig, split: str) -> dict[str, Any]:
    """Counts for one split: candidates screened, kept, valid; the contiguous covered range
    0..n_contiguous-1 (the corpus is the first N valid models within it); rates."""
    rows = read_logs(cfg, split)
    covered = {r.index for r in rows}
    n_contiguous = 0
    while n_contiguous in covered:
        n_contiguous += 1
    valid = [r.index for r in rows if r.valid]
    return {"split": split, "screened": len(rows), "n_contiguous": n_contiguous,
            "kept": sum(r.kept for r in rows), "valid": len(valid),
            "valid_in_contiguous": sum(i < n_contiguous for i in valid),
            "valid_rate": len(valid) / max(1, len(rows))}


def first_valid(cfg: CorpusConfig, split: str, n: int) -> list[int]:
    """The indices of the split's corpus: its first n valid models, all within the
    contiguously screened range (raises if that range does not hold n valid models yet)."""
    status = corpus_status(cfg, split)
    rows = [r.index for r in read_logs(cfg, split)
            if r.valid and r.index < status["n_contiguous"]]
    if len(rows) < n:
        raise ValueError(f"{split}: only {len(rows)} valid models in the contiguous range "
                         f"0..{status['n_contiguous'] - 1}, {n} requested")
    return rows[:n]


def acceptance_summary(logs: list[CandidateLog]) -> str:
    """One line: screen acceptance, validity, why candidates were rejected, and how many
    valid models look like slow drift (mean off-diagonal FC > 0.9, generation.md §4)."""
    kept = [r for r in logs if r.kept]
    valid = [r for r in logs if r.valid]
    rejected = [r for r in logs if not r.kept]
    all_diverged = sum(r.screen_flat == 0 for r in rejected)
    all_flat = sum(r.screen_diverged == 0 for r in rejected)
    drift = sum(r.mean_fc > 0.9 for r in valid)
    return (f"{len(valid)}/{len(logs)} valid ({len(valid) / max(1, len(logs)):.1%}), "
            f"{len(kept)} passed the screen; screen rejections: {all_diverged} all diverged, "
            f"{all_flat} all flat, {len(rejected) - all_diverged - all_flat} mixed; "
            f"valid with mean FC > 0.9: {drift}")
