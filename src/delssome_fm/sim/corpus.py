"""The stage-1 corpus: screen sampled cards and simulate the kept ones (generation.md §6, §7).

For each candidate index i in a range, deterministically from (seed, i):

    1. sample the card (spec/sampler.py) and pick its connectome: one of the 64 HCP-YA
       training group SCs, uniformly, rescaled to max 0.02 (generation.md §9)
    2. screen: `screen_draws` random parameter draws, `screen_frames` frames each. Reject if
       every draw diverges, or every draw's observed signal is flat (all regions' temporal
       SD <= flat_rel_sd * max(1, |mean|)). Nothing else (generation.md §6)
    3. kept: simulate draws first_draw .. first_draw + n_param_sets - 1 at full length and
       summarise each with sim/summary.py. Divergent draws are kept, labelled diverged,
       with NaN statistics (generation.md §7)

Every candidate, kept or not, gets one log row (`CandidateLog`): this is the acceptance rate
and the mean off-diagonal FC per model that brief §8 asks to log. Each draw's randomness is
keyed by (seed, i, draw), so splitting work across jobs, or appending draws later, gives
the same numbers.

This module does not decide job granularity or submit anything (scripts/build_corpus.py and
cluster/submit.py do).
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
from delssome_fm.data.groups import groups_from_config
from delssome_fm.sim.integrate import make_batch_simulate, rescale_sc
from delssome_fm.sim.summary import summarize
from delssome_fm.spec.card import ModelCard, card_to_dict
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.sampler import draw_parameters, sample_card

P_MAX = 5   # regional parameters: 1 + up to 4 (generation.md §4)
K_MAX = 2   # global parameters: one per channel, at most 2

# Stream tags for np.random.default_rng([seed, index, tag, ...]); the card itself uses
# [seed, index] (sampler.sample_card).
_SC, _DRAW, _SCREEN = 1, 2, 3


@dataclass(frozen=True)
class CandidateLog:
    index: int
    name: str
    sc_group: int              # index into the 64 training groups
    n_states: int
    n_regional: int
    n_global: int
    n_nodes: int
    kept: bool
    screen_diverged: int       # of screen_draws
    screen_flat: int           # of screen_draws
    mean_fc: float             # mean off-diagonal FC over kept draws (screen if rejected); nan if none
    n_draws_diverged: int      # of n_param_sets (0 if rejected)


def training_scs(data_cfg: DataConfig) -> np.ndarray:
    """(64, N, N) training group SCs, rescaled to max 0.02 as the simulator sees them."""
    groups = [g for g in groups_from_config(data_cfg) if g.split == "train"]
    return np.stack([np.asarray(rescale_sc(jnp.asarray(load_reference_group(data_cfg, g).sc)))
                     for g in groups])


def sc_group(cfg: CorpusConfig, index: int, n_groups: int) -> int:
    return int(np.random.default_rng([cfg.seed, index, _SC]).integers(n_groups))


def draws(cfg: CorpusConfig, card: ModelCard, index: int, tag: int, first: int, n: int):
    """theta (n, N, P), psi (n, K), x0 (n, N, V), keys (n,) for draws first..first+n-1."""
    thetas, psis, x0s = [], [], []
    for j in range(first, first + n):
        rng = np.random.default_rng([cfg.seed, index, tag, j])
        theta, psi = draw_parameters(card, rng, 1, N_REGIONS)
        thetas.append(theta[0])
        psis.append(psi[0])
        x0s.append(rng.uniform(-cfg.initial_state_scale, cfg.initial_state_scale,
                               (N_REGIONS, card.n_states)))
    base = jax.random.fold_in(jax.random.fold_in(jax.random.key(cfg.seed), index), tag)
    keys = jnp.stack([jax.random.fold_in(base, j) for j in range(first, first + n)])
    return np.stack(thetas), np.stack(psis), np.stack(x0s), keys


def is_flat(observed: np.ndarray, rel_sd: float) -> np.ndarray:
    """observed: (B, T, N) -> (B,) True where every region's temporal SD is below the floor."""
    sd = observed.std(axis=1)
    floor = rel_sd * np.maximum(1.0, np.abs(observed.mean(axis=1)))
    return np.all(sd <= floor, axis=1)


def screen(cfg: CorpusConfig, sim_cfg: SimConfig, model, index: int,
           sc: np.ndarray) -> tuple[bool, int, int, float]:
    """(kept, n_diverged, n_flat, mean off-diagonal FC over usable screen draws)."""
    short = SimConfig(dt=sim_cfg.dt, tr=sim_cfg.tr, n_frames=cfg.screen_frames,
                      burn_in_frames=cfg.screen_burn_in_frames,
                      divergence_bound=sim_cfg.divergence_bound,
                      fcd_window=sim_cfg.fcd_window, fcd_bins=sim_cfg.fcd_bins)
    theta, psi, x0, keys = draws(cfg, model.card, index, _SCREEN, 0, cfg.screen_draws)
    out = make_batch_simulate(model, short)(keys, x0, theta, psi,
                                            np.broadcast_to(sc, (len(keys), *sc.shape)))
    y, diverged = np.asarray(out.observed, np.float64), np.asarray(out.diverged)
    flat = ~diverged & is_flat(np.nan_to_num(y), cfg.flat_rel_sd)
    usable = ~diverged & ~flat
    mean_fc = (float(np.mean([_mean_off_diagonal(np.corrcoef(y[b].T)) for b in np.flatnonzero(usable)]))
               if usable.any() else float("nan"))
    kept = bool(usable.any())
    return kept, int(diverged.sum()), int(flat.sum()), mean_fc


def _mean_off_diagonal(fc: np.ndarray) -> float:
    return float(fc[np.triu_indices(fc.shape[0], 1)].mean())


@lru_cache(maxsize=None)
def _summariser(cfg: SimConfig, stride: int):
    """jit(map(summarize)) for one batch; cached so it compiles once per process."""
    def one(y):
        s = summarize(y, cfg)
        return s._replace(fcd_cdf=s.fcd_cdf[stride - 1::stride])
    return jax.jit(lambda ys: jax.lax.map(one, ys, batch_size=8))


def simulate_kept(cfg: CorpusConfig, sim_cfg: SimConfig, model, index: int,
                  sc: np.ndarray) -> dict[str, np.ndarray]:
    """Full-length simulations of draws first_draw.. for one kept model: per-draw arrays."""
    card = model.card
    theta, psi, x0, keys = draws(cfg, card, index, _DRAW, cfg.first_draw, cfg.n_param_sets)
    simulate = make_batch_simulate(model, sim_cfg)
    summarise = _summariser(sim_cfg, cfg.fcd_store_stride)
    parts = []
    for start in range(0, cfg.n_param_sets, cfg.batch_size):
        sl = slice(start, start + cfg.batch_size)
        n = len(keys[sl])
        pad = cfg.batch_size - n
        # pad the last batch with copies of its final draw so shapes stay static
        k = keys[sl] if not pad else jnp.concatenate([keys[sl], jnp.repeat(keys[sl][-1:], pad, 0)])
        x0_b, theta_b, psi_b = (np.concatenate([a[sl]] + [a[sl][-1:]] * pad) if pad else a[sl]
                                for a in (x0, theta, psi))
        out = simulate(k, x0_b, theta_b, psi_b, np.broadcast_to(sc, (cfg.batch_size, *sc.shape)))
        s = summarise(out.observed)
        parts.append({"diverged": np.asarray(out.diverged)[:n],
                      **{k: np.asarray(v, np.float32)[:n] for k, v in s._asdict().items()}})
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for k in ("fc", "fcd_cdf", "regional", "pairwise", "fc_moments"):
        res[k][res["diverged"]] = np.nan
    res["theta"], res["psi"] = _pad(theta, P_MAX), _pad(psi, K_MAX)
    res["draw"] = np.arange(cfg.first_draw, cfg.first_draw + cfg.n_param_sets)
    del res["fc"]  # pairwise holds arctanh(FC) upper triangle; fc_moments its summary
    return res


def _pad(a: np.ndarray, width: int) -> np.ndarray:
    if a.shape[-1] > width:
        raise ValueError(f"parameter count {a.shape[-1]} exceeds the corpus maximum {width}")
    out = np.full((*a.shape[:-1], width), np.nan, np.float32)
    out[..., :a.shape[-1]] = a
    return out


def run_candidates(cfg: CorpusConfig, sim_cfg: SimConfig, scs: np.ndarray, start: int,
                   stop: int) -> tuple[list[CandidateLog], dict[str, Any], list[dict]]:
    """Screen candidates start..stop-1 and simulate the kept ones.

    returns: (one log row per candidate,
              per-simulation arrays of the kept models concatenated, with `model` index,
              the kept cards as plain dicts)
    """
    logs, rows, cards = [], [], []
    for i in range(start, stop):
        card = sample_card(cfg.seed, i)
        model = compile_card(card)
        g = sc_group(cfg, i, len(scs))
        kept, n_div, n_flat, mean_fc = screen(cfg, sim_cfg, model, i, scs[g])
        n_draw_div = 0
        if kept:
            res = simulate_kept(cfg, sim_cfg, model, i, scs[g])
            res["model"] = np.full(len(res["draw"]), i)
            ok = ~res["diverged"]
            n_draw_div = int((~ok).sum())
            mean_fc = float(np.nanmean(res["fc_moments"][ok, 0])) if ok.any() else float("nan")
            rows.append(res)
            cards.append({"index": i, "sc_group": g, "card": card_to_dict(card)})
        logs.append(CandidateLog(i, card.name, g, card.n_states, len(card.regional_params),
                                 len(card.global_params), model.dag.n_nodes, kept, n_div,
                                 n_flat, mean_fc, n_draw_div))
    arrays = ({k: np.concatenate([r[k] for r in rows]) for k in rows[0]} if rows else {})
    return logs, arrays, cards


def chunk_paths(cfg: CorpusConfig, start: int, stop: int) -> dict[str, Path]:
    stem = f"candidates{start:07d}-{stop:07d}_draws{cfg.first_draw}-{cfg.first_draw + cfg.n_param_sets}"
    base = cfg.output_dir / stem
    return {"arrays": base.with_suffix(".npz"), "cards": base.with_suffix(".cards.jsonl"),
            "log": base.with_suffix(".log.jsonl"), "manifest": base.with_suffix(".manifest.json")}


def write_chunk(cfg: CorpusConfig, start: int, stop: int, logs: list[CandidateLog],
                arrays: dict[str, Any], cards: list[dict]) -> dict[str, Path]:
    paths = chunk_paths(cfg, start, stop)
    paths["arrays"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths["arrays"], **arrays)
    paths["cards"].write_text("".join(json.dumps(c) + "\n" for c in cards))
    paths["log"].write_text("".join(json.dumps(asdict(r)) + "\n" for r in logs))
    return paths


def acceptance_summary(logs: list[CandidateLog]) -> str:
    """One line: acceptance rate, why candidates were rejected, and how many kept models look
    like slow drift (mean off-diagonal FC > 0.9, generation.md §4)."""
    kept = [r for r in logs if r.kept]
    rejected = [r for r in logs if not r.kept]
    all_diverged = sum(r.screen_flat == 0 for r in rejected)
    all_flat = sum(r.screen_diverged == 0 for r in rejected)
    drift = sum(r.mean_fc > 0.9 for r in kept)
    return (f"{len(kept)}/{len(logs)} kept ({len(kept) / max(1, len(logs)):.1%}); rejected: "
            f"{all_diverged} all diverged, {all_flat} all flat, "
            f"{len(rejected) - all_diverged - all_flat} mixed; "
            f"kept with mean FC > 0.9: {drift}")
