"""Analytic features of a training store's rows (architecture.md §7.14): for each draw, the FC
and FCD of its linearised network (sim/linear.py), to be read by the network as inputs.

Per row of a store split:

    analytic_status    int8   STATUS_CODE: 0 stable, 1 unstable, 2 none (sim/linear.STATUSES),
                              3 a region without variance in the linearised network
    analytic_decay     f32    the slowest mode's decay rate, 1/s (NaN for status 2)
    analytic_pairwise  f32 E  arctanh of the analytic FC upper triangle, as the target
                              `pairwise` before z-scoring (summary.stable_arctanh); NaN unless 0
    analytic_fcd       f32 L  the FCD CDF at the store's L levels, the mean over n_surrogates
                              surrogate series (summary.summarize on each); NaN unless status 0

A row's parameters are the store's float32 theta and psi (what the simulator ran), its SC the
model's group SC rescaled as the simulator sees it, and its start the model's resting state
(sampler ModelSample.rest; zero without one). A surrogate's random numbers depend only on
(corpus seed, split, model index, draw, surrogate number), so any chunking gives the same rows.

`analytic_models` computes the rows of a range of store models; a job writes them with
`write_chunk`; `pack_analytic` checks that the chunks cover the split exactly once and writes
the four arrays into the store split. `analytic_single_model` does a split that holds one
model given by its card (the reference-model stores, sim/reference_eval.py) in one go.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from delssome_fm.config import CorpusConfig, SimConfig
from delssome_fm.sim.corpus import SPLIT_STREAM, model_sample
from delssome_fm.sim.integrate import rescale_sc
from delssome_fm.sim.linear import LinearModel, analytic_fc, surrogate
from delssome_fm.sim.summary import fcd_levels, stable_arctanh, summarize
from delssome_fm.spec.card import ModelCard
from delssome_fm.spec.compile import compile_card

ANALYTIC_TAG = 8          # random stream [seed, split, index, 8, draw, surrogate] (corpus.py: 1-7)
STATUS_CODE = {"stable": 0, "unstable": 1, "none": 2, "no_variance": 3}
ARRAYS = ("analytic_status", "analytic_decay", "analytic_pairwise", "analytic_fcd")


def analytic_models(corpus_cfg: CorpusConfig, sim_cfg: SimConfig, store_dir: Path, split: str,
                    first: int, stop: int, n_surrogates: int) -> dict[str, np.ndarray]:
    """The analytic features of store models first..stop-1 of `split`, one row per store row
    of theirs, in store order; also `row` (R,), the rows' store indices."""
    root = Path(store_dir) / split
    with np.load(root / "models.npz") as m:
        index, group, n_rows, offsets = m["index"], m["sc_group"], m["n_rows"], m["node_offset"]
    if not 0 <= first < stop <= len(index):
        raise ValueError(f"models {first}..{stop - 1} outside the split's 0..{len(index) - 1}")
    start = np.concatenate([[0], np.cumsum(n_rows)])
    theta, psi, draw = (np.load(root / f"{k}.npy", mmap_mode="r")
                        for k in ("theta", "psi", "draw"))
    n_levels = np.load(root / "fcd.npy", mmap_mode="r").shape[1]
    sc = np.load(root / "sc.npy")                    # (G, N, N), each divided by its maximum
    out = _empty(np.arange(start[first], start[stop]), sc.shape[1], n_levels)
    for m in range(first, stop):
        sample = model_sample(corpus_cfg, split, int(index[m]))
        card = sample.card
        compiled = compile_card(card)
        if compiled.dag.n_nodes != offsets[m + 1] - offsets[m]:
            raise ValueError(f"{split} model {m} (candidate {index[m]}): the sampled card has "
                             f"{compiled.dag.n_nodes} DAG nodes, the store "
                             f"{offsets[m + 1] - offsets[m]}: not the store's corpus")
        linear = LinearModel(compiled)
        P, K, V = len(card.regional_params), len(card.global_params), card.n_states
        x0 = np.tile(sample.rest if sample.rest is not None else np.zeros(V), (sc.shape[1], 1))
        for r in range(start[m], start[m + 1]):
            seed = [corpus_cfg.seed, SPLIT_STREAM[split], int(index[m]), ANALYTIC_TAG, int(draw[r])]
            _row(out, r - start[first], linear, x0, theta[r, :, :P], psi[r, :K],
                 np.asarray(rescale_sc(jnp.asarray(sc[group[m]]))), sim_cfg, seed, n_surrogates)
    return out


def _empty(rows: np.ndarray, n_regions: int, n_levels: int) -> dict[str, np.ndarray]:
    E = n_regions * (n_regions - 1) // 2
    return {"row": rows, "analytic_status": np.full(len(rows), -1, np.int8),
            "analytic_decay": np.full(len(rows), np.nan, np.float32),
            "analytic_pairwise": np.full((len(rows), E), np.nan, np.float32),
            "analytic_fcd": np.full((len(rows), n_levels), np.nan, np.float32)}


def analytic_single_model(root: Path, card: ModelCard, sim_cfg: SimConfig, seed: list[int],
                          n_surrogates: int) -> dict[str, float]:
    """Write the ARRAYS of a store split `root` whose rows are all draws of `card` (one model,
    one SC), started at zero; returns the share of each status. A surrogate's random numbers
    are [*seed, ANALYTIC_TAG, the row's draw, surrogate number]."""
    with np.load(root / "models.npz") as m:
        if len(m["index"]) != 1:
            raise ValueError(f"{root} holds {len(m['index'])} models, expected one")
    theta, psi, draw = (np.load(root / f"{k}.npy", mmap_mode="r")
                        for k in ("theta", "psi", "draw"))
    n_levels = np.load(root / "fcd.npy", mmap_mode="r").shape[1]
    sc = np.asarray(rescale_sc(jnp.asarray(np.load(root / "sc.npy")[0])))
    linear = LinearModel(compile_card(card))
    P, K, V = len(card.regional_params), len(card.global_params), card.n_states
    out = _empty(np.arange(len(theta)), len(sc), n_levels)
    for r in range(len(theta)):
        _row(out, r, linear, np.zeros((len(sc), V)), theta[r, :, :P], psi[r, :K], sc, sim_cfg,
             [*seed, ANALYTIC_TAG, int(draw[r])], n_surrogates)
    for k in ARRAYS:
        np.save(root / f"{k}.npy", out[k])
    status = out["analytic_status"]
    return {name: float((status == code).mean()) for name, code in STATUS_CODE.items()}


def _row(out: dict, k: int, linear: LinearModel, x0, theta, psi, sc, sim_cfg: SimConfig,
         seed: list[int], n_surrogates: int) -> None:
    lin = linear.linearise(x0, theta, psi, sc, sim_cfg.dt)
    out["analytic_status"][k] = STATUS_CODE[lin.status]
    out["analytic_decay"][k] = lin.decay
    if lin.status != "stable":
        return
    cov = lin.read @ lin.cov @ lin.read.T
    if not (np.isfinite(np.diag(cov)).all() and (np.diag(cov) > 0).all()):
        out["analytic_status"][k] = STATUS_CODE["no_variance"]
        return
    out["analytic_pairwise"][k] = np.asarray(
        stable_arctanh(jnp.asarray(analytic_fc(lin), jnp.float32)))
    fcd = _fcd(sim_cfg, sim_cfg.fcd_bins // out["analytic_fcd"].shape[1])
    cdfs = []
    for j in range(n_surrogates):
        y = surrogate(lin, sim_cfg, np.random.default_rng([*seed, j]))
        counts = np.asarray(fcd(jnp.asarray(y, jnp.float32)), np.float64)
        cdfs.append(counts / counts[-1])
    out["analytic_fcd"][k] = np.mean(cdfs, axis=0)   # NaN if a surrogate's FCD is undefined


@functools.lru_cache(maxsize=None)
def _fcd(sim_cfg: SimConfig, stride: int):
    """(T, N) series -> the FCD CDF's counts at the stored levels, compiled once per process:
    a job's memory then grows only by each model's own jitted functions (about 15 MB a model;
    clearing JAX's caches per model instead recompiled this and grew by 100 MB a model)."""
    return jax.jit(lambda y: fcd_levels(summarize(y, sim_cfg).fcd_cdf, stride))


def chunk_path(store_dir: Path, split: str, first: int, stop: int) -> Path:
    return Path(store_dir) / "analytic" / split / f"models{first:06d}-{stop:06d}.npz"


def write_chunk(store_dir: Path, split: str, first: int, stop: int, arrays: dict[str, np.ndarray],
                n_surrogates: int) -> Path:
    path = chunk_path(store_dir, split, first, stop)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")               # write then rename: never a partial chunk
    np.savez(tmp, first=first, stop=stop, n_surrogates=n_surrogates, **arrays)
    tmp.replace(path)
    return path


def pack_analytic(store_dir: Path, split: str) -> dict[str, int | float]:
    """Write the split's ARRAYS (<store_dir>/<split>/<name>.npy) from its chunks and return a
    summary. Raises unless the chunks cover every model once, hold every store row in order,
    and were made with one number of surrogates."""
    root = Path(store_dir) / split
    with np.load(root / "models.npz") as m:
        n_models, n = len(m["index"]), int(m["n_rows"].sum())
    chunks = sorted((Path(store_dir) / "analytic" / split).glob("models*.npz"))
    spans = [tuple(int(x) for x in c.stem[len("models"):].split("-")) for c in chunks]
    joined = spans and [a for a, _ in spans] == [0] + [b for _, b in spans[:-1]]
    if not joined or spans[-1][1] != n_models:
        raise ValueError(f"{split}: the chunks cover models {spans[:3]}...{spans[-3:]}, "
                         f"expected 0..{n_models} without gaps or overlaps")
    with np.load(chunks[0]) as d:
        shapes = {k: (n, *d[k].shape[1:]) for k in ARRAYS}
        dtypes = {k: d[k].dtype for k in ARRAYS}
        surrogates = int(d["n_surrogates"])
    out = {k: np.lib.format.open_memmap(root / f"{k}.tmp.npy", mode="w+", dtype=dtypes[k],
                                        shape=shapes[k]) for k in ARRAYS}
    at = 0
    for c in chunks:
        with np.load(c) as d:
            if int(d["n_surrogates"]) != surrogates:
                raise ValueError(f"{c}: {int(d['n_surrogates'])} surrogates, {chunks[0]} has "
                                 f"{surrogates}")
            rows = d["row"]
            if not np.array_equal(rows, np.arange(at, at + len(rows))):
                raise ValueError(f"{c}: rows {rows[:2]}..{rows[-2:]}, expected {at}.."
                                 f"{at + len(rows) - 1}")
            for k in ARRAYS:
                out[k][at:at + len(rows)] = d[k]
            at += len(rows)
    if at != n:
        raise ValueError(f"{split}: the chunks hold {at} rows, the store {n}")
    status = np.asarray(out["analytic_status"])
    if (status < 0).any():
        raise ValueError(f"{split}: {int((status < 0).sum())} rows were never computed")
    for k in ARRAYS:
        out[k].flush()
        (root / f"{k}.tmp.npy").replace(root / f"{k}.npy")
    summary = {"rows": n, "n_surrogates": surrogates,
               **{name: float((status == code).mean()) for name, code in STATUS_CODE.items()}}
    (root / "analytic_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
