"""Pack one split of the corpus into the stage-1 training store: flat arrays that a data
loader can read at random, plus each model's equation graph as tensors.

Which draws become training rows (architecture.md §5.3): the draws of the split's corpus
models (its first n valid models, `corpus.first_valid`) that are neither diverged nor flat
and whose FCD is complete (every window pair present; decided 2026-10-14). A draw whose
targets are not all finite after the transforms below is also left out and counted.

Targets are stored transformed but not z-scored; `normalisation` computes the z-score
statistics from the train split, and the data loader applies them (decided 2026-10-14):

    regional (N, 6)   asinh(mean / SD), log10 SD, asinh(skew), log(excess kurtosis + 3),
                      FC node strength, lag-1 autocorrelation
    pairwise (2278,)  stable arctanh of the FC upper triangle, as stored by the corpus
    fcd (L,)          the FCD CDF at its L = 100 stored levels, divided by the number of
                      window pairs
    fc_moments (3,)   mean, log10 SD, skewness of the FC upper triangle
    catch24 (N, 24)   the catch24 features of each region (sim/summary.py), when the corpus
                      has them (2026-10-25), each made roughly symmetric (CATCH24_TRANSFORMS):
                      log10 for lags, counts and other positive heavy-tailed features,
                      asinh(trev / 0.01), the regional transforms for the mean and SD, the
                      rest (bounded) unchanged

The scale-free statistics are transformed because observables have arbitrary units: raw
regional SD spans 15 orders of magnitude, and a plain z-score would collapse most values to
one number while a few extremes dominated the loss.

Store layout, under <store_dir>/<split>/ (rows are ordered by model, then draw):

    model.npy (R,) int32       row of the model in models.npz
    draw.npy (R,) int32
    theta.npy (R, N, 5), psi.npy (R, 2)            float32, NaN beyond the model's P, K
    regional.npy, pairwise.npy, fcd.npy, fc_moments.npy   float32 targets
    catch24.npy (R, N, 24)     float32, only for corpora that store catch24
    fc_net.npy (R, 21)         float32, stable arctanh of the Yeo-7 network FC, written
                               afterwards by pack_network_fc (2026-10-26)
    models.npz                 one entry per model: corpus index, SC group, sizes, graph roles
    node_features.npy (sum M, 21), node_degree.npy (sum M, 2)    concatenated over models
    distances.npy (sum M^2,) int16                 each model's (M, M) shortest paths, flat
    sc.npy (G, N, N)           the split's group SCs, each divided by its maximum
    summary.json               row counts, why draws were left out, and the corpus's
                               parameter_policy (spec/sampler.py), which decides how the data
                               loader encodes theta and psi (train/data.encode_parameters)

This module only reads the corpus and writes the store. It does not know about torch.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from delssome_fm.config import CorpusConfig, GeneralisationConfig, SimConfig, TrainingDataConfig
from delssome_fm.sim.corpus import K_MAX, P_MAX, first_valid
from delssome_fm.sim.generalisation import CELLS, read_sample
from delssome_fm.spec.card import card_from_dict
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.sampler import POLICIES
from delssome_fm.spec.dag import Dag, node_features, shortest_paths

N_REGIONS = 68
V_MAX = 6          # state variables per model (generation.md §2)
C_MAX = 2          # coupling channels per model
TARGETS = ("regional", "pairwise", "fcd", "fc_moments")
OPTIONAL_TARGETS = ("catch24", "fc_net")   # catch24 when the corpus has it; fc_net added later
ROW_ARRAYS = ("model", "draw", "theta", "psi", *TARGETS)


# ---------------------------------------------------------------- targets


def transform_regional(regional: np.ndarray) -> np.ndarray:
    """(..., N, 6) raw regional statistics -> the transformed columns of the module docstring."""
    mean, sd, skew, excess_kurtosis = (regional[..., j].astype(np.float64) for j in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.stack([np.arcsinh(mean / sd), np.log10(sd), np.arcsinh(skew),
                        np.log(excess_kurtosis + 3.0), regional[..., 4], regional[..., 5]], axis=-1)
    return out.astype(np.float32)


# per catch24 feature (sim/summary.CATCH24_NAMES order): how it is transformed. Chosen from
# the feature ranges in the sinh corpus (2026-10-25): lags and counts span 1 to ~1200 frames.
CATCH24_TRANSFORMS = (
    "raw", "raw", "raw", "raw",          # histogram modes, outlier-include (about -4 .. 4)
    "log10", "log10",                    # first 1/e crossing and first minimum of the ACF
    "raw", "log10",                      # fft area below 1/5 band (0..1); wq_50 (0.003..3)
    "raw", "log10",                      # local-mean stde (0..1.5); taurat (0.001..2.6)
    "raw",                               # pNN40 (0..1)
    "log10", "log10p1",                  # longest stretches of 1s (1..), of 0s (0..)
    "raw", "raw",                        # motif entropy (0..2.2), AMI at lag 2 (0..1.6)
    "asinh_trev",                        # time-reversal statistic, signed, IQR ~ +-0.01
    "log10",                             # first minimum of the AMI (2..40)
    "log10_eps",                         # transition-matrix covariance (0..0.33)
    "log10p1",                           # Wang periodicity (0..400)
    "log10",                             # Embed2 exponential-fit mean difference (0.005..1e6)
    "raw", "raw",                        # fluctuation-analysis proportions (0.25..0.75)
    "mean_over_sd", "log10")             # DN_mean (as asinh(mean / SD)), DN_Spread_std


def transform_catch24(catch24: np.ndarray) -> np.ndarray:
    """(..., N, 24) raw catch24 features -> the transformed features (CATCH24_TRANSFORMS)."""
    x = catch24.astype(np.float64)
    out = np.empty_like(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j, kind in enumerate(CATCH24_TRANSFORMS):
            v = x[..., j]
            out[..., j] = {"raw": lambda: v, "log10": lambda: np.log10(v),
                           "log10p1": lambda: np.log10(1.0 + v),
                           "log10_eps": lambda: np.log10(v + 1e-5),
                           "asinh_trev": lambda: np.arcsinh(v / 0.01),
                           "mean_over_sd": lambda: np.arcsinh(v / x[..., 23])}[kind]()
    return out.astype(np.float32)


def transform_fc_moments(fc_moments: np.ndarray) -> np.ndarray:
    """(..., 3) mean, SD, skewness of the FC upper triangle -> mean, log10 SD, skewness."""
    out = fc_moments.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[..., 1] = np.log10(out[..., 1])
    return out.astype(np.float32)


def n_window_pairs(sim_cfg: SimConfig) -> int:
    """Number of upper-triangle FCD entries: C(W, 2) with W sliding windows."""
    windows = sim_cfg.n_frames - sim_cfg.fcd_window + 1
    return windows * (windows - 1) // 2


# ---------------------------------------------------------------- graphs


def graph_arrays(dag: Dag) -> dict[str, np.ndarray]:
    """The tensors the template encoder reads for one model (architecture.md §2):
    features (M, 21), degree (M, 2) in/out, distances (M, M), and the role indices."""
    edges = dag.edges()
    degree = np.zeros((dag.n_nodes, 2), dtype=np.int16)
    np.add.at(degree[:, 0], edges[:, 1], 1)   # in-degree: parents of a node
    np.add.at(degree[:, 1], edges[:, 0], 1)   # out-degree: its operands
    return {"features": node_features(dag), "degree": degree,
            "distances": shortest_paths(dag).astype(np.int16),
            "regional_par": _pad(dag.regional_par_nodes, P_MAX),
            "global_par": _pad(dag.global_par_nodes, K_MAX),
            "deriv_roots": _pad(dag.deriv_roots, V_MAX),
            "agg": _pad(dag.agg_nodes, C_MAX), "out_root": np.int32(dag.out_root)}


def _pad(values: tuple[int, ...], width: int) -> np.ndarray:
    if len(values) > width:
        raise ValueError(f"{len(values)} entries, at most {width} expected")
    return np.array(list(values) + [-1] * (width - len(values)), dtype=np.int32)


# ---------------------------------------------------------------- packing


def pack_split(corpus_cfg: CorpusConfig, sim_cfg: SimConfig, cfg: TrainingDataConfig,
               split: str, scs: np.ndarray) -> dict[str, Any]:
    """Write <store_dir>/<split>/ (module docstring) and return its summary.
    scs: (G, N, N) the split's group SCs (corpus.split_scs)."""
    members = first_valid(corpus_cfg, split, getattr(cfg.n_models, split))
    chunks = sorted((corpus_cfg.output_dir / split).glob("candidates*_draws0-*.npz"))
    return pack_chunks(chunks, members, n_window_pairs(sim_cfg), cfg.store_dir / split, scs,
                       {"split": split, "last_candidate": members[-1],
                        "parameter_policy": corpus_cfg.parameter_policy})


def pack_chunks(chunks: list[Path], members: list[int], full: int, out: Path, scs: np.ndarray,
                info: dict[str, Any]) -> dict[str, Any]:
    """Pack the draws of `members` found in `chunks` (arrays .npz with a `model` column, each
    beside a .cards.jsonl) into a store split at `out`. full: the number of FCD window pairs
    of a complete FCD. scs: (G, N, N), indexed by the cards' sc_group. `info` is added to
    summary.json and must name the corpus's parameter_policy. Used by pack_split, the
    generalisation diagnostic and the reference test."""
    if info.get("parameter_policy") not in POLICIES:
        raise ValueError(f"info must give parameter_policy, one of {POLICIES}, found {info}")
    keep, summary = _select_rows(chunks, np.array(members), full)
    out.mkdir(parents=True, exist_ok=True)
    rows = _write_rows(chunks, keep, members, full, out)
    summary.update(rows)
    _write_models(chunks, members, rows["rows_per_model"], out)
    np.save(out / "sc.npy", (scs / scs.max(axis=(1, 2), keepdims=True)).astype(np.float32))
    summary = {k: v for k, v in summary.items() if k != "rows_per_model"}
    summary.update(n_models=len(members), **info)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _select_rows(chunks: list[Path], members: np.ndarray, full: int,
                 ) -> tuple[dict[Path, np.ndarray], dict[str, int]]:
    """Pass 1, flags only: per chunk, the boolean mask of rows to keep."""
    keep, counts = {}, dict(draws=0, diverged=0, flat=0, incomplete_fcd=0, non_finite=0)
    for path in chunks:
        with np.load(path) as d:
            if "model" not in d.files:
                continue
            ours = np.isin(d["model"], members)
            usable = ours & ~d["diverged"] & ~d["flat"]
            complete = d["fcd_cdf"][:, -1] == full
            finite = (np.isfinite(transform_regional(d["regional"])).all(axis=(1, 2))
                      & np.isfinite(transform_fc_moments(d["fc_moments"])).all(axis=1))
            if "catch24" in d.files:
                finite &= np.isfinite(transform_catch24(d["catch24"])).all(axis=(1, 2))
            counts["draws"] += int(ours.sum())
            counts["diverged"] += int((ours & d["diverged"]).sum())
            counts["flat"] += int((ours & d["flat"]).sum())
            counts["incomplete_fcd"] += int((usable & ~complete).sum())
            counts["non_finite"] += int((usable & complete & ~finite).sum())
            keep[path] = usable & complete & finite
    return keep, counts


def _write_rows(chunks: list[Path], keep: dict[Path, np.ndarray], members: list[int],
                full: int, out: Path) -> dict[str, Any]:
    """Pass 2: the row arrays, written straight into memory-mapped .npy files."""
    n = int(sum(m.sum() for m in keep.values()))
    if n == 0:
        raise ValueError(f"{out}: no usable draws among the corpus models")
    with np.load(next(p for p, m in keep.items() if m.any())) as d:
        n_levels = d["fcd_cdf"].shape[1]                  # 100 with fcd_store_stride 100
        has_catch24 = "catch24" in d.files
    shapes = {"model": (n,), "draw": (n,), "theta": (n, N_REGIONS, P_MAX), "psi": (n, K_MAX),
              "regional": (n, N_REGIONS, 6), "pairwise": (n, N_REGIONS * (N_REGIONS - 1) // 2),
              "fcd": (n, n_levels), "fc_moments": (n, 3)}
    if has_catch24:
        shapes["catch24"] = (n, N_REGIONS, 24)
    dtypes = {"model": np.int32, "draw": np.int32}
    arrays = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+",
                                           dtype=dtypes.get(k, np.float32), shape=s)
              for k, s in shapes.items()}
    model_row = {index: row for row, index in enumerate(members)}
    rows_per_model = np.zeros(len(members), dtype=np.int64)
    start = 0
    for path in chunks:
        mask = keep.get(path)
        if mask is None or not mask.any():
            continue
        with np.load(path) as d:
            block = {"model": np.array([model_row[i] for i in d["model"][mask]], np.int32),
                     "draw": d["draw"][mask], "theta": d["theta"][mask], "psi": d["psi"][mask],
                     "regional": transform_regional(d["regional"][mask]),
                     "pairwise": d["pairwise"][mask], "fcd": d["fcd_cdf"][mask] / full,
                     "fc_moments": transform_fc_moments(d["fc_moments"][mask])}
            if has_catch24:
                block["catch24"] = transform_catch24(d["catch24"][mask])
        stop = start + len(block["model"])
        for k, v in block.items():
            arrays[k][start:stop] = v
        np.add.at(rows_per_model, block["model"], 1)
        start = stop
    for a in arrays.values():
        a.flush()
    if np.any(np.diff(arrays["model"]) < 0):
        raise ValueError("rows are not ordered by model: chunks were read out of order")
    return {"rows": n, "rows_per_model": rows_per_model,
            "models_without_rows": int((rows_per_model == 0).sum())}


def _write_models(chunks: list[Path], members: list[int], rows_per_model: np.ndarray,
                  out: Path) -> None:
    """models.npz and the concatenated graph arrays, in the order of `members`."""
    cards = {}
    for path in chunks:
        for line in path.with_suffix(".cards.jsonl").read_text().splitlines():
            c = json.loads(line)
            cards[c["index"]] = c
    graphs = [graph_arrays(compile_card(card_from_dict(cards[i]["card"])).dag) for i in members]
    sizes = np.array([len(g["features"]) for g in graphs])
    np.save(out / "node_features.npy", np.concatenate([g["features"] for g in graphs]))
    np.save(out / "node_degree.npy", np.concatenate([g["degree"] for g in graphs]))
    np.save(out / "distances.npy", np.concatenate([g["distances"].ravel() for g in graphs]))
    np.savez(out / "models.npz",
             index=np.array(members, np.int64),
             sc_group=np.array([cards[i]["sc_group"] for i in members], np.int32),
             n_rows=rows_per_model,
             node_offset=np.concatenate([[0], np.cumsum(sizes)]),
             distance_offset=np.concatenate([[0], np.cumsum(sizes ** 2)]),
             **{k: np.stack([g[k] for g in graphs])
                for k in ("regional_par", "global_par", "deriv_roots", "agg", "out_root")})


# ---------------------------------------------------------------- network FC


def pack_network_fc(split_dir: Path, block_rows: int = 20_000) -> int:
    """Write <split_dir>/fc_net.npy: per row, the stable arctanh of the FC between the Yeo-7
    networks' averaged signals (data/networks.py), from the stored region FC (pairwise,
    arctanh) and regional SDs (regional column 1, log10 SD). Returns the number of rows."""
    from delssome_fm.data.networks import NETWORK_PAIRS, network_fc, stable_arctanh, \
        yeo7_assignment
    networks = yeo7_assignment()
    pairwise = np.load(split_dir / "pairwise.npy", mmap_mode="r")
    regional = np.load(split_dir / "regional.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(split_dir / "fc_net.npy", mode="w+", dtype=np.float32,
                                    shape=(len(pairwise), len(NETWORK_PAIRS)))
    for s in range(0, len(pairwise), block_rows):
        fc = np.tanh(np.asarray(pairwise[s:s + block_rows], np.float64))
        sd = 10.0 ** np.asarray(regional[s:s + block_rows, :, 1], np.float64)
        out[s:s + block_rows] = stable_arctanh(network_fc(fc, sd, networks))
    out.flush()
    if not np.isfinite(out).all():
        raise ValueError(f"{split_dir}: non-finite network FC")
    return len(out)


# ---------------------------------------------------------------- z-scores


def normalisation(store_dir: Path, block_rows: int = 20_000) -> dict[str, np.ndarray]:
    """Mean and SD of each z-scored target over the train split, written to
    <store_dir>/normalisation.npz: per column for regional (6), fc_moments (3), catch24
    (24) and fc_net (21) when packed, one value pooled over all edges for pairwise. A column that is
    constant over the train split (SD 0, e.g. a catch24 proportion of very short runs) gets
    SD 1: it is centred, not scaled. The FCD CDF is not z-scored: it is a probability, and
    the monotone head predicts it directly (architecture.md §5.2)."""
    train = store_dir / "train"
    pool = {"regional": (0, 1), "pairwise": (0, 1), "fc_moments": (0,)}
    if (train / "catch24.npy").is_file():
        pool["catch24"] = (0, 1)
    if (train / "fc_net.npy").is_file():
        pool["fc_net"] = (0,)
    stats = {}
    for name, axes in pool.items():
        a = np.load(train / f"{name}.npy", mmap_mode="r")
        total, total_sq, count = 0.0, 0.0, 0
        for s in range(0, len(a), block_rows):
            x = np.asarray(a[s:s + block_rows], dtype=np.float64)
            total = total + x.sum(axis=axes)
            total_sq = total_sq + (x ** 2).sum(axis=axes)
            count += int(np.prod([x.shape[i] for i in axes]))
        mean = total / count
        stats[f"{name}_mean"] = np.asarray(mean, np.float32)
        sd = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0.0))
        stats[f"{name}_sd"] = np.asarray(np.where(sd > 0, sd, 1.0), np.float32)
    # write then rename, so a job reading the file meanwhile never sees a partial one
    tmp = store_dir / "normalisation.tmp.npz"
    np.savez(tmp, **stats)
    tmp.replace(store_dir / "normalisation.npz")
    return stats


# ---------------------------------------------------------------- generalisation diagnostic


def pack_generalisation(cfg: GeneralisationConfig, sim_cfg: SimConfig,
                        scs: dict[str, np.ndarray], training_store: Path,
                        parameter_policy: str) -> dict[str, Any]:
    """Pack the four cells of sim/generalisation.py into a store at <output_dir>/store, one
    split per cell, z-scored with the training store's statistics. scs: split -> (G, N, N)
    group SCs; parameter_policy: that of the corpus config the cells were simulated with.
    Raises if a cell's chunks do not hold every sampled model."""
    sample = read_sample(cfg)
    store = cfg.output_dir / "store"
    summaries = {}
    for cell, (model_split, sc_split) in CELLS.items():
        chunks = sorted((cfg.output_dir / "chunks" / cell).glob("*.npz"))
        found = sorted(json.loads(line)["index"] for c in chunks
                       for line in c.with_suffix(".cards.jsonl").read_text().splitlines())
        if found != sample[model_split]:
            missing = sorted(set(sample[model_split]) - set(found))
            raise ValueError(f"{cell}: {len(missing)} sampled models have no chunk, e.g. "
                             f"{missing[:5]}; rerun their simulation jobs")
        summaries[cell] = pack_chunks(chunks, sample[model_split], n_window_pairs(sim_cfg),
                                      store / cell, scs[sc_split],
                                      {"cell": cell, "parameter_policy": parameter_policy})
    shutil.copy(training_store / "normalisation.npz", store / "normalisation.npz")
    return summaries
