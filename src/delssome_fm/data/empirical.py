"""Empirical HCP-YA SC, FC and FCD CDF: loading from disk and averaging up to group level.

A group's arrays can be obtained in two ways. `load_reference_group` reads the saved
reference files. `load_subjects` followed by `average_group` recomputes them from
subject-level data, reproducing Tianchu Zeng's MATLAB pipeline (docs/data.md §4.2):

    run -> subject:   Fisher mean of FC, mean of FCD CDF, over the subject's runs
    subject -> group: Fisher mean of FC, mean of FCD CDF, `group_sc` for SC

FCD CDFs stay in stored units: cumulative counts over 10,000 equal bins on [-1, 1].
`normalise_cdf` converts them to a CDF on [0, 1]. Reducing them to fewer levels is not done
here. Group membership comes from groups.py; comparison against the reference is in verify.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from delssome_fm.config import DataConfig
from delssome_fm.data.groups import Group

N_REGIONS = 68
FCD_BINS = 10_000
# File suffixes of the four HCP-YA resting-state runs: REST1_LR, REST1_RL, REST2_LR, REST2_RL.
RUN_LABELS: tuple[str, ...] = ("bld001", "bld002", "bld003", "bld004")


@dataclass(frozen=True)
class GroupData:
    """
    sc:      (N, N)  log mean nonzero streamline count, zero diagonal
    fc:      (N, N)  Fisher-averaged Pearson FC, diagonal 1 - eps
    fcd_cdf: (B,)    FCD CDF as cumulative counts, B = 10,000 bins on [-1, 1]
    """

    sc: np.ndarray
    fc: np.ndarray
    fcd_cdf: np.ndarray

    def __post_init__(self) -> None:
        _check_shape("group SC", self.sc, (N_REGIONS, N_REGIONS))
        _check_shape("group FC", self.fc, (N_REGIONS, N_REGIONS))
        _check_shape("group FCD CDF", self.fcd_cdf, (FCD_BINS,))


@dataclass(frozen=True)
class SubjectData:
    """
    subject_ids: (S,)
    sc:          (S, N, N)  raw streamline counts; all-NaN for the 27 subjects without SC
    fc:          (S, N, N)  Fisher mean over the subject's runs
    fcd_cdf:     (S, B)     mean of the run FCD CDFs (counts)
    n_runs:      (S,)       runs found for each subject, 1 to 4
    """

    subject_ids: tuple[int, ...]
    sc: np.ndarray
    fc: np.ndarray
    fcd_cdf: np.ndarray
    n_runs: np.ndarray

    def position(self, subject_id: int) -> int:
        try:
            return self.subject_ids.index(subject_id)
        except ValueError:
            raise KeyError(f"subject {subject_id} was not loaded") from None


# ---------------------------------------------------------------- averaging


def stable_atanh(r: np.ndarray) -> np.ndarray:
    """arctanh with r clipped to [-1, 1] and ±inf replaced by atanh(±(1 - eps)).

    Matches CBIG_StableAtanh, so a correlation of exactly 1 survives a Fisher round trip as
    1 - eps rather than becoming inf. NaN passes through.
    """
    eps = np.finfo(np.float64).eps
    with np.errstate(divide="ignore"):
        z = np.arctanh(np.clip(r, -1.0, 1.0))
    z[np.isposinf(z)] = np.arctanh(1.0 - eps)
    z[np.isneginf(z)] = np.arctanh(-1.0 + eps)
    return z


def nan_mean(x: np.ndarray) -> np.ndarray:
    """Mean over axis 0 ignoring NaN (CBIG_nanmean). Raises where every value is NaN."""
    missing = np.isnan(x)
    count = x.shape[0] - missing.sum(axis=0)
    if np.any(count == 0):
        raise ValueError("nan_mean: some entries have no non-NaN values to average")
    return np.where(missing, 0.0, x).sum(axis=0) / count


def fisher_mean(r: np.ndarray) -> np.ndarray:
    """Fisher-z average of correlation matrices over axis 0: tanh(nan_mean(stable_atanh(r)))."""
    return np.tanh(nan_mean(stable_atanh(r)))


def group_sc(sc: np.ndarray) -> np.ndarray:
    """
    sc: (S, N, N) subject SC; subjects without SC are entirely NaN and are dropped
    returns: (N, N) group SC

    For i != j, if at least half of the subjects with SC have SC_ij != 0, the entry is
    log(mean of the nonzero SC_ij); otherwise it is 0. The diagonal is 0. (group_sc.m)
    """
    missing = np.isnan(sc).reshape(sc.shape[0], -1)
    partial = missing.any(axis=1) & ~missing.all(axis=1)
    if partial.any():
        raise ValueError(f"group_sc: subjects at positions {np.flatnonzero(partial).tolist()} "
                         "have partly-NaN SC; expected each subject fully present or fully NaN")
    valid = sc[~missing.all(axis=1)]
    if valid.shape[0] == 0:
        raise ValueError("group_sc: no subject in the group has SC")
    nonzero = (valid != 0).sum(axis=0)
    total = valid.sum(axis=0)  # zeros add nothing, so this is the sum of the nonzero values
    keep = nonzero >= 0.5 * valid.shape[0]
    out = np.zeros(valid.shape[1:])
    out[keep] = np.log(total[keep] / nonzero[keep])
    np.fill_diagonal(out, 0.0)
    return out


def normalise_cdf(counts: np.ndarray) -> np.ndarray:
    """(..., B) cumulative counts -> (..., B) CDF on [0, 1], by dividing by the last bin."""
    if np.any(np.diff(counts, axis=-1) < 0):
        raise ValueError("normalise_cdf: counts are not non-decreasing")
    total = counts[..., -1:]
    if np.any(total <= 0):
        raise ValueError("normalise_cdf: final cumulative count is not positive")
    return counts / total


def average_group(subjects: SubjectData, subject_ids: Sequence[int]) -> GroupData:
    """Group-level arrays from already-loaded subject-level data."""
    idx = [subjects.position(s) for s in subject_ids]
    return GroupData(sc=group_sc(subjects.sc[idx]),
                     fc=fisher_mean(subjects.fc[idx]),
                     fcd_cdf=nan_mean(subjects.fcd_cdf[idx]))


# ---------------------------------------------------------------- loading


def read_csv_array(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """Read a headerless numeric CSV exactly (round-trip parsing) and check its shape.

    Empty fields become NaN, which is how subjects without SC are stored.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"expected data file not found: {path}")
    arr = pd.read_csv(path, header=None, float_precision="round_trip").to_numpy(np.float64)
    if len(shape) == 1 and arr.shape == (1, shape[0]):
        arr = arr[0]
    _check_shape(str(path), arr, shape)
    return arr


def load_reference_group(cfg: DataConfig, group: Group) -> GroupData:
    """The saved group-level arrays for `group` (files {dir}/{group.name}.csv)."""
    name = f"{group.name}.csv"
    return GroupData(
        sc=read_csv_array(cfg.reference_sc_dir / name, (N_REGIONS, N_REGIONS)),
        fc=read_csv_array(cfg.reference_fc_dir / name, (N_REGIONS, N_REGIONS)),
        fcd_cdf=read_csv_array(cfg.reference_fcd_dir / name, (FCD_BINS,)),
    )


def load_subject(cfg: DataConfig, subject_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    returns: sc (N, N), fc (N, N), fcd_cdf (B,), n_runs
    SC is all-NaN if the subject has none. FC and FCD CDF are averaged over the runs present.
    """
    sc = read_csv_array(cfg.subject_sc_dir / f"{subject_id}.csv", (N_REGIONS, N_REGIONS))
    if np.isnan(sc).any() and not np.isnan(sc).all():
        raise ValueError(f"subject {subject_id}: SC is partly NaN")
    fc_runs, fcd_runs = [], []
    for label in RUN_LABELS:
        fc_path = cfg.run_fc_dir / f"{subject_id}_{label}.csv"
        fcd_path = cfg.run_fcd_dir / f"{subject_id}_{label}.csv"
        if fc_path.is_file() != fcd_path.is_file():
            raise FileNotFoundError(f"subject {subject_id} run {label}: FC and FCD files must "
                                    f"both exist or both be absent ({fc_path}, {fcd_path})")
        if fc_path.is_file():
            fc_runs.append(read_csv_array(fc_path, (N_REGIONS, N_REGIONS)))
            fcd_runs.append(read_csv_array(fcd_path, (FCD_BINS,)))
    if not fc_runs:
        raise FileNotFoundError(f"subject {subject_id}: no runs found in {cfg.run_fc_dir}")
    fc_stack, fcd_stack = np.stack(fc_runs), np.stack(fcd_runs)
    if np.isnan(fc_stack).any() or np.isnan(fcd_stack).any():
        raise ValueError(f"subject {subject_id}: NaN in run-level FC or FCD CDF")
    return sc, fisher_mean(fc_stack), nan_mean(fcd_stack), len(fc_runs)


def load_subjects(cfg: DataConfig, subject_ids: Sequence[int], n_workers: int = 16) -> SubjectData:
    """Load many subjects. Threads only overlap NAS reads; results keep `subject_ids` order."""
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        loaded = list(pool.map(lambda s: load_subject(cfg, s), subject_ids))
    return SubjectData(subject_ids=tuple(subject_ids),
                       sc=np.stack([x[0] for x in loaded]),
                       fc=np.stack([x[1] for x in loaded]),
                       fcd_cdf=np.stack([x[2] for x in loaded]),
                       n_runs=np.array([x[3] for x in loaded]))


def _check_shape(what: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise ValueError(f"{what}: expected shape {shape}, found {arr.shape}")
