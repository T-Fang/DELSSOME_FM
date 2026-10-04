"""Phase 0 verification: recompute every reference group from subject-level data and compare.

The tolerances are those of the codegen brief §5.3. They are module constants rather than
arguments so they cannot be loosened per call; if a comparison fails, report it, do not
change them.

    SC, FC:   relative error per edge < 1e-6. Relative error is |x - y| / |y|; where the
              reference is exactly 0 (SC zeros, SC diagonal) x must be exactly 0 as well,
              otherwise the error is infinite.
    FCD CDF:  max absolute difference < 1e-6, in stored units (cumulative counts). That is
              stricter than comparing normalised CDFs, which are counts / 624,403.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from delssome_fm.config import DataConfig
from delssome_fm.data.empirical import GroupData, average_group, load_reference_group, load_subjects
from delssome_fm.data.groups import groups_from_config

SC_REL_TOL = 1e-6
FC_REL_TOL = 1e-6
FCD_ABS_TOL = 1e-6


@dataclass(frozen=True)
class Discrepancy:
    """The worst entry of one quantity in one group."""

    quantity: str          # "SC", "FC" or "FCD CDF"
    group: str             # e.g. "train0"
    entry: tuple[int, ...]
    computed: float
    reference: float
    error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.error < self.tolerance

    def describe(self) -> str:
        status = "ok  " if self.passed else "FAIL"
        return (f"{status} {self.quantity:8s} worst {self.error:.3e} (tol {self.tolerance:.0e}) "
                f"in {self.group} at {self.entry}: computed {self.computed!r}, "
                f"reference {self.reference!r}")


def relative_error(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|x - ref| / |ref|, with 0 where both are 0 and inf where only ref is 0."""
    diff = np.abs(x - ref)
    scale = np.abs(ref)
    out = np.full(diff.shape, np.inf)
    nonzero = scale > 0
    out[nonzero] = diff[nonzero] / scale[nonzero]
    out[~nonzero & (diff == 0)] = 0.0
    return out


def compare_group(name: str, computed: GroupData, reference: GroupData) -> list[Discrepancy]:
    """Worst SC, FC and FCD CDF entries of one recomputed group against its reference."""
    return [
        _worst("SC", name, relative_error(computed.sc, reference.sc), computed.sc,
               reference.sc, SC_REL_TOL),
        _worst("FC", name, relative_error(computed.fc, reference.fc), computed.fc,
               reference.fc, FC_REL_TOL),
        _worst("FCD CDF", name, np.abs(computed.fcd_cdf - reference.fcd_cdf),
               computed.fcd_cdf, reference.fcd_cdf, FCD_ABS_TOL),
    ]


def verify_all_groups(cfg: DataConfig, n_workers: int = 16) -> list[Discrepancy]:
    """Compare all configured groups. Loads each subject once (about 8,500 CSVs)."""
    groups = groups_from_config(cfg)
    needed = sorted({s for g in groups for s in g.subject_ids})
    subjects = load_subjects(cfg, needed, n_workers=n_workers)
    results = []
    for group in groups:
        computed = average_group(subjects, group.subject_ids)
        results.extend(compare_group(group.name, computed, load_reference_group(cfg, group)))
    return results


def worst_per_quantity(results: list[Discrepancy]) -> dict[str, Discrepancy]:
    """The single worst discrepancy for each quantity across all groups."""
    worst: dict[str, Discrepancy] = {}
    for d in results:
        if d.quantity not in worst or d.error > worst[d.quantity].error:
            worst[d.quantity] = d
    return worst


def _worst(quantity: str, group: str, err: np.ndarray, x: np.ndarray, ref: np.ndarray,
           tol: float) -> Discrepancy:
    k = np.unravel_index(int(np.argmax(err)), err.shape)
    return Discrepancy(quantity=quantity, group=group, entry=tuple(int(i) for i in k),
                       computed=float(x[k]), reference=float(ref[k]), error=float(err[k]),
                       tolerance=tol)
