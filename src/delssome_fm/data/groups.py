"""Group membership for the HCP-YA group-level data.

The 1029 subjects, sorted by ascending ID, are cut into three contiguous splits (train, val,
test). Within each split, groups are overlapping sliding windows: `group_size` consecutive
subjects, advancing by `group_stride`, with the last window truncated at the end of the
split. This reproduces Tianchu Zeng's `HCP_group_mats_bootstrap.m` exactly; despite the file
name there is no resampling and no RNG. docs/data.md records how this was established.

This module only says which subjects belong to which group. It does not load or average any
arrays; that is empirical.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from delssome_fm.config import DataConfig, SplitBounds

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class Group:
    """One group. `name` matches the reference file stem, e.g. 'train0' for train0.csv."""

    split: str
    index: int
    subject_ids: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"{self.split}{self.index}"


def load_subject_order(path: Path, n_subjects: int) -> tuple[int, ...]:
    """Read the subject list (one ID per line) and check it is the expected ascending list."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"subject list not found: {path}")
    ids = tuple(int(tok) for tok in path.read_text().split())
    if len(ids) != n_subjects:
        raise ValueError(f"{path}: expected {n_subjects} subject IDs, found {len(ids)}")
    for pos, (a, b) in enumerate(zip(ids, ids[1:])):
        if not a < b:
            raise ValueError(
                f"{path}: IDs must be strictly ascending (groups are defined by this order), "
                f"but position {pos} has {a} followed by {b}")
    return ids


def window_bounds(n: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Half-open [start, stop) windows over n positions.

    Matches MATLAB's `for sub = 1:stride:(n - size + stride)` with the stop clipped to n, so
    the final window can be shorter than `size` (test12 has 49 subjects).
    """
    if size < 1 or stride < 1:
        raise ValueError(f"group size and stride must be positive, found {size} and {stride}")
    if n < size:
        raise ValueError(f"a split of {n} subjects cannot hold a group of {size}")
    return [(s, min(s + size, n)) for s in range(0, n - size + stride, stride)]


def define_groups(subject_ids: Sequence[int], splits: SplitBounds, size: int,
                  stride: int) -> tuple[Group, ...]:
    """All groups, in split order (train, val, test) and by index within each split."""
    bounds = {name: getattr(splits, name) for name in SPLIT_NAMES}
    _check_splits(bounds, len(subject_ids))
    groups = []
    for split in SPLIT_NAMES:
        start, stop = bounds[split]
        members = tuple(subject_ids[start:stop])
        for index, (a, b) in enumerate(window_bounds(len(members), size, stride)):
            groups.append(Group(split=split, index=index, subject_ids=members[a:b]))
    return tuple(groups)


def groups_from_config(cfg: DataConfig) -> tuple[Group, ...]:
    """The 91 HCP-YA groups as configured in configs/data.yaml."""
    ids = load_subject_order(cfg.subject_list, cfg.n_subjects)
    return define_groups(ids, cfg.splits, cfg.group_size, cfg.group_stride)


def _check_splits(bounds: dict[str, tuple[int, int]], n: int) -> None:
    for name, (start, stop) in bounds.items():
        if not 0 <= start < stop <= n:
            raise ValueError(f"split '{name}' = [{start}, {stop}) is not inside [0, {n})")
    ordered = sorted(bounds.items(), key=lambda item: item[1][0])
    for (name_a, (_, stop_a)), (name_b, (start_b, _)) in zip(ordered, ordered[1:]):
        if start_b < stop_a:
            raise ValueError(f"splits '{name_a}' and '{name_b}' overlap")
