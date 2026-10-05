"""Configuration: plain YAML loaded into frozen dataclasses.

Each YAML file maps onto one dataclass defined here. Loading is strict: a missing field, an
unknown field or a value of the wrong type raises, naming the field and what was found. No
`dict[str, Any]` leaves this module. It checks types only; whether a path exists or a range
makes sense is checked by the code that uses the field.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class SplitBounds:
    """Half-open [start, stop) positions of each split in the ascending subject list."""

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


@dataclass(frozen=True)
class DataConfig:
    """configs/data.yaml: where the HCP-YA data lives and how groups are formed (docs/data.md)."""

    subject_list: Path
    n_subjects: int
    splits: SplitBounds
    group_size: int
    group_stride: int
    subject_sc_dir: Path
    run_fc_dir: Path
    run_fcd_dir: Path
    reference_sc_dir: Path
    reference_fc_dir: Path
    reference_fcd_dir: Path


@dataclass(frozen=True)
class SimConfig:
    """configs/sim.yaml: integration and summary settings, shared by every simulation so that
    stage 1 and stage 2 statistics come from the identical pipeline (brief §7.3)."""

    dt: float                  # s, Euler-Maruyama step
    tr: float                  # s, one observed frame; must be a whole number of steps
    n_frames: int              # observed frames kept
    burn_in_frames: int        # frames simulated and discarded before those
    divergence_bound: float    # |state| above this marks a run as diverged
    fcd_window: int            # frames per FCD sliding window (stride 1)
    fcd_bins: int              # equal FCD histogram bins on [-1, 1]


@dataclass(frozen=True)
class CorpusConfig:
    """configs/corpus.yaml: the stage-1 synthetic corpus (generation.md §5-§7, §9)."""

    seed: int                     # cards, SC choice, parameter draws and noise all derive from it
    n_param_sets: int             # parameter draws simulated per kept model
    first_draw: int               # index of the first draw (to append draws to models later)
    batch_size: int               # simulations per jit call (draws of one model)
    screen_draws: int             # generation.md §6: 4 random draws
    screen_frames: int            # ~3 min of simulated time
    screen_burn_in_frames: int
    flat_rel_sd: float            # flat if SD <= flat_rel_sd * max(1, |mean|) in every region
    initial_state_scale: float    # x0 ~ U(-s, s), drawn per simulation
    fcd_store_stride: int         # store the FCD CDF at every k-th bin (summary.fcd_levels)
    output_dir: Path


@dataclass(frozen=True)
class ReproduceModel:
    """One reference model in the build-step-6 gate, with the original's test protocol."""

    name: str                           # reference card name: mfm | fic | hopf
    original_dir: str                   # subdirectory of original_params_root for this model
    dt: float                           # s, the original's test-time Euler step
    burn_in_frames: int                 # warm-up + burn-in of the original, in frames
    initial_state: tuple[float, ...]    # (V,) the original's initial state; FIC's S_E is solved


@dataclass(frozen=True)
class ReproduceConfig:
    """configs/reproduce.yaml: build step 6, reproducing the published costs (docs/data.md §7)."""

    original_params_root: Path          # tzeng DELSSOME_plus/params (read-only)
    original_input_dir: Path            # the original test inputs: SC/FC_test.csv, FCD_CDF_test.mat
    export_dir: Path                    # where scripts/export_original_params.py wrote .npz files
    output_dir: Path
    n_sets: int                         # saved CMA-ES runs per model (seed1..seedN)
    n_noise_repeats: int                # independent cost evaluations per parameter set
    n_dup: int                          # simulations averaged per cost (the original's param_dup)
    seed: int
    batch_size: int                     # simulations per jit call; fixed so shapes are static
    z_threshold: float                  # a set misses if |recorded - our mean| > z * our SD
    max_misses: int                     # allowed misses per component
    bias_alpha: float                   # family-wise level of the paired bias t-tests,
                                        # Bonferroni-split over models x cost components
    models: tuple[ReproduceModel, ...]


@dataclass(frozen=True)
class ClusterConfig:
    """configs/cluster.yaml: how jobs are submitted through CBIG_pbsubmit (cluster/submit.py)."""

    pbsubmit: Path
    conda_init: Path
    conda_env: str
    cuda_version: str
    repo_dir: Path
    job_dir: Path
    headnode_ssh: tuple[str, ...]   # command prefix that runs a shell command on the headnode


def load_config(path: Path, cls: type[T]) -> T:
    """Load the YAML file at `path` into the dataclass `cls`, raising on any mismatch."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    return _build(cls, raw, str(path))


def _build(tp: Any, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(tp):
        return _build_dataclass(tp, value, where)
    if typing.get_origin(tp) is tuple:
        return _build_tuple(tp, value, where)
    if tp is Path:
        if not isinstance(value, str):
            raise TypeError(f"{where}: expected a path string, found {value!r}")
        return Path(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: expected a number, found {value!r}")
        return float(value)
    if tp in (int, str, bool):
        if type(value) is not tp:  # exact: YAML `true` must not pass as an int
            raise TypeError(f"{where}: expected {tp.__name__}, found {value!r}")
        return value
    raise TypeError(f"{where}: field type {tp!r} is not supported by the config loader")


def _build_dataclass(cls: Any, value: Any, where: str) -> Any:
    if not isinstance(value, dict):
        raise TypeError(f"{where}: expected a mapping for {cls.__name__}, found {value!r}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {unknown} for {cls.__name__}")
    kwargs = {}
    for name, field in fields.items():
        if name not in value:
            has_default = (field.default is not dataclasses.MISSING
                           or field.default_factory is not dataclasses.MISSING)
            if not has_default:
                raise KeyError(f"{where}: required field '{name}' is missing")
            continue
        kwargs[name] = _build(hints[name], value[name], f"{where}.{name}")
    return cls(**kwargs)


def _build_tuple(tp: Any, value: Any, where: str) -> tuple:
    if not isinstance(value, list):
        raise TypeError(f"{where}: expected a list, found {value!r}")
    args = typing.get_args(tp)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_build(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
    if len(value) != len(args):
        raise ValueError(f"{where}: expected {len(args)} items, found {len(value)}")
    return tuple(_build(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
