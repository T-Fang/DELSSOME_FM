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
    flat_rel_sd: float            # flat if SD <= flat_rel_sd * |mean| in any region (corpus.py)
    initial_state_scale: float    # x0 ~ U(-s, s), drawn per simulation
    fcd_store_stride: int         # store the FCD CDF at every k-th bin (summary.fcd_levels)
    parameter_policy: str         # spec/sampler.py: "multiplier" (corpus, d10) or "sinh" (pilot)
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
class SplitSizes:
    """A count per split."""

    train: int
    val: int
    test: int


@dataclass(frozen=True)
class TrainingDataConfig:
    """configs/training_data.yaml: the stage-1 training store packed from the corpus
    (train/pack.py, architecture.md §5.1)."""

    store_dir: Path
    n_models: SplitSizes        # each split's corpus: its first n valid models (corpus.first_valid)


@dataclass(frozen=True)
class Stage1Config:
    """configs/stage1.yaml: the stage-1 network and its training (architecture.md §2-§5)."""

    store_dir: Path               # written by scripts/pack_training_data.py
    output_dir: Path              # checkpoints, one subdirectory per run
    seed: int
    # network (architecture.md §1: d = d_t = 128, k = 32, 4 layers, 8 heads)
    d_model: int                  # transformer width d
    d_template: int               # template encoder width d_t
    template_layers: int          # Graphormer layers over the equation DAG
    backbone_layers: int          # transformer layers over [CLS, regions, template]
    n_heads: int
    ffn_multiplier: int           # feed-forward width = ffn_multiplier * width
    dropout: float
    fc_rank: int                  # k of the bilinear FC head
    fc_offset: bool               # add a per-draw offset b(h_CLS) to the FC head (nn/heads.py)
    pair_head: str                # FC head: bilinear | mlp_symmetric | mlp_concat | latent
    pair_hidden: int              # hidden width of the MLP pair heads
    fc_autoencoder: str           # "latent" head: FC autoencoder checkpoint; else "none"
    stat_loss: bool               # train on the summary statistics (regional, FC, FCD, moments)
    catch24_loss: bool            # train a catch24 head on the regions' catch24 features
    fc_net_loss: bool             # train a head on the Yeo-7 network FC (data/networks.py)
    cost_loss: bool               # add 1 - r, d and KS (raw units, all edges) to the loss
    max_distance: int             # DAG shortest-path lengths are clipped to this (attention bias)
    max_degree: int               # node in/out degrees are clipped to this (degree embedding)
    # training data (the scaling sweep of architecture.md §7.2 varies these)
    train_models: int             # number of train corpus models to use
    draws_per_model: int          # and at most this many usable draws of each (100: all)
    oversample_coupled: bool      # false: the first train_models models, natural composition
    coupled_share: float          # if oversample_coupled: share of coupled models
    coupled_threshold: float      # coupled: the model's mean FC over its draws exceeds this
    coupled_draws_only: bool      # train and validate on coupled draws only (draw mean FC >
                                  # coupled_threshold; the coupled-only test, 2026-10-26)
    # optimisation
    batch_size: int
    edges_per_step: int           # FC edges in the pairwise loss per step (architecture.md §5.2)
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    val_rows: int                 # fixed random subset of the val split used for validation
    val_every_steps: int
    final_eval_rows: int          # rows per split in the evaluation after training (train/fit.py)
    num_workers: int              # DataLoader worker processes
    precision: str                # Lightning precision, e.g. bf16-mixed
    gradient_clip: float          # max gradient norm
    # tracking
    log_every_steps: int           # training metrics are logged every this many steps
    mlflow_tracking_uri: str
    mlflow_artifact_dir: Path     # where run artifacts go (else MLflow uses ./mlruns)
    mlflow_experiment: str


@dataclass(frozen=True)
class FCAutoencoderConfig:
    """configs/fc_autoencoder.yaml: the FC autoencoder (nn/fc_autoencoder.py,
    train/autoencoder.py), 2026-10-25."""

    store_dir: Path               # the stage-1 training store whose pairwise target it encodes
    output_dir: Path              # checkpoints, one subdirectory per run
    seed: int
    latent_dim: int
    hidden: tuple[int, ...]       # encoder widths between input and latent (decoder: reversed)
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    val_rows: int                 # fixed random subset of the val split used for validation
    val_every_steps: int
    num_workers: int
    precision: str
    gradient_clip: float          # max gradient norm (without it, training diverged: 2026-10-25)
    log_every_steps: int
    mlflow_tracking_uri: str
    mlflow_artifact_dir: Path
    mlflow_experiment: str


@dataclass(frozen=True)
class GeneralisationConfig:
    """configs/generalisation.yaml: the 2 x 2 diagnostic of sim/generalisation.py (seen or
    unseen model x seen or unseen SC), decided 2026-10-16."""

    output_dir: Path
    seed: int
    n_models: int                 # models sampled from each of the train and val corpora
    first_draw: int               # new draws only: stage 1 trained on draws 0..99
    n_draws: int                  # draws per model and cell


@dataclass(frozen=True)
class ReferenceEvalConfig:
    """configs/reference_eval.yaml: stage-1 predictions vs simulations of the reference models
    at their published test fits (sim/reference_eval.py), decided 2026-10-19."""

    output_dir: Path
    seed: int
    n_repeats: int                # simulations per parameter set, independent noise
    models: tuple[str, ...]       # reference cards with fitted sets in configs/reproduce.yaml
    training_store: Path          # its normalisation.npz z-scores the targets
    corpus_config: Path           # the corpus whose simulation protocol and parameter policy
                                  # (multiplier: median rescale; sinh: actual values) apply


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
    return config_from_dict(cls, raw, str(path))


def override(cfg: T, assignments: list[str]) -> T:
    """A copy of the config dataclass `cfg` with "field=value" assignments applied; values are
    parsed as YAML and checked as strictly as by load_config (e.g. "dropout=0.1")."""
    raw = _plain(dataclasses.asdict(cfg))
    for a in assignments:
        key, sep, value = a.partition("=")
        if not sep or key not in raw:
            raise ValueError(f"override {a!r}: expected field=value with a field of "
                             f"{type(cfg).__name__}, found {key!r}")
        raw[key] = yaml.safe_load(value)
    return config_from_dict(type(cfg), raw, f"overrides {assignments}")


def to_yaml(cfg: Any) -> str:
    """The config as YAML that load_config reads back to an equal dataclass."""
    return yaml.safe_dump(_plain(dataclasses.asdict(cfg)), sort_keys=False)


def _plain(value: Any) -> Any:
    """asdict output -> plain YAML values: Path -> str, tuple -> list."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value) if isinstance(value, Path) else value


def config_from_dict(cls: type[T], raw: Any, where: str) -> T:
    """Build the dataclass `cls` from plain values (as YAML gives them), with the same strict
    checks as load_config; `where` names the source in error messages."""
    return _build(cls, raw, where)


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
