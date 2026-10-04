# DELSSOME-FM

A neural surrogate that predicts the cost of fitting a biophysical brain circuit model to
fMRI, for many different circuit models rather than one. It extends DELSSOME (a surrogate for a
single model) to be model-agnostic. It has three parts:

1. **Generator.** Samples synthetic circuit models from a declarative template (the *model
   card*). It compiles each one into both an equation graph and a simulator, from the same
   SymPy expression tree, and produces a training corpus.
2. **Encoder.** Reads the equation graph alongside the regional parameters and the structural
   connectivity.
3. **Two-stage training.** Pretraining on simulated summary statistics, then fine-tuning to
   predict cost against empirical data.

Everything is group-level (HCP-YA, Desikan-Killiany 68 regions), following the original
DELSSOME paper.

The design is in [docs/architecture.md](docs/architecture.md) (network) and
[docs/generation.md](docs/generation.md) (synthetic models). The build plan and coding
standards are in [docs/CODEGEN_BRIEF.md](docs/CODEGEN_BRIEF.md). The empirical data is
documented in [docs/data.md](docs/data.md).

## Status

| Build step | State |
|---|---|
| 1. Phase 0: data and groups (gate) | **done**, verification passes |
| 2. Repo skeleton, config loading, cluster wrapper | **done** |
| 3. Model card and compiler | **done** |
| 4. Reference cards (gate) | five cards compile; **awaiting review** |
| 5. Simulator | not started |
| 6. Reference reproduction (gate) | not started |
| 7. Sampler and corpus | not started |
| 8. Network and training | not started |

## Install

```bash
source CBIG_init_conda              # on the CBIG cluster
conda create -n delssome_fm python=3.12 pip
conda activate delssome_fm
pip install -e ".[test]"
```

`pip install -e .` installs CPU JAX. For GPU jobs, also run `pip install "jax[cuda12]"`
(the `delssome_fm` env on the cluster already has it). Without a GPU, JAX falls back to the
CPU.

## Run

All commands run from the repository root.

**Tests.** The fast tests run anywhere:

```bash
pytest -m "not cluster"
```

The tests marked `cluster` need the lab NAS and skip without it:

```bash
pytest -m cluster        # about 9 minutes, mostly NAS reads
```

**Phase 0: verify the empirical groups.** This recomputes all 91 HCP-YA group-level SC, FC and
FCD CDFs from subject-level data and compares them with the saved reference. It exits 1 on any
failure.

```bash
python scripts/verify_groups.py --config configs/data.yaml
```

**Submit a job to the cluster.** By default it prints a dry run. Add `--submit` to actually
submit, and do that from `headnode`; GPU jobs refuse anywhere else.

```bash
python scripts/submit_job.py --name verify --walltime 01:00:00 --mem 16G \
    -- python scripts/verify_groups.py
python scripts/submit_job.py --name sim --walltime 02:00:00 --mem 32G --ngpus 1 --gpu-type L40S \
    -- python <some script>
```

## Layout

```
configs/            plain YAML, one file per dataclass in src/delssome_fm/config.py
docs/               design documents, the codegen brief, data provenance
scripts/            thin CLI entry points, no logic
src/delssome_fm/
    config.py       config dataclasses and the strict YAML loader
    data/           group membership, empirical SC/FC/FCD loading and averaging, Phase 0 check
    spec/           model card -> SymPy -> equation DAG + simulator rhs; see spec/README.md
    cluster/        CBIG_pbsubmit wrapper
tests/
```

`sim/`, `nn/` and `train/` are added in build steps 5 to 8.

## Where outputs land

`outputs/` is untracked. Job scripts and PBS logs go to `outputs/jobs/`, set by `job_dir` in
`configs/cluster.yaml`.
