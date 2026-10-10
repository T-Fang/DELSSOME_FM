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
documented in [docs/data.md](docs/data.md), together with the comparison of our simulator
against the original DELSSOME (§9).

## Status

| Build step | State |
|---|---|
| 1. Phase 0: data and groups (gate) | **done**, verification passes |
| 2. Repo skeleton, config loading, cluster wrapper | **done** |
| 3. Model card and compiler | **done** |
| 4. Reference cards (gate) | **done**: seven cards (adds MPR and Jansen–Rit) |
| 5. Simulator | **done**: integrate, observe, summary; summary reproduces the empirical pipeline |
| 6. Reference reproduction (gate) | **deferred, not passed** (2026-10-05, project lead): code and criterion ready, gate not run. Direct comparisons with the original simulations are done (docs/data.md §9) |
| 7. Sampler and corpus | **in progress**: sampler done; corpus generation started 2026-10-10 (train 100,000, val and test 10,000 valid models each) |
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

**Build step 6: reproduce the published costs (gate).** The original fitted parameters are
torch pickles, so export them once with an environment that has torch:

```bash
/home/ftian/storage/miniconda/envs/lifespan_ei/bin/python scripts/export_original_params.py \
    --params-root /mnt/nas/CSC21/Yeolab/Users/tzeng/Python/DELSSOME_plus/params \
    --out-dir outputs/reproduce/original --n-sets 50
```

Then run one single-CPU cluster job per (model, parameter set), 150 in all, ~4 min each,
and apply the pass criterion in `configs/reproduce.yaml`:

```bash
python scripts/submit_reproduction.py --submit                # dry run without --submit
python scripts/submit_reproduction.py --submit --only-missing # resubmit failed sets
python scripts/check_reproduction.py                          # exits 1 if the gate fails
```

**Status: deferred.** On 2026-10-05 the project lead deferred this gate and asked to proceed
as if it had passed. It has not been run. Spot checks on CPU agree with the original code
(MFM seed 30: ours 0.426 +- 0.015, original 0.425 +- 0.025 over seven seeds), but that is
not the gate.

**Comparison with the original simulations** (docs/data.md §9). This runs one original-style
evaluation of ours per parameter set and replays Tianchu Zeng's own test runs with their
stored torch seeds. It then plots the cost between the two implementations, and their FC and
FCD matrices, with the original colour tables. The plotting and replay scripts need torch or
matplotlib, so they run in the `lifespan_ei` env:

```bash
python scripts/submit_reproduction.py --config configs/reproduce_once.yaml --submit
python scripts/submit_original_test.py --out-dir outputs/reproduce/original_sim --submit
/home/ftian/storage/miniconda/envs/lifespan_ei/bin/python scripts/plot_reproduction.py
/home/ftian/storage/miniconda/envs/lifespan_ei/bin/python scripts/plot_comparison.py
```

**Build step 7: the synthetic corpus** (docs/generation.md §9). Each job screens one chunk of
candidate indices of one split and simulates the valid models, on one CPU. A chunk of 500
takes about 3.6 h. Dry run without `--submit`:

```bash
python scripts/submit_corpus.py --split train --start 0 --stop 660000 --chunk 500 \
    --walltime 12:00:00 --mem 6G --submit
python scripts/corpus_status.py --target train=100000 --target val=10000 --target test=10000
```

`corpus_status.py` prints, per split, how many candidates are screened and valid, and how
many more are needed for the target. To grow a split, submit a further index range. To redo
failed chunks, rerun the same range with `--only-missing`, but only once none of its jobs are
still queued, because otherwise those would be submitted twice. To add parameter draws to
existing models, use `--extend --first-draw 100 --n-param-sets 100`. One chunk locally:

```bash
python scripts/build_corpus.py --split val --start 0 --stop 5
```

**Submit a job to the cluster.** By default it prints a dry run. Add `--submit` to actually
submit. CBIG_pbsubmit always runs on `headnode`. From any other machine, the call is forwarded
with `headnode_ssh` in `configs/cluster.yaml`. The `submit_*.py` scripts send a whole batch of
jobs through a single ssh session (`cluster.submit.submit_batch`), because each ssh login costs
about 6 s and CBIG_pbsubmit pauses 3 s per job.

```bash
python scripts/submit_job.py --name verify --walltime 01:00:00 --mem 16G \
    -- python scripts/verify_groups.py
python scripts/submit_job.py --name sim --walltime 02:00:00 --mem 32G --ngpus 1 --gpu-type L40S \
    -- python <some script>
```

## Layout

```
configs/            plain YAML, one file per dataclass in src/delssome_fm/config.py
                    (data, sim, cluster, corpus, reproduce; reproduce_once is a second
                    ReproduceConfig)
assets/colormaps/   the original DELSSOME colour tables for FC and FCD figures
docs/               design documents, the codegen brief, data provenance
scripts/            thin CLI entry points, no logic
src/delssome_fm/
    config.py       config dataclasses and the strict YAML loader
    data/           group membership, empirical SC/FC/FCD loading and averaging, Phase 0 check
    spec/           model card -> SymPy -> equation DAG + simulator rhs, the card sampler;
                    see spec/README.md
    sim/            Euler-Maruyama in JAX, Balloon-Windkessel, FC/FCD/regional summaries,
                    the original cost, build-step-6 reproduction, the corpus builder
    cluster/        CBIG_pbsubmit wrapper
tests/
```

`nn/` and `train/` are added in build step 8.

## Where outputs land

`outputs/` is untracked. Job scripts and PBS logs go to `outputs/jobs/`, set by `job_dir` in
`configs/cluster.yaml`.

| Path | Contents |
|---|---|
| `outputs/corpus/<split>/` | corpus chunks: `.npz` statistics, `.cards.jsonl`, `.log.jsonl`, `.manifest.json` (docs/generation.md §9) |
| `outputs/reproduce/original/` | the original fitted parameters and recorded costs, exported from torch |
| `outputs/reproduce/once/` | one original-style evaluation of ours per set (two with independent noise) |
| `outputs/reproduce/original_sim/` | replays of the original test runs, with their FC and FCD |
| `outputs/reproduce/comparison/` | cost box plots and FC/FCD matrix figures |
| `outputs/superseded/` | pilot runs from earlier sampler versions, kept for reference |

The outputs live on the shared CSC21 NAS, which is close to full. On 2026-10-10 it filled up
for about 90 minutes, and 75 corpus jobs failed with `Disk quota exceeded`. A failed job
writes nothing, so rerunning its range is enough.
