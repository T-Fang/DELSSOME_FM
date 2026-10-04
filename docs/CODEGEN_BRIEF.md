# DELSSOME-FM: code generation brief

Read this in full before writing any code. It is the specification for the repository, the build order, and the standards the code has to meet.

Two design documents accompany this brief and are the source of truth for the science:

- `docs/architecture.md` (from `delssome-fm-architecture-v2.md`)
- `docs/generation.md` (from `delssome-fm-poc-generation.md`)

Where this brief and those documents disagree, **stop and ask**. Do not resolve it yourself.

---

## 1. What we are building

A neural surrogate that predicts the cost of fitting a biophysical brain circuit model to fMRI, for *many different circuit models* rather than one. It has three parts:

1. A **generator** that samples synthetic circuit models from a declarative template, compiles each to both an equation graph and a simulator, and produces a training corpus.
2. An **encoder** that reads the equation graph alongside parameters and structural connectivity.
3. A **two-stage training** pipeline: pretrain on simulated summary statistics, then fine-tune to predict cost against empirical data.

**Group-level only for now**, following the original DELSSOME paper. No individual-subject fitting.

---

## 2. Non-negotiables

These are the invariants. Violating any of them produces bugs with no symptom until the results are silently wrong.

### 2.1 One source, two artifacts

The equation graph the encoder reads and the integrator that runs **must be generated from the same SymPy expression tree, in the same pass**. Never maintain a symbolic description and a numerical implementation side by side.

```
ModelCard ──► sympy expressions ──► canonicalise ──┬──► walk A: DAG nodes + edges
                                                   └──► walk B: JAX drift function
```

Both walks live in the same module and traverse the same tree. Add a round-trip assertion: rebuild the expression from the DAG and check it equals the canonicalised original. Run it on every compile, not just in tests.

### 2.2 Fail loudly

No bare `except`. No `try: ... except: pass`. No silent fallback when a file is missing, a shape mismatches, or a value is out of range. If something is wrong, raise with a message that says what was expected and what was found.

The one exception is simulation divergence, which is expected and is handled by an explicit `diverged` flag on the result, never by catching a NaN.

### 2.3 Verify against reference data before building on it

Phase 0 ends with a passing comparison against existing group-level FC, FCD and SC. Nothing downstream starts until that passes. See §5.

### 2.4 Determinism

Every stochastic step takes an explicit seed. No module reads global RNG state. A corpus must be reproducible from its config plus seed.

---

## 3. Repository layout

```
delssome-fm/
├── README.md                    # what this is, how to run it
├── CLAUDE.md                    # instructions for future Claude Code sessions
├── pyproject.toml
├── docs/
│   ├── architecture.md          # provided
│   ├── generation.md            # provided
│   └── data.md                  # written in Phase 0: what the groups are, how verified
├── configs/
│   ├── data.yaml
│   ├── corpus.yaml
│   └── train.yaml
├── src/delssome_fm/
│   ├── data/
│   │   ├── groups.py            # group membership, bootstrap definitions
│   │   └── empirical.py         # load group SC, FC, FCD CDF
│   ├── spec/
│   │   ├── card.py              # ModelCard dataclasses
│   │   ├── sampler.py           # procedural sampling
│   │   ├── compile.py           # card -> sympy -> DAG + drift  (§2.1 lives here)
│   │   ├── dag.py               # node types, edges, feature encoding
│   │   ├── ops.py               # backend op interface
│   │   └── reference/           # hand-written cards for the five models
│   ├── sim/
│   │   ├── integrate.py         # Euler-Maruyama, nested scan
│   │   ├── observe.py           # Balloon-Windkessel, direct downsample
│   │   ├── summary.py           # FC, FCD CDF, regional statistics
│   │   └── corpus.py            # batched generation, screening
│   ├── nn/
│   │   ├── template_encoder.py  # Graphormer over the DAG
│   │   ├── tokens.py            # region token construction
│   │   ├── backbone.py          # transformer
│   │   └── heads.py             # stage-1 and stage-2 heads
│   ├── train/
│   │   ├── stage1.py
│   │   ├── stage2.py
│   │   └── losses.py
│   └── cluster/
│       └── submit.py            # wrapper around CBIG_pbsubmit
├── scripts/                     # thin CLI entry points, no logic
└── tests/
```

Rules on layout:

- **Two levels of nesting maximum** under `src/delssome_fm/`.
- **Scripts contain no logic.** They parse arguments, call one function from the library, and write output. If a script grows past about 50 lines, the logic belongs in the library.
- **One responsibility per module.** If you cannot describe a module's job in one sentence, split it.

---

## 4. Coding standards

### 4.1 Abstraction, without ceremony

Use a class when it holds state across calls or when there is more than one implementation. Otherwise use a function.

**Do not create:** abstract base classes with one subclass, registries, factories, plugin systems, dependency injection, or a `BaseModel` that everything inherits from. This codebase has a small number of well-understood pieces; indirection makes it harder to read, not easier to extend.

**Do create:** dataclasses for structured data (the model card especially), pure functions for transformations, and a thin protocol for the one place where there really are multiple backends (`ops.py`).

### 4.2 Types and shapes

Type-annotate every public function. For arrays, put the expected shape in the docstring using the symbol names from `docs/architecture.md`:

```python
def build_region_tokens(theta: Array, sc: Array, kappa: Array) -> Array:
    """
    theta: (N, P)   regional parameter values
    sc:    (N, N)   structural connectivity
    kappa: (P, d)   per-parameter positional tags from the template encoder
    returns: (N, d) one token per region
    """
```

Assert shapes at module boundaries, not inside hot loops.

### 4.3 Size limits

- Functions: aim for under 50 lines. Over 80 means it is doing more than one thing.
- Modules: aim for under 300 lines. Over 500 means it should be split.

These are guides, not gates. A single long table of constants is fine.

### 4.4 Dependencies

Allowed without asking: `jax`, `numpy`, `sympy`, `pandas`, `scipy`, `pyyaml`, `pytest`, `equinox` (if you need JAX modules).

Anything else, ask first. In particular do **not** add: diffrax (see §7.2), a config framework beyond plain YAML, a logging framework beyond `logging`, or an experiment tracker.

### 4.5 Configuration

Plain YAML, loaded into dataclasses with explicit field names. No nested dictionaries passed around as `dict[str, Any]`. If a config field is required, it has no default and loading fails when it is absent.

### 4.6 Tests

Every numerical module needs at least one test against a value computed independently. "Independently" means by hand, by a reference implementation, or by a different code path, not by running the same function and recording its output.

Specifically required:

- `spec/compile.py`: round-trip assertion, and the five reference cards compile without error.
- `sim/integrate.py`: an Ornstein-Uhlenbeck process whose stationary variance is known analytically.
- `sim/summary.py`: FC and FCD on synthetic signals with known correlation structure.
- `data/groups.py`: the Phase 0 verification (§5).

---

## 5. Phase 0: data and groups

**This phase is a gate. Nothing else starts until its verification passes.**

### 5.1 Inputs

| | Path |
|---|---|
| Demographics | `/home/ftian/storage/projects/lifespan_EI/data/HCP-YA/demogr/demogr.csv` |
| Subject SC | `/home/ftian/storage/projects/lifespan_EI/data/HCP-YA/SC/DK68` |
| Reference group FC / FCD / SC | `/home/ftian/storage/projects/lifespan_EI/data/HCP-YA/FC/DK68/group_dl_ds` |
| Original grouping script (MATLAB) | `/home/tzeng/storage/Matlab/DELSSOME` |

### 5.2 First task: investigate, then report before coding

Do **not** start by porting the MATLAB script. Start by answering these questions and writing the answers into `docs/data.md`:

1. What is in the reference directory? How many groups, what files per group, what shapes?
2. **Is the group membership recorded anywhere** (a subject list, an index file, a saved split)? If so, read it. Reading a saved split is far safer than re-deriving one.
3. If membership is not recorded, read the MATLAB script and determine whether the split is deterministic (for example, sort by subject ID then partition) or RNG-dependent.
4. If it is RNG-dependent, **stop and report**. MATLAB's RNG does not match NumPy's, so bit-exact reproduction may require either reading the saved assignment or a careful port of the specific generator. This is a decision for the human, not for you.

Report findings before writing `groups.py`.

### 5.3 Verification

Once group membership is established, recompute from the subject-level data and compare against the reference:

| Quantity | Tolerance |
|---|---|
| Group SC | relative error < 1e-6 per edge |
| Group FC | relative error < 1e-6 per edge |
| Group FCD CDF | max absolute difference < 1e-6 |

If any comparison fails, do not adjust the tolerance. Report the discrepancy with the worst-case entry, the group it came from, and your best hypothesis.

Write this as a test in `tests/test_groups.py` that runs against the real paths, marked so it can be skipped off-cluster.

### 5.4 Output of Phase 0

- `data/groups.py` exposing group membership and a loader for group-level SC, FC and FCD CDF.
- `docs/data.md` documenting what the groups are, how membership was determined, how verification was done, and what the tolerances were.
- A passing verification test.

---

## 6. Phase 1: model spec and compiler

Follows `docs/generation.md` §1 to §5.

### 6.1 The card

`ModelCard` is a frozen dataclass. It is the only description of a model that exists; everything else is derived from it. It must be serialisable to and from YAML without loss.

Fields follow the template in `docs/generation.md` §1: the linear matrix, the per-variable nonlinear terms, the coupling channels, the observable, the observation model, the free-parameter assignment, and the constants.

### 6.2 The compiler

`compile.py` is the most important module in the repository. It does:

1. Card to SymPy expressions, one per state variable, plus one for the observable.
2. Canonicalisation (constant folding, term ordering, removal of `0 * x`).
3. Walk A: emit DAG nodes and edges with the 21-column node features from `docs/architecture.md` §1 and §2.
4. Walk B: emit a drift function against the `ops` interface.
5. Round-trip assertion.

Coupling channels are **not** expressed in SymPy. The scalar expression references a channel symbol; the channel itself is a matrix-vector product computed once per step outside the per-region expression.

### 6.3 Reference cards

Hand-write all five: Linear, MFM, FIC, Wilson-Cowan, Hopf. The slot assignments are tabulated in `docs/generation.md` §1 and in the presentation material.

These are the correctness gate for the whole spec layer. If the template cannot express them exactly, the template is wrong and you should report rather than work around it.

Verification means **matching dynamics and published cost values**, not matching the paper's constant tables. Nondimensionalisation and constant absorption mean the cards will not look character-for-character like the published equations. Expect this.

### 6.4 The ops interface

```python
class Ops(Protocol):
    def add(self, a, b): ...
    def mul(self, a, b): ...
    def exp(self, x): ...
    # ... one method per primitive
```

Two implementations: `JaxOps` and `NumpyOps`. The NumPy one exists for GPU-free unit tests and as a cross-check. A PyTorch one is not needed now; the interface is there so it would be a day's work later.

---

## 7. Phase 2: simulation

### 7.1 Framework

**JAX.** `lax.scan` for the time loop, `vmap` over parameter sets, the whole thing under one `jit`.

### 7.2 The integrator

Euler-Maruyama, fixed `dt = 1 ms`, float32.

**Nest two scans.** The inner one runs integration steps within one TR; the outer one emits BOLD once per TR. A single flat scan materialises 900,000 timepoints per simulation and will exhaust memory before anything else becomes a problem.

Split the PRNG key inside the scan. Do not pre-generate the noise array; it does not fit.

**Do not use diffrax.** Its SDE path carries overhead for adaptive stepping and path reconstruction that this workload does not use.

### 7.3 Observation and summary

`observe.py`: Balloon-Windkessel (appended to the same scan as extra state variables, not a second pass) and direct downsampling to TR.

`summary.py`: computes the statistics in `docs/architecture.md` §5.1. **Reduce to FC and FCD on the GPU.** Returning raw BOLD for a batch is hundreds of megabytes and the transfer will dominate.

The summary pipeline must be used identically in stage 1 and stage 2: same TR, same FCD window length and stride, same arctanh. Put it in one function and call it from both.

### 7.4 JAX-specific costs to plan around

Each distinct model triggers a recompile. Run all parameter sets for one model in a few large batches rather than interleaving models, and keep batch size fixed so shapes stay static.

---

## 8. Phase 3: corpus generation

Follows `docs/generation.md` §5 to §7.

The screen is deliberately minimal: simulate 68 regions, four parameter draws, reject if every draw diverges or every draw's **observed signal** is flat. No parameter-box calibration, no cost-landscape screening.

**Log the acceptance rate.** The corpus budget depends on it and it is currently a guess.

**Log mean off-diagonal FC per model.** A value near 1 across all parameter draws identifies slow-drift models that pass the flatness check but contribute nothing. See `docs/generation.md` §4.

---

## 9. Phase 4: network and training

Follows `docs/architecture.md` in full. Do not start this phase until Phases 0 to 3 are complete and the five reference cards reproduce their published costs.

---

## 10. Cluster usage

### 10.1 Submission

Wrap `/mnt/nas/CSC21/Yeolab/Users/ftian/CBIG_private/setup/CBIG_pbsubmit`.

**Read that script first** to learn its argument format. Do not guess the interface.

`cluster/submit.py` should expose one function that takes a command, resource requests (walltime, memory, GPU), and a job name, builds the call, and submits it. It should also support a dry-run mode that prints the command without submitting, and that mode should be the default in tests.

### 10.2 GPU jobs

Prepend `module load cuda/{CUDA_VERSION};` to the command. Make the CUDA version a config field rather than hard-coding it, and report which version you used in the job log.

### 10.3 Job design

Corpus generation is embarrassingly parallel over models. One job per model (or per small group of models) is the right granularity: it keeps individual jobs short, makes failures cheap to retry, and avoids one long job holding a GPU.

Every job writes a manifest alongside its output recording the config, the seed, the git commit, and the wall time. This is what makes a corpus reproducible.

---

## 11. Documentation

Write these as you go, not at the end.

| File | Contents |
|---|---|
| `README.md` | What this is, how to install, how to run each phase, where outputs land |
| `CLAUDE.md` | Repo map, the non-negotiables from §2, the build order, common pitfalls |
| `docs/data.md` | Phase 0 findings: the groups, how membership was determined, verification results |
| `src/delssome_fm/spec/README.md` | How the card, compiler and DAG fit together. This layer is the least obvious to a newcomer and needs a diagram |
| Module docstrings | One paragraph per module saying what it does and what it does not do |

The test for documentation: **could a new PhD student in the group extend this without reading every line of source?** If the answer depends on reading `compile.py` end to end, the spec README is not good enough.

---

## 12. What not to do

- Do not add abstraction for extensibility that is not yet needed. The design documents list what will be added later; the code does not need to anticipate it beyond keeping the slot lists easy to lengthen.
- Do not write a config system. Plain YAML into dataclasses.
- Do not catch exceptions to keep a pipeline running. A failed simulation is data (flag it); a failed compile is a bug (raise).
- Do not tune tolerances to make a verification pass.
- Do not optimise before measuring. The design document's cost estimates are guesses.
- Do not resolve a disagreement between this brief and the design documents on your own.

---

## 13. Build order

Work through these in order. Each ends with something runnable and tested.

1. **Phase 0.** Investigate the data, report, then implement groups and verification. **Gate.**
2. **Repo skeleton.** `pyproject.toml`, layout, `README.md`, `CLAUDE.md`, config loading, the cluster wrapper with dry-run.
3. **Card and compiler.** `card.py`, `dag.py`, `ops.py`, `compile.py` with the round-trip assertion.
4. **Reference cards.** All five, compiling cleanly. **Gate.**
5. **Simulator.** `integrate.py`, `observe.py`, `summary.py`, tested against the analytic OU case.
6. **Reference reproduction.** Simulate MFM, FIC and Hopf with published parameters and reproduce their published costs against the group-level empirical data from Phase 0. **Gate, and the most important one in the project.**
7. **Sampler and corpus.** `sampler.py`, `corpus.py`, cluster jobs, acceptance-rate logging.
8. **Network and training.** Phase 4.

Stop at each gate and report before continuing.
