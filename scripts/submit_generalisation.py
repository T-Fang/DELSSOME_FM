"""Generalisation diagnostic, step 1 of 3 (sim/generalisation.py): sample the models, then
submit one single-CPU job per cell and chunk of models. Dry run unless --submit.

    python scripts/submit_generalisation.py --chunk 50 --submit
    then: scripts/pack_generalisation.py, scripts/evaluate_stage1.py
"""

import argparse
from pathlib import Path

from delssome_fm.cluster.submit import Resources, submit_batch
from delssome_fm.config import (ClusterConfig, CorpusConfig, GeneralisationConfig,
                                TrainingDataConfig, load_config)
from delssome_fm.sim.generalisation import CELLS, write_sample

ENV = "JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/generalisation.yaml"))
    p.add_argument("--chunk", type=int, required=True, help="models per job")
    p.add_argument("--walltime", default="04:00:00")
    p.add_argument("--mem", default="6G")
    p.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    args = p.parse_args()

    cfg = load_config(args.config, GeneralisationConfig)
    corpus_sizes = load_config(Path("configs/training_data.yaml"), TrainingDataConfig).n_models
    write_sample(load_config(Path("configs/corpus.yaml"), CorpusConfig), corpus_sizes, cfg)
    resources = Resources(walltime=args.walltime, memory=args.mem, ncpus=1)
    requests = [(f"{ENV} python scripts/simulate_generalisation.py --config {args.config} "
                 f"--cell {cell} --start {a} --stop {min(a + args.chunk, cfg.n_models)}",
                 f"gen_{cell}_{a}", resources)
                for cell in CELLS for a in range(0, cfg.n_models, args.chunk)]
    submit_batch(requests, load_config(Path("configs/cluster.yaml"), ClusterConfig),
                 dry_run=not args.submit)
    print(f"{'submitted' if args.submit else 'dry run:'} {len(requests)} jobs")


if __name__ == "__main__":
    main()
