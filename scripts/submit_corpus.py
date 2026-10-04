"""Stage-1 corpus: submit one single-CPU job per chunk of candidate indices. Dry run unless
--submit. --only-missing skips chunks whose arrays file already exists (for resubmits).

    python scripts/submit_corpus.py --start 0 --stop 650000 --chunk 200
    python scripts/submit_corpus.py --start 0 --stop 650000 --chunk 200 --submit
"""

import argparse
from pathlib import Path

from delssome_fm.cluster.submit import Resources, submit
from delssome_fm.config import ClusterConfig, CorpusConfig, load_config
from delssome_fm.sim.corpus import chunk_paths

# One thread per job: JAX's CPU backend would otherwise use every core of the node.
ENV = "JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    p.add_argument("--cluster-config", type=Path, default=Path("configs/cluster.yaml"))
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--chunk", type=int, required=True, help="candidates per job")
    p.add_argument("--walltime", default="08:00:00")
    p.add_argument("--mem", default="6G")
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    args = p.parse_args()

    cfg = load_config(args.config, CorpusConfig)
    cluster = load_config(args.cluster_config, ClusterConfig)
    resources = Resources(walltime=args.walltime, memory=args.mem, ncpus=1)
    n = 0
    for a in range(args.start, args.stop, args.chunk):
        b = min(a + args.chunk, args.stop)
        if args.only_missing and chunk_paths(cfg, a, b)["arrays"].exists():
            continue
        command = f"{ENV} python scripts/build_corpus.py --config {args.config} --start {a} --stop {b}"
        submit(command, f"corpus_{a}", resources, cluster, dry_run=not args.submit)
        n += 1
    print(f"{'submitted' if args.submit else 'dry run:'} {n} jobs")


if __name__ == "__main__":
    main()
