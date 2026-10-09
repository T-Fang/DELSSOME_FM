"""Corpus: submit one single-CPU job per chunk of candidate indices of one split. Dry run
unless --submit. --only-missing skips chunks whose arrays file already exists (resubmits).
--extend submits extension jobs (more draws for valid models) instead of screening jobs.

    python scripts/submit_corpus.py --split train --start 0 --stop 500000 --chunk 500 --submit
"""

import argparse
import dataclasses
from pathlib import Path

from delssome_fm.cluster.submit import Resources, submit_batch
from delssome_fm.config import ClusterConfig, CorpusConfig, load_config
from delssome_fm.data.groups import SPLIT_NAMES
from delssome_fm.sim.corpus import chunk_paths

# One thread per job: JAX's CPU backend would otherwise use every core of the node.
ENV = "JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    p.add_argument("--cluster-config", type=Path, default=Path("configs/cluster.yaml"))
    p.add_argument("--split", choices=SPLIT_NAMES, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--chunk", type=int, required=True, help="candidates per job")
    p.add_argument("--extend", action="store_true")
    p.add_argument("--first-draw", type=int, default=None)
    p.add_argument("--n-param-sets", type=int, default=None)
    p.add_argument("--walltime", default="08:00:00")
    p.add_argument("--mem", default="6G")
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    args = p.parse_args()

    cfg = load_config(args.config, CorpusConfig)
    overrides = {k: v for k, v in (("first_draw", args.first_draw),
                                   ("n_param_sets", args.n_param_sets)) if v is not None}
    cfg = dataclasses.replace(cfg, **overrides)
    extra = "".join(f" --{k.replace('_', '-')} {v}" for k, v in overrides.items())
    extra += " --extend" if args.extend else ""
    cluster = load_config(args.cluster_config, ClusterConfig)
    resources = Resources(walltime=args.walltime, memory=args.mem, ncpus=1)
    requests = []
    for a in range(args.start, args.stop, args.chunk):
        b = min(a + args.chunk, args.stop)
        if args.only_missing and chunk_paths(cfg, args.split, a, b)["arrays"].exists():
            continue
        command = (f"{ENV} python scripts/build_corpus.py --config {args.config} "
                   f"--split {args.split} --start {a} --stop {b}{extra}")
        requests.append((command, f"corpus_{args.split}_{a}", resources))
    submit_batch(requests, cluster, dry_run=not args.submit)
    print(f"{'submitted' if args.submit else 'dry run:'} {len(requests)} jobs")


if __name__ == "__main__":
    main()
