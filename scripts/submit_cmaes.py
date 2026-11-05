"""CMA-ES around coupled draws (sim/cmaes.py): one single-CPU job per --chunk plan positions
of one split (scripts/cmaes_data.py plan first). Dry run unless --submit. --only-missing skips
chunks whose arrays exist; --set is passed to every job.

    python scripts/submit_cmaes.py --config configs/cmaes_pilot.yaml --split train \\
        --first 0 --stop 20 --chunk 1 --walltime 01:00:00 --submit
"""

import argparse
from pathlib import Path

from delssome_fm.cluster.submit import Resources, submit_batch
from delssome_fm.config import ClusterConfig, CmaesConfig, load_config, override
from delssome_fm.data.groups import SPLIT_NAMES
from delssome_fm.sim.cmaes import chunk_paths, read_plan

ENV = "JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE")
    p.add_argument("--cluster-config", type=Path, default=Path("configs/cluster.yaml"))
    p.add_argument("--split", choices=SPLIT_NAMES, required=True)
    p.add_argument("--first", type=int, default=0, help="first plan position")
    p.add_argument("--stop", type=int, default=None, help="default: the plan's length")
    p.add_argument("--chunk", type=int, required=True, help="models per job")
    p.add_argument("--walltime", required=True)
    p.add_argument("--mem", default="6G")
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    args = p.parse_args()

    cfg = override(load_config(args.config, CmaesConfig), args.set)
    n = len(read_plan(cfg, args.split))
    stop = n if args.stop is None else args.stop
    if not 0 <= args.first < stop <= n:
        p.error(f"plan positions {args.first}..{stop - 1} outside 0..{n - 1}")
    resources = Resources(walltime=args.walltime, memory=args.mem, ncpus=1)
    sets = f" --set {' '.join(args.set)}" if args.set else ""
    requests = []
    for a in range(args.first, stop, args.chunk):
        b = min(a + args.chunk, stop)
        if args.only_missing and chunk_paths(cfg, args.split, a, b)["arrays"].is_file():
            continue
        requests.append((f"{ENV} python scripts/cmaes_data.py run --config {args.config}{sets} "
                         f"--split {args.split} --first {a} --stop {b}",
                         f"cma_{args.split}_{a}", resources))
    submit_batch(requests, load_config(args.cluster_config, ClusterConfig),
                 dry_run=not args.submit)
    print(f"{'submitted' if args.submit else 'dry run:'} {len(requests)} jobs")


if __name__ == "__main__":
    main()
