"""Build step 6: submit one single-CPU cluster job per (model, parameter set). Dry run unless
--submit. With --only-missing, skips sets whose result file already exists (for resubmits).

    python scripts/submit_reproduction.py                    # print what would be submitted
    python scripts/submit_reproduction.py --submit
    python scripts/submit_reproduction.py --submit --only-missing
"""

import argparse
from pathlib import Path

from delssome_fm.cluster.submit import Resources, submit_batch
from delssome_fm.config import ClusterConfig, ReproduceConfig, load_config
from delssome_fm.sim.reproduce import output_path

# One thread per job: JAX's CPU backend would otherwise use every core of the node.
ENV = "JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/reproduce.yaml"))
    p.add_argument("--cluster-config", type=Path, default=Path("configs/cluster.yaml"))
    p.add_argument("--walltime", default="01:00:00")
    p.add_argument("--mem", default="4G")
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    args = p.parse_args()

    cfg = load_config(args.config, ReproduceConfig)
    cluster = load_config(args.cluster_config, ClusterConfig)
    resources = Resources(walltime=args.walltime, memory=args.mem, ncpus=1)
    requests = []
    for model in cfg.models:
        for s in range(cfg.n_sets):
            if args.only_missing and output_path(cfg, model.name, range(s, s + 1)).exists():
                continue
            command = (f"{ENV} python scripts/reproduce_costs.py --config {args.config} "
                       f"--model {model.name} --sets {s}:{s + 1}")
            requests.append((command, f"repro_{model.name}_{s}", resources))
    submit_batch(requests, cluster, dry_run=not args.submit)
    print(f"{'submitted' if args.submit else 'dry run:'} {len(requests)} jobs")


if __name__ == "__main__":
    main()
