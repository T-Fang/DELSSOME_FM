"""Submit one single-CPU job per (model, seed) replaying the original DELSSOME test run with
scripts/run_original_test.py (in the lifespan_ei env, which has torch). Dry run unless
--submit; --only-missing skips (model, seed) pairs whose output already exists.

    python scripts/submit_original_test.py --out-dir outputs/reproduce/original_sim --submit
"""

import argparse
from pathlib import Path

from delssome_fm.cluster.submit import Resources, submit_batch
from delssome_fm.config import ClusterConfig, load_config

TORCH_PYTHON = "/home/ftian/storage/miniconda/envs/lifespan_ei/bin/python"
ENV = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cluster-config", type=Path, default=Path("configs/cluster.yaml"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--walltime", default="02:00:00")
    p.add_argument("--mem", default="4G")
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    args = p.parse_args()

    cluster = load_config(args.cluster_config, ClusterConfig)
    resources = Resources(walltime=args.walltime, memory=args.mem, ncpus=1)
    requests = []
    for model in ("mfm", "fic", "hopf"):
        for k in range(1, args.n_seeds + 1):
            if args.only_missing and (args.out_dir / f"{model}_seed{k}.npz").exists():
                continue
            command = (f"{ENV} {TORCH_PYTHON} scripts/run_original_test.py --model {model} "
                       f"--seed-index {k} --out-dir {args.out_dir}")
            requests.append((command, f"orig_{model}_{k}", resources))
    submit_batch(requests, cluster, dry_run=not args.submit)
    print(f"{'submitted' if args.submit else 'dry run:'} {len(requests)} jobs")


if __name__ == "__main__":
    main()
