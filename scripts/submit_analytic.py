"""Analytic features for a training store (scripts/analytic_features.py): one single-CPU job
per --chunk store models of one split. Dry run unless --submit. --only-missing skips chunks
that exist. --code-dir runs the jobs from that checkout (a git worktree of the submitted
commit: its configs, and its src first on PYTHONPATH), so later edits to the repository
cannot reach jobs still in the queue.

    python scripts/submit_analytic.py --split val --chunk 250 --walltime 08:00:00 --submit
"""

import argparse
from pathlib import Path

import numpy as np

from delssome_fm.cluster.submit import Resources, submit_batch
from delssome_fm.config import ClusterConfig, TrainingDataConfig, load_config
from delssome_fm.data.groups import SPLIT_NAMES
from delssome_fm.train.analytic import chunk_path

ENV = ("JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1 "
       "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/training_data_anchored.yaml"))
    p.add_argument("--corpus-config", type=Path, default=Path("configs/corpus_anchored.yaml"))
    p.add_argument("--cluster-config", type=Path, default=Path("configs/cluster.yaml"))
    p.add_argument("--split", choices=SPLIT_NAMES, required=True)
    p.add_argument("--first", type=int, default=0, help="first store model")
    p.add_argument("--stop", type=int, default=None, help="default: every model of the split")
    p.add_argument("--chunk", type=int, required=True, help="store models per job")
    p.add_argument("--surrogates", type=int, default=4)
    p.add_argument("--walltime", required=True)
    p.add_argument("--mem", default="4G")
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--patch", action="store_true", help="recompute the rows without a stable "
                   "answer only (scripts/analytic_features.py --patch)")
    p.add_argument("--code-dir", type=Path, default=None,
                   help="run from this checkout (git worktree) instead of the live repository")
    p.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    args = p.parse_args()

    store = load_config(args.config, TrainingDataConfig).store_dir
    with np.load(store / args.split / "models.npz") as m:
        n = len(m["index"])
    stop = n if args.stop is None else args.stop
    if not 0 <= args.first < stop <= n:
        p.error(f"models {args.first}..{stop - 1} outside the split's 0..{n - 1}")
    prefix = ENV
    if args.code_dir is not None:
        code = args.code_dir.resolve()
        if not (code / "src" / "delssome_fm").is_dir() or not (code / ".git").exists():
            p.error(f"{code} is not a checkout of this repository (src/delssome_fm, .git)")
        for cfg in (args.config, args.corpus_config):
            if (code / cfg).read_text() != cfg.read_text():
                p.error(f"{code / cfg} differs from {cfg}")
        prefix = f"cd {code} && {ENV} PYTHONPATH={code / 'src'}"
    resources = Resources(walltime=args.walltime, memory=args.mem, ncpus=1)
    requests = []
    for a in range(args.first, stop, args.chunk):
        b = min(a + args.chunk, stop)
        if args.only_missing and chunk_path(store, args.split, a, b, args.patch).is_file():
            continue
        requests.append((f"{prefix} python scripts/analytic_features.py --config {args.config} "
                         f"--corpus-config {args.corpus_config} --split {args.split} "
                         f"--first {a} --stop {b} --surrogates {args.surrogates}"
                         f"{' --patch' if args.patch else ''}",
                         f"analytic_{args.split}_{a}", resources))
    submit_batch(requests, load_config(args.cluster_config, ClusterConfig),
                 dry_run=not args.submit)
    print(f"{'submitted' if args.submit else 'dry run:'} {len(requests)} jobs")


if __name__ == "__main__":
    main()
