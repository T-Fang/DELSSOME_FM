"""Reference-model variants, step 1 of 3 (sim/reference_eval.py, 2026-11-04): one single-CPU
job per model, variant and chunk of draws (random: --n-random corpus draws; the others: one
per fitted set). Dry run unless --submit.

    python scripts/submit_reference_variants.py --n-random 10000 --chunk 200 --submit
    python scripts/submit_reference_variants.py --models fic --variants shuffled \
        scrambled_solved --chunk 50 --submit
    then: scripts/reference_variants_data.py pack ..., scripts/reference_variants_compare.py
"""

import argparse
from pathlib import Path

from delssome_fm.cluster.submit import Resources, submit_batch
from delssome_fm.config import ClusterConfig, ReferenceEvalConfig, ReproduceConfig, load_config
from delssome_fm.sim.reference_eval import VARIANTS

ENV = "JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/reference_eval_sinh.yaml"))
    p.add_argument("--models", nargs="+", default=None, help="default: the config's models")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=["scrambled", "random"])
    p.add_argument("--n-random", type=int, default=None, help="random draws per model")
    p.add_argument("--chunk", type=int, required=True, help="draws per job")
    p.add_argument("--walltime", default="00:45:00")
    p.add_argument("--mem", default="6G")
    p.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    args = p.parse_args()

    cfg = load_config(args.config, ReferenceEvalConfig)
    n_sets = load_config(Path("configs/reproduce.yaml"), ReproduceConfig).n_sets
    resources = Resources(walltime=args.walltime, memory=args.mem, ncpus=1)
    requests = []
    if "random" in args.variants and args.n_random is None:
        p.error("the random variant needs --n-random")
    for name in args.models or cfg.models:
        for variant in args.variants:
            total = args.n_random if variant == "random" else n_sets
            for a in range(0, total, args.chunk):
                n = min(args.chunk, total - a)
                requests.append((f"{ENV} python scripts/reference_variants_data.py simulate "
                                 f"--config {args.config} --model {name} --variant {variant} "
                                 f"--first {a} --n {n}", f"refvar_{name}_{variant}_{a}",
                                 resources))
    submit_batch(requests, load_config(Path("configs/cluster.yaml"), ClusterConfig),
                 dry_run=not args.submit)
    print(f"{'submitted' if args.submit else 'dry run:'} {len(requests)} jobs")


if __name__ == "__main__":
    main()
