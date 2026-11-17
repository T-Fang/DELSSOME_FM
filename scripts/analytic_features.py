"""Analytic features (train/analytic.py, architecture.md §7.14) of store models
--first..--stop-1 of one split: the linearised network's FC and FCD for each of their rows,
written as one chunk under <store>/analytic/<split>/. Run by scripts/submit_analytic.py.

Each model compiles its own functions and JAX keeps them, so the models are computed
--models-per-process at a time in a fresh worker process: memory stays near 2 GB whatever
the chunk size.

    python scripts/analytic_features.py --split val --first 0 --stop 250
"""

import argparse
import multiprocessing
import time
from pathlib import Path

import numpy as np

from delssome_fm.cluster.manifest import write_manifest
from delssome_fm.config import CorpusConfig, SimConfig, TrainingDataConfig, load_config
from delssome_fm.data.groups import SPLIT_NAMES
from delssome_fm.train.analytic import STATUS_CODE, analytic_models, patch_targets, write_chunk


def _part(job: tuple) -> dict[str, np.ndarray]:
    corpus_config, sim_config, store, split, first, stop, surrogates, patch = job
    only = patch_targets(store, split) if patch else None
    return analytic_models(load_config(corpus_config, CorpusConfig),
                           load_config(sim_config, SimConfig), store, split, first, stop,
                           surrogates, only)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/training_data_anchored.yaml"))
    p.add_argument("--corpus-config", type=Path, default=Path("configs/corpus_anchored.yaml"))
    p.add_argument("--sim-config", type=Path, default=Path("configs/sim.yaml"))
    p.add_argument("--split", choices=SPLIT_NAMES, required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--surrogates", type=int, default=4, help="surrogate series per row for the FCD")
    p.add_argument("--models-per-process", type=int, default=10)
    p.add_argument("--patch", action="store_true",
                   help="only the rows without a stable answer in the packed arrays, into "
                        "<store>/analytic_patch (train/analytic.patch_analytic applies them)")
    args = p.parse_args()

    t0 = time.time()
    store = load_config(args.config, TrainingDataConfig).store_dir
    corpus_cfg = load_config(args.corpus_config, CorpusConfig)
    jobs = [(args.corpus_config, args.sim_config, store, args.split, a,
             min(a + args.models_per_process, args.stop), args.surrogates, args.patch)
            for a in range(args.first, args.stop, args.models_per_process)]
    with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        parts = pool.map(_part, jobs, chunksize=1)
    arrays = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    path = write_chunk(store, args.split, args.first, args.stop, arrays, args.surrogates,
                       args.patch)
    write_manifest(path.with_suffix(".manifest.json"),
                   [args.config, args.corpus_config, args.sim_config], corpus_cfg.seed,
                   time.time() - t0, vars(args) | {"config": str(args.config),
                                                   "corpus_config": str(args.corpus_config),
                                                   "sim_config": str(args.sim_config)})
    status = arrays["analytic_status"]
    shares = ", ".join(f"{name} {np.mean(status == code):.3f}"
                       for name, code in STATUS_CODE.items())
    print(f"{path}: {len(status)} rows of models {args.first}..{args.stop - 1} in "
          f"{time.time() - t0:.0f} s; {shares}")


if __name__ == "__main__":
    main()
