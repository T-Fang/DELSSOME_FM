"""Pack one split's CMA-ES draws into the training store as <split>_cma (train/pack.py,
pack_cma_split; generation.md §9): at most 100 per model of its kept run, beside the packed base
split, whose models and z-score statistics they share. Every chunk of the split must exist.
Reads ~200 GB for train: run it as a job.

    python scripts/submit_job.py --name pack_cma_train --walltime 08:00:00 --mem 16G --submit \\
        -- python scripts/pack_cmaes.py --split train
"""

import argparse
import json
import time
from pathlib import Path

from delssome_fm.config import (CmaesConfig, CorpusConfig, SimConfig, TrainingDataConfig,
                                load_config)
from delssome_fm.data.groups import SPLIT_NAMES
from delssome_fm.train.pack import pack_cma_split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/cmaes.yaml"))
    p.add_argument("--training-config", type=Path,
                   default=Path("configs/training_data_sinh_fixed_gain.yaml"))
    p.add_argument("--sim-config", type=Path, default=Path("configs/sim.yaml"))
    p.add_argument("--split", choices=SPLIT_NAMES, required=True)
    args = p.parse_args()

    t0 = time.time()
    cfg = load_config(args.config, CmaesConfig)
    store_dir = load_config(args.training_config, TrainingDataConfig).store_dir
    summary = pack_cma_split(cfg, load_config(cfg.corpus_config, CorpusConfig),
                             load_config(args.sim_config, SimConfig), store_dir, args.split)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
