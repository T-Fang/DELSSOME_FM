"""Pack one split of the corpus into the stage-1 training store (train/pack.py); for the
train split also compute the z-score statistics. Reads ~100 GB for train: run it as a job.

    python scripts/submit_job.py --name pack_train --walltime 08:00:00 --mem 32G --submit \\
        -- python scripts/pack_training_data.py --split train
"""

import argparse
import json
import time
from pathlib import Path

from delssome_fm.config import (CorpusConfig, DataConfig, SimConfig, TrainingDataConfig,
                                load_config)
from delssome_fm.data.groups import SPLIT_NAMES
from delssome_fm.sim.corpus import split_scs
from delssome_fm.train.pack import normalisation, pack_split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/training_data.yaml"))
    p.add_argument("--corpus-config", type=Path, default=Path("configs/corpus.yaml"))
    p.add_argument("--sim-config", type=Path, default=Path("configs/sim.yaml"))
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--split", choices=SPLIT_NAMES, required=True)
    args = p.parse_args()

    t0 = time.time()
    cfg = load_config(args.config, TrainingDataConfig)
    scs = split_scs(load_config(args.data_config, DataConfig), args.split)
    summary = pack_split(load_config(args.corpus_config, CorpusConfig),
                         load_config(args.sim_config, SimConfig), cfg, args.split, scs)
    print(json.dumps(summary, indent=2))
    if args.split == "train":
        stats = normalisation(cfg.store_dir)
        print({k: v.round(4).tolist() for k, v in stats.items()})
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
