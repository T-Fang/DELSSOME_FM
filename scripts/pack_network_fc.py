"""Add the Yeo-7 network FC target (train/pack.pack_network_fc) to every split of a training
store, then recompute its z-score statistics. Reads the store's FC (~40 GB for train): run it
as a CPU job:

    python scripts/submit_job.py --name pack_fc_net --walltime 03:00:00 --mem 16G --submit \\
        -- python scripts/pack_network_fc.py --config configs/training_data_sinh.yaml
"""

import argparse
import time
from pathlib import Path

from delssome_fm.config import TrainingDataConfig, load_config
from delssome_fm.train.pack import normalisation, pack_network_fc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/training_data_sinh.yaml"))
    args = p.parse_args()
    t0 = time.time()
    store = load_config(args.config, TrainingDataConfig).store_dir
    for split in ("train", "val", "test"):
        print(f"{split}: {pack_network_fc(store / split)} rows")
    stats = normalisation(store)
    print("fc_net mean", stats["fc_net_mean"].round(3).tolist())
    print("fc_net sd", stats["fc_net_sd"].round(3).tolist())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
