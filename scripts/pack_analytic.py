"""Pack one split's analytic-feature chunks (scripts/submit_analytic.py) into the store split
as analytic_{status,decay,pairwise,fcd}.npy (train/analytic.pack_analytic). Every chunk of the
split must exist.

    python scripts/pack_analytic.py --split val
"""

import argparse
import json
import time
from pathlib import Path

from delssome_fm.config import TrainingDataConfig, load_config
from delssome_fm.data.groups import SPLIT_NAMES
from delssome_fm.train.analytic import pack_analytic, patch_analytic


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/training_data_anchored.yaml"))
    p.add_argument("--split", choices=SPLIT_NAMES, required=True)
    p.add_argument("--patch", action="store_true",
                   help="apply the patch chunks over the packed arrays (patch_analytic)")
    args = p.parse_args()
    t0 = time.time()
    store = load_config(args.config, TrainingDataConfig).store_dir
    summary = (patch_analytic if args.patch else pack_analytic)(store, args.split)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
