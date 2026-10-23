"""Generalisation diagnostic, step 2 of 3: pack the four simulated cells into a store
(train/pack.pack_generalisation), z-scored like the training store.

    python scripts/pack_generalisation.py
"""

import argparse
import json
from pathlib import Path

from delssome_fm.config import (CorpusConfig, DataConfig, GeneralisationConfig, SimConfig,
                                TrainingDataConfig, load_config)
from delssome_fm.sim.corpus import split_scs
from delssome_fm.train.pack import pack_generalisation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/generalisation.yaml"))
    args = p.parse_args()

    data_cfg = load_config(Path("configs/data.yaml"), DataConfig)
    training_store = load_config(Path("configs/training_data.yaml"), TrainingDataConfig).store_dir
    summaries = pack_generalisation(load_config(args.config, GeneralisationConfig),
                                    load_config(Path("configs/sim.yaml"), SimConfig),
                                    {s: split_scs(data_cfg, s) for s in ("train", "val")},
                                    training_store,
                                    # the corpus config scripts/simulate_generalisation.py uses
                                    load_config(Path("configs/corpus.yaml"),
                                                CorpusConfig).parameter_policy)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
