"""Add each model's Const values (node_value.npy, train/pack.pack_node_values) to every split
of a training store packed before 2026-10-29, for the parameter-token backbone. Recompiles
every model's card from the corpus chunks (~4 ms each) and checks the node features against
the stored ones:

    python scripts/pack_node_values.py --config configs/training_data_sinh.yaml \\
        --corpus-config configs/corpus_sinh.yaml
"""

import argparse
import time
from pathlib import Path

from delssome_fm.config import CorpusConfig, TrainingDataConfig, load_config
from delssome_fm.train.pack import pack_node_values


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/training_data_sinh.yaml"))
    p.add_argument("--corpus-config", type=Path, default=Path("configs/corpus_sinh.yaml"))
    args = p.parse_args()
    t0 = time.time()
    store = load_config(args.config, TrainingDataConfig).store_dir
    corpus = load_config(args.corpus_config, CorpusConfig).output_dir
    for split in ("train", "val", "test"):
        chunks = sorted((corpus / split).glob("candidates*_draws0-*.npz"))   # as pack_split
        print(f"{split}: {pack_node_values(store / split, chunks)} models "
              f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
