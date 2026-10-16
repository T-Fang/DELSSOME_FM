"""Generalisation diagnostic: simulate one chunk of one cell (sim/generalisation.py). Run by
the jobs of scripts/submit_generalisation.py.

    python scripts/simulate_generalisation.py --cell seen_model_unseen_sc --start 0 --stop 50
"""

import argparse
from pathlib import Path

from delssome_fm.config import (CorpusConfig, DataConfig, GeneralisationConfig, SimConfig,
                                load_config)
from delssome_fm.sim.corpus import split_scs
from delssome_fm.sim.generalisation import CELLS, chunk_path, read_sample, simulate_cell


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/generalisation.yaml"))
    p.add_argument("--cell", choices=sorted(CELLS), required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    args = p.parse_args()

    cfg = load_config(args.config, GeneralisationConfig)
    model_split, sc_split = CELLS[args.cell]
    indices = read_sample(cfg)[model_split][args.start:args.stop]
    scs = split_scs(load_config(Path("configs/data.yaml"), DataConfig), sc_split)
    path = simulate_cell(load_config(Path("configs/corpus.yaml"), CorpusConfig),
                         load_config(Path("configs/sim.yaml"), SimConfig), cfg, args.cell,
                         indices, scs, chunk_path(cfg, args.cell, args.start, args.stop))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
