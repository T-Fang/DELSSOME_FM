"""Corpus progress per split: candidates screened, the contiguous range, valid models, and
how many more candidates the targets need at the observed valid rate.

    python scripts/corpus_status.py --target train=100000 --target val=10000 --target test=10000
"""

import argparse
from pathlib import Path

from delssome_fm.config import CorpusConfig, load_config
from delssome_fm.sim.corpus import corpus_status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    p.add_argument("--target", action="append", default=[], help="split=N")
    args = p.parse_args()
    cfg = load_config(args.config, CorpusConfig)
    for item in args.target:
        split, n = item.split("=")
        s = corpus_status(cfg, split)
        need = int(n) - s["valid_in_contiguous"]
        more = int(need / s["valid_rate"]) if need > 0 and s["valid_rate"] > 0 else 0
        print(f"{split}: screened {s['screened']} (contiguous 0..{s['n_contiguous'] - 1}), "
              f"kept {s['kept']}, valid {s['valid']} ({s['valid_rate']:.1%}); "
              f"target {n}: {'reached' if need <= 0 else f'need {need} more, ~{more} candidates'}")


if __name__ == "__main__":
    main()
