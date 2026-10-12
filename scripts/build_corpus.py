"""Corpus chunk: screen candidates start..stop-1 of one split and simulate the base draws of
the valid ones; or, with --extend, simulate further draws of the models in start..stop-1
already found valid; or, with --rebuild-from DIR, recompute the screening chunk start..stop-1
that an earlier version wrote under DIR (corpus.rebuild_candidates). Writes the chunk (arrays
.npz, plus cards and per-candidate log for screening runs) and a manifest.

    python scripts/build_corpus.py --split train --start 0 --stop 500
    python scripts/build_corpus.py --split train --start 0 --stop 500 --extend \\
        --first-draw 100 --n-param-sets 900
    python scripts/build_corpus.py --split train --start 0 --stop 500 \\
        --rebuild-from outputs/superseded/corpus_v1
"""

import argparse
import dataclasses
import time
from pathlib import Path

from delssome_fm.cluster.manifest import write_manifest
from delssome_fm.config import CorpusConfig, DataConfig, SimConfig, load_config
from delssome_fm.data.groups import SPLIT_NAMES
from delssome_fm.sim.corpus import (acceptance_summary, extend_models, read_chunk, read_logs,
                                    rebuild_candidates, run_candidates, split_scs, write_chunk)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    p.add_argument("--sim-config", type=Path, default=Path("configs/sim.yaml"))
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--split", choices=SPLIT_NAMES, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--extend", action="store_true", help="more draws for valid models")
    p.add_argument("--first-draw", type=int, default=None, help="override; recorded")
    p.add_argument("--n-param-sets", type=int, default=None, help="override; recorded")
    p.add_argument("--rebuild-from", type=Path, default=None, help="old corpus output_dir")
    args = p.parse_args()

    cfg = load_config(args.config, CorpusConfig)
    overrides = {k: v for k, v in (("first_draw", args.first_draw),
                                   ("n_param_sets", args.n_param_sets)) if v is not None}
    cfg = dataclasses.replace(cfg, **overrides)
    sim_cfg = load_config(args.sim_config, SimConfig)
    t0 = time.time()
    scs = split_scs(load_config(args.data_config, DataConfig), args.split)
    if args.extend:
        indices = [r.index for r in read_logs(cfg, args.split)
                   if r.valid and args.start <= r.index < args.stop]
        paths = write_chunk(cfg, args.split, args.start, args.stop, None,
                            extend_models(cfg, sim_cfg, args.split, scs, indices), None)
        summary = f"extended {len(indices)} valid models"
    elif args.rebuild_from is not None:
        old_logs, old_arrays = read_chunk(dataclasses.replace(cfg, output_dir=args.rebuild_from),
                                          args.split, args.start, args.stop)
        logs, arrays, cards = rebuild_candidates(cfg, sim_cfg, args.split, scs, args.start,
                                                 args.stop, old_logs, old_arrays)
        paths = write_chunk(cfg, args.split, args.start, args.stop, logs, arrays, cards)
        summary = acceptance_summary(logs)
    else:
        logs, arrays, cards = run_candidates(cfg, sim_cfg, args.split, scs, args.start, args.stop)
        paths = write_chunk(cfg, args.split, args.start, args.stop, logs, arrays, cards)
        summary = acceptance_summary(logs)
    write_manifest(paths["manifest"], [args.config, args.sim_config, args.data_config],
                   cfg.seed, time.time() - t0,
                   {"split": args.split, "start": args.start, "stop": args.stop,
                    "extend": args.extend, "first_draw": cfg.first_draw,
                    "rebuild_from": None if args.rebuild_from is None else str(args.rebuild_from),
                    "n_param_sets": cfg.n_param_sets})
    print(f"{paths['arrays']}: {summary} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
