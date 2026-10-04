"""Stage-1 corpus: screen candidates start..stop-1, simulate the kept ones, write one chunk
(arrays .npz, cards .jsonl, per-candidate log .jsonl, manifest .json) and print the
acceptance summary.

    python scripts/build_corpus.py --start 0 --stop 100
"""

import argparse
import dataclasses
import time
from pathlib import Path

from delssome_fm.cluster.manifest import write_manifest
from delssome_fm.config import CorpusConfig, DataConfig, SimConfig, load_config
from delssome_fm.sim.corpus import acceptance_summary, run_candidates, training_scs, write_chunk


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    p.add_argument("--sim-config", type=Path, default=Path("configs/sim.yaml"))
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--n-param-sets", type=int, default=None,
                   help="override (pilots only); the manifest records it")
    args = p.parse_args()

    cfg = load_config(args.config, CorpusConfig)
    if args.n_param_sets is not None:
        cfg = dataclasses.replace(cfg, n_param_sets=args.n_param_sets,
                                  batch_size=min(cfg.batch_size, args.n_param_sets))
    sim_cfg = load_config(args.sim_config, SimConfig)
    t0 = time.time()
    scs = training_scs(load_config(args.data_config, DataConfig))
    logs, arrays, cards = run_candidates(cfg, sim_cfg, scs, args.start, args.stop)
    paths = write_chunk(cfg, args.start, args.stop, logs, arrays, cards)
    write_manifest(paths["manifest"], [args.config, args.sim_config, args.data_config],
                   cfg.seed, time.time() - t0,
                   {"start": args.start, "stop": args.stop, "n_param_sets": cfg.n_param_sets})
    print(f"{paths['arrays']}: {acceptance_summary(logs)} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
