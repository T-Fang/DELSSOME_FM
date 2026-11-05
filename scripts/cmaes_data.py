"""CMA-ES around coupled draws (sim/cmaes.py). plan: list a split's rarely coupled corpus
models (<output_dir>/<split>/plan.json). run: CMA-ES for plan positions first..stop-1 (one
job; scripts/submit_cmaes.py submits them). --set overrides config fields (recorded).

    python scripts/cmaes_data.py plan --config configs/cmaes_pilot.yaml --split train
    python scripts/cmaes_data.py run --config configs/cmaes_pilot.yaml --split train \\
        --first 0 --stop 1
"""

import argparse
import time
from pathlib import Path

from delssome_fm.cluster.manifest import write_manifest
from delssome_fm.config import (CmaesConfig, CorpusConfig, DataConfig, SimConfig, load_config,
                                override)
from delssome_fm.data.groups import SPLIT_NAMES
from delssome_fm.sim.cmaes import plan_split, run_chunk, write_plan
from delssome_fm.sim.corpus import split_scs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("plan", "run"))
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE")
    p.add_argument("--sim-config", type=Path, default=Path("configs/sim.yaml"))
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--split", choices=SPLIT_NAMES, required=True)
    p.add_argument("--first", type=int, help="run: first plan position")
    p.add_argument("--stop", type=int, help="run: one past the last plan position")
    args = p.parse_args()

    t0 = time.time()
    cfg = override(load_config(args.config, CmaesConfig), args.set)
    corpus_cfg = load_config(cfg.corpus_config, CorpusConfig)
    if args.command == "plan":
        models = plan_split(cfg, corpus_cfg, args.split)
        path = write_plan(cfg, args.split, models)
        print(f"{path}: {len(models)} of {getattr(cfg.n_models, args.split)} corpus models "
              f"[{time.time() - t0:.0f}s]")
        return
    if args.first is None or args.stop is None:
        p.error("run needs --first and --stop")
    scs = split_scs(load_config(args.data_config, DataConfig), args.split)
    paths = run_chunk(cfg, corpus_cfg, load_config(args.sim_config, SimConfig),
                      load_config(args.data_config, DataConfig), args.split, args.first,
                      args.stop, scs)
    write_manifest(paths["manifest"], [args.config, cfg.corpus_config, args.sim_config,
                                       args.data_config], corpus_cfg.seed, time.time() - t0,
                   {"split": args.split, "first": args.first, "stop": args.stop,
                    "overrides": args.set})
    print(f"{paths['arrays']} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
