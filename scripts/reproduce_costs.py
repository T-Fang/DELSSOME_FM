"""Build step 6: evaluate a range of the original's saved parameter sets for one model with
our simulator, and write the costs plus a job manifest.

    python scripts/reproduce_costs.py --model mfm --sets 0:50
"""

import argparse
import time
from pathlib import Path

import numpy as np

from delssome_fm.cluster.manifest import write_manifest
from delssome_fm.config import ReproduceConfig, SimConfig, load_config
from delssome_fm.sim.reproduce import evaluate, load_original, load_test_inputs, output_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/reproduce.yaml"))
    p.add_argument("--sim-config", type=Path, default=Path("configs/sim.yaml"))
    p.add_argument("--model", required=True)
    p.add_argument("--sets", required=True, help="start:stop, 0-based, e.g. 0:50")
    args = p.parse_args()

    cfg = load_config(args.config, ReproduceConfig)
    sim_cfg = load_config(args.sim_config, SimConfig)
    (model,) = [m for m in cfg.models if m.name == args.model]
    start, stop = (int(v) for v in args.sets.split(":"))
    sets = range(start, stop)

    t0 = time.time()
    costs = evaluate(cfg, sim_cfg, model, load_original(cfg, model), load_test_inputs(cfg), sets)
    out = output_path(cfg, model.name, sets)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, sets=np.arange(start, stop), **costs)
    write_manifest(out.with_suffix(".manifest.json"), [args.config, args.sim_config], cfg.seed,
                   time.time() - t0, {"model": model.name, "sets": args.sets})
    print(f"wrote {out}: mean total {np.nanmean(costs['total']):.4f}, "
          f"diverged simulations {int((cfg.n_dup - costs['n_valid']).sum())}")


if __name__ == "__main__":
    main()
