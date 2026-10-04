"""Build step 6 gate: apply the agreed pass criterion to the costs written by
scripts/reproduce_costs.py. Exits 1 if any model fails.

    python scripts/check_reproduction.py
"""

import argparse
import sys
from pathlib import Path

from delssome_fm.config import ReproduceConfig, load_config
from delssome_fm.sim.reproduce import judge, load_original, load_results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/reproduce.yaml"))
    args = p.parse_args()
    cfg = load_config(args.config, ReproduceConfig)
    failed = False
    for model in cfg.models:
        print(f"== {model.name}")
        for report in judge(cfg, load_original(cfg, model), load_results(cfg, model.name)):
            print("  " + report.describe())
            failed |= not report.passed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
