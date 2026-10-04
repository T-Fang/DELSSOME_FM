"""Phase 0 gate: recompute all reference groups from subject-level data and report the worst
discrepancy per quantity. Exits 1 if any comparison exceeds its tolerance.

    python scripts/verify_groups.py --config configs/data.yaml
"""

import argparse
import sys
from pathlib import Path

from delssome_fm.config import DataConfig, load_config
from delssome_fm.data.verify import verify_all_groups, worst_per_quantity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()

    results = verify_all_groups(load_config(args.config, DataConfig))
    for d in worst_per_quantity(results).values():
        print(d.describe())
    failures = [d for d in results if not d.passed]
    for d in failures:
        print(d.describe())
    print(f"{len(results) // 3} groups compared, {len(failures)} comparisons failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
