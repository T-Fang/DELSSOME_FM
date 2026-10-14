"""Stage 1: pretrain the backbone on the synthetic corpus (train/stage1.py). Needs the
training store (scripts/pack_training_data.py) and a GPU; run it as a job in the training env:

    python scripts/submit_job.py --name stage1 --walltime 48:00:00 --mem 64G --ncpus 9 \\
        --ngpus 1 --gpu-type H100 --conda-env delssome_fm_train --submit \\
        -- env OMP_NUM_THREADS=1 python scripts/train_stage1.py --run-name first

OMP_NUM_THREADS=1: torch otherwise starts one thread per core of the node in every data
loader worker, far more than the job's CPUs.
"""

import argparse
from pathlib import Path

from delssome_fm.config import Stage1Config, load_config
from delssome_fm.train.stage1 import fit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/stage1.yaml"))
    p.add_argument("--run-name", default=None, help="MLflow run and checkpoint directory name")
    args = p.parse_args()
    checkpoints = fit(load_config(args.config, Stage1Config), args.config, args.run_name)
    print(f"checkpoints in {checkpoints}")


if __name__ == "__main__":
    main()
