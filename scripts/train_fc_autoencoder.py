"""Train the FC autoencoder (train/autoencoder.py) on the training store's FC. Needs the
packed store and a GPU; run it as a job in the training env:

    python scripts/submit_job.py --name fc_ae_64 --walltime 12:00:00 --mem 32G --ncpus 9 \\
        --ngpus 1 --gpu-type L40S --conda-env delssome_fm_train --submit \\
        -- env OMP_NUM_THREADS=1 python scripts/train_fc_autoencoder.py --run-name fc_ae_64
"""

import argparse
from pathlib import Path

from delssome_fm.config import FCAutoencoderConfig, load_config, override
from delssome_fm.train.autoencoder import fit_autoencoder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/fc_autoencoder.yaml"))
    p.add_argument("--run-name", default=None, help="MLflow run and checkpoint directory name")
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="override config fields, e.g. --set latent_dim=32")
    args = p.parse_args()
    cfg = override(load_config(args.config, FCAutoencoderConfig), args.set)
    print(f"best checkpoint: {fit_autoencoder(cfg, args.config, args.run_name)}")


if __name__ == "__main__":
    main()
