"""Yeo-7 network FC of one stage-1 checkpoint (train/report.network_fc_fit): simulated vs
predicted by its fc_net head and/or downsampled from its region-level predictions, on test
and on the rows it trained on. Writes <out>/network_fc_<split>.npz; plot_stage1_report.py
draws network_fc.png from them. Run on a GPU (L40S) job:

    python scripts/network_fc_report_data.py --checkpoint outputs/stage1/sinh_orig/step140000.ckpt \\
        --out outputs/report_sinh/sinh_orig
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from delssome_fm.train.evaluate import load_stage1
from delssome_fm.train.report import network_fc_fit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=50_000, help="rows per split")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    module = load_stage1(args.checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
    for split in ("test", "train"):
        r = network_fc_fit(module, split, args.rows)
        np.savez(args.out / f"network_fc_{split}.npz", checkpoint=str(args.checkpoint), **r)
        c = r["coupled"]
        print(split, ", ".join(f"{k} 1-r coupled {np.nanmedian(r[f'{k}_corr'][c]):.3f} "
                               f"MSE {np.nanmean(r[f'{k}_mse']):.3f} "
                               f"(undefined rows {int(np.isnan(r[f'{k}_mse']).sum())})"
                               for k in ("direct", "downsampled") if f"{k}_corr" in r))


if __name__ == "__main__":
    main()
