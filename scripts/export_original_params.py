"""Export the original DELSSOME test results (torch .pth) to .npz for build step 6.

Needs torch, which this project does not depend on, so run it once in an environment that
has it (the lifespan_ei env does). Imports nothing from delssome_fm.

    /home/ftian/storage/miniconda/envs/lifespan_ei/bin/python scripts/export_original_params.py \\
        --params-root /mnt/nas/CSC21/Yeolab/Users/tzeng/Python/DELSSOME_plus/params \\
        --out-dir outputs/reproduce/original --n-sets 50

Writes <out-dir>/<model>.npz with params (S, 205), corr, mean, ks (S,), seed (S,) and source.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

MODELS = {"mfm": "pMFM_HCPYA/trial1/test", "fic": "pFIC_HCPYA/trial1/test",
          "hopf": "Hopf_HCPYA/trial1/test"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--params-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-sets", type=int, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, sub in MODELS.items():
        rows = []
        for s in range(1, args.n_sets + 1):
            path = args.params_root / sub / f"seed{s}" / "test_results.pth"
            d = torch.load(path, map_location="cpu", weights_only=False)
            rows.append((d["parameter"].double().numpy().reshape(-1), float(d["corr_loss"][0]),
                         float(d["l1_loss"][0]), float(d["ks_loss"][0]), int(d["seed"]), str(path)))
        params, corr, mean, ks, seed, source = map(np.array, zip(*rows))
        np.savez(args.out_dir / f"{name}.npz", params=params, corr=corr, mean=mean, ks=ks,
                 seed=seed, source=source)
        print(f"{name}: {params.shape}, mean recorded total {np.mean(corr + mean + ks):.3f}")


if __name__ == "__main__":
    main()
