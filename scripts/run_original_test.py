"""Replay one original DELSSOME test evaluation with Tianchu Zeng's own code, and save its
simulated FC and FCD histogram (which the original runs did not keep).

Mirrors CBIG_{pMFM,pFIC,Hopf}_optimizer.py *_Tester.test: torch.manual_seed(stored seed),
the surviving parameter vector repeated 3 times, the model simulated at dt_test, FC averaged
and FCD histograms averaged over the valid repeats. With the stored seed this should replay the
recorded run exactly; the recomputed cost against the test data is saved next to the recorded
one as a check.

Needs torch and imports the original code read-only, so run it in the lifespan_ei env:
    /home/ftian/storage/miniconda/envs/lifespan_ei/bin/python scripts/run_original_test.py \\
        --model mfm --seed-index 30 --out-dir outputs/reproduce/original_sim
Imports nothing from delssome_fm.
"""

import argparse
import configparser
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/nas/CSC21/Yeolab/Users/tzeng/Python/DELSSOME_plus")
MODELS = {"mfm": ("pMFM_HCPYA", "HCPYA_CMAES_main_pMFM.ini"),
          "fic": ("pFIC_HCPYA", "HCPYA_CMAES_main_pFIC.ini"),
          "hopf": ("Hopf_HCPYA", "HCPYA_CMAES_main_Hopf.ini")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--seed-index", type=int, required=True, help="1..50, the seed<k> directory")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from utils.CBIG_pFIC_utils import (FC_calculate, FCD_calculate, all_loss_calculate_from_fc_fcd,
                                       convert_csv_to_tensor, convert_mat_to_tensor,
                                       set_torch_default)
    torch.set_num_threads(1)
    set_torch_default(-1)  # float64, as the original main scripts do

    params_dir, ini = MODELS[args.model]
    config = configparser.ConfigParser()
    config.read(ROOT / "scripts" / "config" / ini)
    system = config["system"]
    inputs = ROOT / "input" / "HCPYA" / "DK68"
    sc = convert_csv_to_tensor(inputs, "SC_test.csv")
    sc_euler = sc / torch.max(sc) * 0.02
    fc_emp = convert_csv_to_tensor(inputs, "FC_test.csv")
    fcd_emp = convert_mat_to_tensor(inputs, "FCD_CDF_test.mat", "FCD_CDF")
    fcd_emp = fcd_emp / fcd_emp[-1, 0]
    saved = torch.load(ROOT / "params" / params_dir / "trial1" / "test" / f"seed{args.seed_index}"
                       / "test_results.pth", map_location="cpu", weights_only=False)

    torch.manual_seed(int(saved["seed"]))
    parameter = saved["parameter"].repeat(1, 3)
    kw = dict(simulate_time=float(system["simulation_period"]),
              burn_in_time=float(system["t_pre"]), TR=float(system["TR"]),
              warm_up_t=int(system["warmup"]))
    if args.model == "hopf":
        from models.CBIG_Hopf import HopfModel
        model = HopfModel(config, parameter, sc_euler, dt=float(system["dt_test"]))
        bold, valid = model.simulate(simulation_period=kw["simulate_time"],
                                     burn_in_period=kw["burn_in_time"], TR=kw["TR"],
                                     warmup_period=kw["warm_up_t"])
    else:
        module = "models.CBIG_pMFM" if args.model == "mfm" else "models.CBIG_pFIC"
        MfmModel = __import__(module, fromlist=["MfmModel"]).MfmModel
        model = MfmModel(config, parameter, sc_euler, dt=float(system["dt_test"]))
        bold, valid = model.CBIG_mfm_simulation(**kw)

    bold = bold[:, valid, :]                                  # [N, valid repeats, T]
    if bold.shape[1] == 0:
        raise RuntimeError(f"{args.model} seed{args.seed_index}: all 3 repeats invalid")
    fc = FC_calculate(bold).mean(0, keepdim=True)            # [1, N, N]
    _, hist = FCD_calculate(bold, int(system["window_size"]))
    hist = hist.mean(1, keepdim=True)                         # [bins, 1]
    _, corr, l1, ks = all_loss_calculate_from_fc_fcd(fc, hist, fc_emp, fcd_emp)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.model}_seed{args.seed_index}.npz"
    np.savez(out, fc=fc[0].numpy(), fcd_hist=hist[:, 0].numpy(), n_valid=int(valid.sum()),
             recomputed=np.array([float(corr[0]), float(l1[0]), float(ks[0])]),
             recorded=np.array([float(saved["corr_loss"][0]), float(saved["l1_loss"][0]),
                                float(saved["ks_loss"][0])]))
    print(f"{out}: recomputed {np.round([float(corr[0]), float(l1[0]), float(ks[0])], 6)}, "
          f"recorded {np.round([float(saved['corr_loss'][0]), float(saved['l1_loss'][0]), float(saved['ks_loss'][0])], 6)}")


if __name__ == "__main__":
    main()
