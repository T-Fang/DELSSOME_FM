"""Assign each of the 68 Desikan regions to one of Yeo's 7 networks by majority vote of its
fsaverage5 vertices, and write assets/yeo7_dk68.csv (data/networks.py reads it). Needs
nibabel and the CBIG repository ($CBIG_CODE_DIR); run once:

    python scripts/make_yeo7_assignment.py

Region order: the 34 left- then 34 right-hemisphere regions of aparc.annot's colour table,
without "unknown" and "corpuscallosum" (the CBIG Desikan-68 order of the HCP-YA data; checked
2026-10-26: myelin peaks in transverse temporal and pericalcarine, the RSFC gradient separates
default-mode from visual and somatomotor regions, left-right homologues correlate at 0.99).
Yeo labels: Wang et al. 2019 (MFMem), 1000subjects_clusters007_ref.mat, on fsaverage5.
"""

import csv
import os
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import scipy.io as sio

CBIG = Path(os.environ["CBIG_CODE_DIR"]) / "stable_projects"
APARC = CBIG / "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label"
YEO = CBIG / "fMRI_dynamics/Wang2018_MFMem/step3_plot/data/1000subjects_clusters007_ref.mat"
OUT = Path(__file__).resolve().parents[1] / "assets/yeo7_dk68.csv"


def main() -> None:
    yeo = sio.loadmat(YEO)
    rows = []
    for hemi in ("lh", "rh"):
        labels, _, names = fs.read_annot(str(APARC / f"{hemi}.aparc.annot"))
        y = yeo[f"{hemi}_labels"].ravel()
        for k, name in enumerate(n.decode() for n in names):
            if name in ("unknown", "corpuscallosum"):
                continue
            counts = np.bincount(y[labels == k], minlength=8)[1:]    # label 0: medial wall
            network = int(np.argmax(counts))
            rows.append({"region": len(rows), "name": f"{hemi}.{name}", "network": network,
                         "vertices": int((labels == k).sum()),
                         "purity": round(float(counts[network] / counts.sum()), 3)})
    if len(rows) != 68:
        raise ValueError(f"expected 68 Desikan regions, found {len(rows)}")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
