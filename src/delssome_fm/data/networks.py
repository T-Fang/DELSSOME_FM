"""Network-level FC (2026-10-26): the 68 Desikan regions grouped into Yeo's 7 networks
(assets/yeo7_dk68.csv, scripts/make_yeo7_assignment.py), and the FC between the networks'
signals, each network's signal being the plain average of its regions' BOLD time series (no
z-scoring, decided 2026-10-26).

The corpus stores no time series, but this FC follows exactly from what it stores. With
regional SDs s_i and FC r_ij (r_ii = 1), the covariance of regions i and j is r_ij s_i s_j,
and averaging is linear, so for networks A and B with n_A, n_B regions

    cov(A, B) = sum_{i in A, j in B} r_ij s_i s_j / (n_A n_B)
    FC_AB     = cov(A, B) / sqrt(cov(A, A) cov(B, B))

(the regions' means drop out of a correlation). Because the signals are not standardised,
a region with a much larger SD than its network partners dominates the network signal.

NETWORK_PAIRS lists the 21 pairs (A, B), A < B, in the order network_fc returns them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

YEO7_NAMES = ("Visual", "Somatomotor", "Dorsal attention", "Ventral attention", "Limbic",
              "Frontoparietal", "Default")
ASSIGNMENT = Path(__file__).resolve().parents[3] / "assets/yeo7_dk68.csv"
N_REGIONS = 68
NETWORK_PAIRS = tuple((a, b) for a in range(7) for b in range(a + 1, 7))


def yeo7_assignment(path: Path = ASSIGNMENT) -> np.ndarray:
    """(68,) the Yeo network (0..6, YEO7_NAMES order) of each Desikan region."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if [int(r["region"]) for r in rows] != list(range(N_REGIONS)):
        raise ValueError(f"{path}: expected regions 0..{N_REGIONS - 1} in order")
    networks = np.array([int(r["network"]) for r in rows])
    if set(networks) != set(range(7)):
        raise ValueError(f"{path}: every one of the 7 networks needs a region, found "
                         f"{sorted(set(networks))}")
    return networks


def network_fc(fc_upper: np.ndarray, sd: np.ndarray, networks: np.ndarray) -> np.ndarray:
    """fc_upper (B, 2278) region FC r over the upper triangle (stored order), sd (B, 68)
    regional SDs -> (B, 21) the FC between network-averaged signals, NETWORK_PAIRS order
    (module docstring). Computed in float64."""
    B = len(fc_upper)
    i, j = np.triu_indices(N_REGIONS, 1)
    r = np.ones((B, N_REGIONS, N_REGIONS))
    r[:, i, j] = r[:, j, i] = fc_upper
    s = np.asarray(sd, np.float64)
    if not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("regional SDs must be finite and positive (rescale large ones in "
                         "log space before exponentiating)")
    s = s / s.max(axis=1, keepdims=True)    # a common scale cancels; avoids overflow
    cov = r * s[:, :, None] * s[:, None, :]
    M = np.zeros((N_REGIONS, 7))
    M[np.arange(N_REGIONS), networks] = 1.0
    M /= M.sum(axis=0)                                   # averaging over each network
    net = np.einsum("ia,nij,jc->nac", M, cov, M, optimize=True)  # (B, 7, 7) covariances
    d = np.sqrt(np.diagonal(net, axis1=1, axis2=2))
    corr = net / d[:, :, None] / d[:, None, :]
    a, b = np.array(NETWORK_PAIRS).T
    return corr[:, a, b]


def stable_arctanh(r: np.ndarray) -> np.ndarray:
    """arctanh with |r| >= 1 clipped to the largest float32 below 1 (as sim/summary.py's
    stable_arctanh for the region FC), so synchronised networks map to 8.66, not inf."""
    bound = 1.0 - np.finfo(np.float32).epsneg
    return np.arctanh(np.clip(r, -bound, bound))
