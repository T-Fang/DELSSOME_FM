"""FIC's implicit constraint: w_IE solved per region so the excitatory rate sits at r_E.

The FIC card (spec/reference/fic.yaml) carries w_IE as a 4th regional column of Theta that
is not searched (decision of 2026-10-05). This module computes that column exactly as the
original DELSSOME does (tzeng `DELSSOME_plus/scripts/models/CBIG_pFIC.py`, __init__ and
_solve_*): one analytic fixed point per parameter set, not Deco 2014's iterative FIC loop.

    1. S_E_ave solves  S_E / (tau_E gamma (1 - S_E)) = r_E          (S_E's steady state)
    2. I_E_ave solves  H_E(I_E) = r_E
    3. I_I_ave solves  W_I I0 + J w_EI S_E_ave - tau_I H_I(I_I) - I_I = 0, per region
    4. S_I_ave = tau_I H_I(I_I_ave)
    5. w_IE = (W_E I0 + J w_EE S_E_ave + G J (C @ 1) S_E_ave - I_E_ave) / S_I_ave

Constants are the FIC card's published values. This module does not simulate.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

I0, W_E, W_I, J_NMDA = 0.382, 1.0, 0.7, 0.15
A_E, B_E, D_E, TAU_E = 310.0, 125.0, 0.16, 0.1
A_I, B_I, D_I, TAU_I = 615.0, 177.0, 0.087, 0.01
GAMMA = 0.641
R_E = 3.0                     # Hz, target excitatory firing rate
S_I_INITIAL = 0.1433408985    # the original's initial S_I


def _H(current: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
    x = a * current - b
    return x / (1.0 - np.exp(-d * x))


def _solve(func, x0, what: str) -> np.ndarray:
    root, _, ier, message = fsolve(func, x0, full_output=True)
    if ier != 1:
        raise RuntimeError(f"FIC solver did not converge for {what}: {message}")
    return root


def excitatory_steady_state(r_e: float = R_E) -> tuple[float, float]:
    """(S_E_ave, I_E_ave) at excitatory rate r_e, with the original's initial guesses."""
    s_e = _solve(lambda s: s / (TAU_E * GAMMA * (1 - s)) - r_e, 0.1641205151, "S_E")[0]
    i_e = _solve(lambda i: _H(i, A_E, B_E, D_E) - r_e, 0.3772259651, "I_E")[0]
    return float(s_e), float(i_e)


def solve_w_ie(w_ee: np.ndarray, w_ei: np.ndarray, G: float, sc: np.ndarray,
               r_e: float = R_E) -> np.ndarray:
    """
    w_ee, w_ei: (N,) regional parameters;  G: global coupling
    sc:         (N, N) connectivity as simulated (already rescaled to max 0.02)
    returns:    (N,) w_IE
    """
    w_ee, w_ei = np.asarray(w_ee, float), np.asarray(w_ei, float)
    s_e, i_e = excitatory_steady_state(r_e)
    i_i = _solve(lambda i: W_I * I0 + J_NMDA * w_ei * s_e - TAU_I * _H(i, A_I, B_I, D_I) - i,
                 np.full(w_ei.shape, 0.296385800197336), "I_I")
    s_i = TAU_I * _H(i_i, A_I, B_I, D_I)
    coupling = G * J_NMDA * np.asarray(sc).sum(axis=1) * s_e
    return (W_E * I0 + w_ee * J_NMDA * s_e + coupling - i_e) / s_i
