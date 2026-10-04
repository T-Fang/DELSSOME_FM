"""Observation models: Balloon-Windkessel hemodynamics, and direct downsampling to TR.

Balloon-Windkessel is written as four extra state variables per region (z, f, v, q) with a
derivative function, so integrate.py advances it in the same scan as the neural state, with
the same Euler step and no noise (brief §7.3: not a second pass). Its input is the card's
observable. Direct downsampling needs nothing here: integrate.py records the observable at
each TR.

Equations and constants follow the original DELSSOME code (lifespan_EI
`dynamic_model.py:1110-1116`, tzeng `CBIG_pMFM.py:193-198`; Stephan et al., 2007):

    dz/dt = u - kappa z - gamma (f - 1)
    df/dt = z
    dv/dt = (f - v^(1/alpha)) / tau
    dq/dt = (f (1 - (1 - rho)^(1/f)) / rho - q v^(1/alpha - 1)) / tau
    BOLD  = (100 / rho) V0 (k1 (1 - q) + k2 (1 - q/v) + k3 (1 - v))

The factor 100/rho is in the original code (its comment calls it "a nonsense
multiplication"). It is kept so that BOLD amplitudes match the original pipeline; it does not
change FC or FCD, which are scale-invariant.

This module does not integrate anything and does not compute statistics.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

Array = Any

KAPPA = 0.65      # 1/s, signal decay
GAMMA = 0.41      # 1/s, flow-dependent elimination
TAU = 0.98        # s, hemodynamic transit time
ALPHA = 0.33      # Grubb's exponent
RHO = 0.34        # resting oxygen extraction fraction
V0 = 0.02         # resting blood volume fraction
B0 = 3.0          # T, field strength
TE = 0.0331       # s, echo time
R0 = 110.0        # 1/s, intravascular relaxation rate
EPSILON = 0.47    # intra/extravascular signal ratio
K1 = 4.3 * 28.265 * B0 * TE * RHO
K2 = EPSILON * R0 * TE * RHO
K3 = 1.0 - EPSILON
BOLD_SCALE = 100.0 / RHO

N_HEMO = 4  # z, f, v, q


def bw_initial(n_regions: int, dtype: Any = jnp.float32) -> Array:
    """returns: (N, 4) resting state z = 0, f = v = q = 1."""
    rest = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=dtype)
    return jnp.broadcast_to(rest, (n_regions, N_HEMO))


def bw_derivative(state: Array, u: Array) -> Array:
    """
    state: (N, 4)  z, f, v, q
    u:     (N,)    neural input (the card's observable)
    returns: (N, 4)
    """
    z, f, v, q = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    dz = u - KAPPA * z - GAMMA * (f - 1.0)
    df = z
    dv = (f - v ** (1.0 / ALPHA)) / TAU
    dq = (f * (1.0 - (1.0 - RHO) ** (1.0 / f)) / RHO - q * v ** (1.0 / ALPHA - 1.0)) / TAU
    return jnp.stack([dz, df, dv, dq], axis=-1)


def bw_signal(state: Array) -> Array:
    """state: (N, 4) -> BOLD: (N,)."""
    v, q = state[:, 2], state[:, 3]
    return BOLD_SCALE * V0 * (K1 * (1.0 - q) + K2 * (1.0 - q / v) + K3 * (1.0 - v))
