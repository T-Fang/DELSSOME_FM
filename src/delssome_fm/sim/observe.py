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

Deviation form. The state stored and integrated is (z, F, V, Q) = (z, f - 1, v - 1, q - 1),
and the equations above are rewritten exactly in those variables, with log1p/expm1 and the
near-cancelling terms expanded analytically:

    dF/dt = z
    dV/dt = (F - expm1(log1p(V) / alpha)) / tau
    dQ/dt = (E1 - Q - (1 + Q) expm1((1/alpha - 1) log1p(V))) / tau
    E1    = f (1 - (1 - rho)^(1/f)) / rho - 1
          = (F (-expm1(L / f)) - (1 - rho) expm1(-L F / f)) / rho,   L = log(1 - rho)
    BOLD  = (100 / rho) V0 (-k1 Q + k2 (V - Q) / (1 + V) - k3 V)

Why: f, v, q stay within ~1e-3 of 1, and BOLD lives in those deviations. Stored as values near
1, a float32 Euler step at dt = 0.5 ms changes q by about its resolution there (1.2e-7), so
the fluctuations are rounded away. That made MFM's float32 costs ~0.9 where the original's
float64 gives ~0.4 (build step 6). Near 0, float32 keeps full relative precision.

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

N_HEMO = 4  # z, F, V, Q
_LOG_1M_RHO = float(jnp.log1p(-RHO))


def bw_initial(n_regions: int, dtype: Any = jnp.float32) -> Array:
    """returns: (N, 4) the resting state, all deviations zero."""
    return jnp.zeros((n_regions, N_HEMO), dtype=dtype)


def bw_derivative(state: Array, u: Array) -> Array:
    """
    state: (N, 4)  z, F = f - 1, V = v - 1, Q = q - 1
    u:     (N,)    neural input (the card's observable)
    returns: (N, 4)
    """
    z, F, V, Q = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    f = 1.0 + F
    dz = u - KAPPA * z - GAMMA * F
    dF = z
    log_v = jnp.log1p(V)
    dV = (F - jnp.expm1(log_v / ALPHA)) / TAU
    e1 = (F * -jnp.expm1(_LOG_1M_RHO / f) - (1.0 - RHO) * jnp.expm1(-_LOG_1M_RHO * F / f)) / RHO
    dQ = (e1 - Q - (1.0 + Q) * jnp.expm1((1.0 / ALPHA - 1.0) * log_v)) / TAU
    return jnp.stack([dz, dF, dV, dQ], axis=-1)


def bw_signal(state: Array) -> Array:
    """state: (N, 4) -> BOLD: (N,)."""
    V, Q = state[:, 2], state[:, 3]
    return BOLD_SCALE * V0 * (-K1 * Q + K2 * (V - Q) / (1.0 + V) - K3 * V)
