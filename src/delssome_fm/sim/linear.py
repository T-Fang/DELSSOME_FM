"""The linearised network (architecture.md §7.13): what a model's noise does around its
resting state, without simulating it.

For one draw (theta, psi, SC):

    1. resting state x*      the noise-free network's fixed point, reached from the simulation's
                             initial state by linearly implicit Euler steps of growing size,
                             then Newton steps. If that finds none, or one the simulator's map
                             leaves, the noise-free map itself is run for SETTLE_S from the
                             initial state (what the simulation does) and polished by Newton
    2. linearisation         A = d rhs / dx at x*, B = d rhs / d nu, C = d observable / dx.
                             Under Balloon-Windkessel the hemodynamic states (linearised at each
                             region's steady state for its resting observable) are appended, so
                             the observed signal is linear in one state vector either way
    3. the simulator's map   z <- (I + dt A) z + sqrt(dt) B xi: Euler-Maruyama as
                             sim/integrate.py steps it, not continuous time. The step matters:
                             it weakens an oscillating mode's damping by omega^2 dt / 2
    4. FC                    the stationary covariance S of that map (a discrete Lyapunov
                             equation), read out through C and normalised
    5. a surrogate series    the map sampled every TR is exactly z <- M^n z + eta with
                             cov(eta) = S - M^n S M^nT (n steps per frame), so a Gaussian series
                             with the linearised network's statistics costs T matrix products;
                             sim/summary.summarize then gives its FCD like a simulation's

Status: "stable" (the analysis applies); "unstable" (a fixed point was found but the model
leaves it: it is unstable in continuous time); "integrator" (the fixed point is stable in
continuous time, but the simulator's map does not contract there: a mode faster than Euler's
limit 2 / dt, or an oscillating mode the step undamps, in the model or in the hemodynamic
states; what the simulation shows is then the integrator's, not the model's, and such draws
are excluded like diverged ones, architecture.md §7.17); "none" (no fixed point was found, or
a region's resting observable puts Balloon-Windkessel's flow at or below zero).

The Jacobian is autodiff's, except where autodiff gives a non-finite entry while the drift is
finite: a saturated transfer function (logistic or Wong-Wang far from its operating range)
evaluates to inf * 0 in the chain rule, so those entries are central finite differences.

The numerics are float64 (jax.enable_x64 as a context manager): a slowest mode a thousand
times slower than the local rates is the regime of interest, and float32 loses it.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from delssome_fm.config import SimConfig
from delssome_fm.sim import observe
from delssome_fm.sim.integrate import steps_per_frame
from delssome_fm.spec.compile import CompiledModel
from delssome_fm.spec.ops import JaxOps

STATUSES = ("stable", "unstable", "none", "integrator")
IMPLICIT_STEPS, FIRST_STEP, STEP_GROWTH, LAST_STEP = 120, 1e-3, 1.25, 100.0   # s
NEWTON_STEPS = 8
SETTLE_S = 300.0                # the fallback's run of the noise-free map (burn-in is 144 s)
RESIDUAL = 1e-6                 # |rhs| <= RESIDUAL * (|A| |x*|) at an accepted fixed point
FD_STEP = 1e-6                  # finite differences: h_k = FD_STEP * max(|x_k|, 1e-3)
STATE_BOUND = 1e8


@dataclass(frozen=True)
class Linearised:
    """One draw's linearised network. Arrays are None unless status is "stable".

    rest:   (N, V)   the resting state x*
    step:   (n, n)   M = I + dt A, n = N V (+ 4 N hemodynamic states)
    noise:  (n, m)   sqrt(dt) B, one column per noise input
    read:   (N, n)   the observed signal's gradient
    cov:    (n, n)   the stationary covariance S = M S M^T + noise noise^T
    decay:  the decay rate of the model's slowest mode under the map (the hemodynamic states
            excluded), -ln(spectral radius) / dt, 1/s (<= 0 when "unstable" or "integrator"
            through the model's own modes, NaN when "none")
    """

    status: str
    decay: float
    rest: np.ndarray | None = None
    step: np.ndarray | None = None
    noise: np.ndarray | None = None
    read: np.ndarray | None = None
    cov: np.ndarray | None = None


def bw_steady_state(u: np.ndarray) -> np.ndarray:
    """(N,) constant neural input -> (N, 4) Balloon-Windkessel's steady state (z, F, V, Q as
    observe.bw_derivative holds them). Needs 1 + u / GAMMA > 0."""
    F = np.asarray(u, np.float64) / observe.GAMMA
    f = 1.0 + F
    log_v = observe.ALPHA * np.log(f)
    log_1m_rho = np.log1p(-observe.RHO)
    e1 = (F * -np.expm1(log_1m_rho / f)
          - (1.0 - observe.RHO) * np.expm1(-log_1m_rho * F / f)) / observe.RHO
    Q = (e1 + 1.0) * np.exp(-(1.0 / observe.ALPHA - 1.0) * log_v) - 1.0
    return np.stack([np.zeros_like(F), F, np.expm1(log_v), Q], axis=-1)


class LinearModel:
    """The jitted drift of one compiled card and its derivatives; `linearise` per draw."""

    def __init__(self, model: CompiledModel):
        self.card, self.V = model.card, model.card.n_states
        self.hemo = model.card.observable.observation == "balloon_windkessel"
        ops = JaxOps(jnp.float64)
        rhs, obs, V = model.rhs(ops), model.observe(ops), self.V

        def drift(x, nu, theta, psi, sc):
            return rhs(x.reshape(-1, V), nu.reshape(-1, V), theta, psi, sc).reshape(-1)

        def noise_free(x, theta, psi, sc):
            return drift(x, jnp.zeros_like(x), theta, psi, sc)

        def bw(s, u):
            return observe.bw_derivative(s[None], u[None])[0]

        def settle(x, theta, psi, sc, dt, steps):
            x, _ = jax.lax.scan(lambda x, _: (x + dt * noise_free(x, theta, psi, sc), None),
                                x, None, length=steps)
            return x

        def fd_jac(x, h, theta, psi, sc):
            plus = jax.vmap(lambda e: noise_free(x + e, theta, psi, sc))(jnp.diag(h))
            minus = jax.vmap(lambda e: noise_free(x - e, theta, psi, sc))(jnp.diag(h))
            return ((plus - minus) / (2.0 * h)[:, None]).T          # column k: d rhs / d x_k

        self._drift = jax.jit(noise_free)
        self._settle = jax.jit(settle, static_argnames="steps")
        self._jac = jax.jit(jax.jacfwd(noise_free))
        self._fd_jac = jax.jit(fd_jac)
        self._noise = jax.jit(jax.jacfwd(drift, argnums=1))
        self._obs = jax.jit(lambda x: obs(x.reshape(-1, V)))
        self._obs_jac = jax.jit(jax.jacfwd(lambda x: obs(x.reshape(-1, V))))
        self._bw_jac = jax.jit(jax.vmap(jax.jacfwd(bw, argnums=0)))
        self._bw_in = jax.jit(jax.vmap(jax.jacfwd(bw, argnums=1)))
        self._bw_out = jax.jit(jax.vmap(jax.jacfwd(lambda s: observe.bw_signal(s[None])[0])))

    def jacobian(self, x, theta, psi, sc) -> np.ndarray:
        """(n, n) d rhs / dx at x: autodiff, with any non-finite entry replaced by its central
        finite difference (module docstring). Non-finite only if the drift itself is."""
        J = np.asarray(self._jac(x, theta, psi, sc))
        bad = ~np.isfinite(J)
        if bad.any():
            h = FD_STEP * np.maximum(np.abs(np.asarray(x)), 1e-3)
            J = np.where(bad, np.asarray(self._fd_jac(jnp.asarray(x), jnp.asarray(h), theta, psi,
                                                      sc)), J)
        return J

    def resting_state(self, x0: np.ndarray, theta, psi, sc, newton_only: bool = False,
                      ) -> np.ndarray | None:
        """x0 (N, V) -> x* (N V,) or None. newton_only: polish a state already near x*. Call
        inside jax.enable_x64."""
        x, dt, eye = np.asarray(x0, np.float64).reshape(-1).copy(), FIRST_STEP, np.eye(np.size(x0))
        for k in range(IMPLICIT_STEPS * newton_only, IMPLICIT_STEPS + NEWTON_STEPS):
            f = np.asarray(self._drift(x, theta, psi, sc))
            J = self.jacobian(x, theta, psi, sc) if np.isfinite(f).all() else f[:, None]
            if not (np.isfinite(f).all() and np.isfinite(J).all()):
                return None
            lhs = eye / dt - J if k < IMPLICIT_STEPS else -J
            try:
                x = x + np.linalg.solve(lhs, f)
            except np.linalg.LinAlgError:          # a singular step: no fixed point this way
                return None
            if not np.isfinite(x).all() or np.abs(x).max() > STATE_BOUND:
                return None
            dt = min(dt * STEP_GROWTH, LAST_STEP)
        f = np.asarray(self._drift(x, theta, psi, sc))
        scale = np.abs(self.jacobian(x, theta, psi, sc)) @ np.abs(x) + 1e-12
        return x if (np.abs(f) <= RESIDUAL * scale).all() else None

    def linearise(self, x0: np.ndarray, theta: np.ndarray, psi: np.ndarray, sc: np.ndarray,
                  dt: float) -> Linearised:
        """
        x0:    (N, V)  where the simulation starts
        theta: (N, P)  regional parameters;  psi: (K,) global parameters
        sc:    (N, N)  structural connectivity as the simulator sees it (rescaled)
        dt:    the simulator's step, s
        """
        N = np.shape(x0)[0]
        if np.shape(x0) != (N, self.V) or np.shape(sc) != (N, N) or np.shape(theta)[0] != N:
            raise ValueError(f"x0 {np.shape(x0)}, theta {np.shape(theta)}, sc {np.shape(sc)}: "
                             f"expected (N, {self.V}), (N, P), (N, N)")
        with jax.enable_x64(True):
            theta, psi, sc = (jnp.asarray(np.asarray(a, np.float64)) for a in (theta, psi, sc))
            x = self.resting_state(x0, theta, psi, sc)
            radius = None if x is None else self._radius(x, theta, psi, sc, dt)
            if radius is None or radius >= 1.0:          # the fallback: run the map itself
                near = np.asarray(self._settle(jnp.asarray(np.asarray(x0, np.float64).reshape(-1)),
                                               theta, psi, sc, dt, steps=round(SETTLE_S / dt)))
                bounded = np.isfinite(near).all() and np.abs(near).max() <= STATE_BOUND
                settled = (self.resting_state(near, theta, psi, sc, newton_only=True)
                           if bounded else None)
                if settled is not None:
                    x, radius = settled, self._radius(settled, theta, psi, sc, dt)
            if x is None:
                return Linearised("none", float("nan"))
            A = self.jacobian(x, theta, psi, sc)
            B = np.asarray(self._noise(x, jnp.zeros_like(x), theta, psi, sc))
            C = np.asarray(self._obs_jac(x))
            decay = -np.log(radius) / dt
            B = B[:, np.abs(B).sum(axis=0) > 0]
            if radius >= 1.0:
                return Linearised(_why_unstable(A), decay)
            if B.shape[1] == 0:
                return Linearised("unstable", decay)
            if self.hemo:
                rest_obs = np.asarray(self._obs(x))
                if (1.0 + rest_obs / observe.GAMMA <= 0).any():
                    return Linearised("none", float("nan"))
                A, B, C = self._with_hemodynamics(A, B, C, rest_obs)
        M = np.eye(len(A)) + dt * A
        noise = np.sqrt(dt) * B
        # with the hemodynamic states the map can still fail to contract: at an extreme resting
        # flow Balloon-Windkessel's own rates pass Euler's limit. An eigenvalue at -1 within
        # rounding makes the covariance equation singular (scipy's bilinear transform)
        if self.hemo and np.abs(np.linalg.eigvals(M)).max() >= 1.0:
            return Linearised(_why_unstable(A), decay)
        try:
            cov = solve_discrete_lyapunov(M, noise @ noise.T)
        except np.linalg.LinAlgError:
            return Linearised("unstable", decay)
        if not np.isfinite(cov).all():
            return Linearised("unstable", decay)
        return Linearised("stable", decay, x.reshape(N, self.V), M, noise, C, (cov + cov.T) / 2)

    def _radius(self, x, theta, psi, sc, dt: float) -> float:
        """Spectral radius of the map I + dt A at x: the model's own modes decide stability,
        the hemodynamic states only filter."""
        A = self.jacobian(x, theta, psi, sc)
        if not np.isfinite(A).all():
            return float("inf")
        return float(np.abs(np.linalg.eigvals(np.eye(len(A)) + dt * A)).max())

    def _with_hemodynamics(self, A, B, C, rest_obs):
        """Append each region's linearised Balloon-Windkessel states (4 N) to the system."""
        N, n = len(rest_obs), len(A)
        s = jnp.asarray(bw_steady_state(rest_obs))
        u = jnp.asarray(rest_obs)
        Ah, bh, ch = (np.asarray(a)
                      for a in (self._bw_jac(s, u), self._bw_in(s, u), self._bw_out(s)))
        A2 = np.zeros((n + 4 * N, n + 4 * N))
        A2[:n, :n] = A
        C2 = np.zeros((N, n + 4 * N))
        for i in range(N):
            rows = slice(n + 4 * i, n + 4 * i + 4)
            A2[rows, rows] = Ah[i]
            A2[rows, :n] = np.outer(bh[i], C[i])
            C2[i, rows] = ch[i]
        return A2, np.vstack([B, np.zeros((4 * N, B.shape[1]))]), C2


def _why_unstable(A: np.ndarray) -> str:
    """For a fixed point the simulator's map does not hold: "unstable" if the system A (the
    model's own, or with the hemodynamic states) is unstable in continuous time, else
    "integrator": the step, not the model, makes it unstable."""
    return "unstable" if np.linalg.eigvals(A).real.max() >= 0 else "integrator"


def analytic_fc(lin: Linearised) -> np.ndarray:
    """(E,) the upper triangle (row-major, as summary.upper) of the linearised network's FC.
    Raises unless `lin` is stable and every region's observed signal has variance."""
    if lin.status != "stable":
        raise ValueError(f"analytic FC needs a stable linearisation, found {lin.status!r}")
    cov = lin.read @ lin.cov @ lin.read.T
    sd = np.sqrt(np.diag(cov))
    if not (np.isfinite(sd).all() and (sd > 0).all()):
        raise ValueError("a region's observed signal has no variance in the linearised network")
    corr = cov / np.outer(sd, sd)
    return corr[np.triu_indices(len(corr), 1)]


def surrogate(lin: Linearised, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """(T, N) a Gaussian series with the linearised network's statistics at the scan's
    sampling: the observed signal of the map every cfg.tr, cfg.n_frames frames from the
    stationary state (module docstring, 5)."""
    if lin.status != "stable":
        raise ValueError(f"a surrogate needs a stable linearisation, found {lin.status!r}")
    hop = np.linalg.matrix_power(lin.step, steps_per_frame(cfg))
    start, kick = _factor(lin.cov), _factor(lin.cov - hop @ lin.cov @ hop.T)
    z = start @ rng.standard_normal(start.shape[1])
    out = np.empty((cfg.n_frames, lin.read.shape[0]))
    for t in range(cfg.n_frames):
        out[t] = lin.read @ z
        z = hop @ z + kick @ rng.standard_normal(kick.shape[1])
    return out


def _factor(cov: np.ndarray) -> np.ndarray:
    """F with F F^T = cov for a symmetric positive semi-definite matrix (eigenvalues below
    1e-12 of the largest, the rounding of a rank-deficient covariance, are dropped)."""
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    keep = w > 1e-12 * w.max()
    return v[:, keep] * np.sqrt(w[keep])
