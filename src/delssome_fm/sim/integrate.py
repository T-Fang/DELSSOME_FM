"""Euler-Maruyama integration of a compiled model in JAX, emitting one frame per TR.

Two nested `lax.scan`s (brief §7.2): the inner one takes the round(TR/dt) integration steps
of one frame, the outer one emits the observed signal once per frame. Nothing per-step is
materialised. The PRNG key is split inside the inner scan; no noise array is pre-generated.

One step, for the card's right-hand side f (compile.py) with noise input nu = xi / sqrt(dt):

    x <- x + dt f(x, nu)   ==   x + dt drift(x) + sqrt(dt) sigma xi

which is exactly Euler-Maruyama for the template's additive noise.

Compensated summation. The update x <- x + dt f is done with Kahan compensation: each state
carries the rounding error of its last update and adds it back on the next one. In float32 at
dt = 0.5 ms the per-step increment of a slow state (Balloon-Windkessel's f, v, q; MFM's S) is
near the resolution of the state's own value, so plain accumulation loses the fluctuations
that FC and FCD are made of. Build step 6 measured it: MFM seed 30 cost 0.52 +- 0.03 in plain
float32 against 0.43 +- 0.03 in float64 and in the original code. Compensation keeps float32
arithmetic (generation.md §8) at three extra operations per state.

When the card observes
through Balloon-Windkessel, the hemodynamic state advances in the same step with input
o(x) (observe.py); otherwise the observable itself is recorded at each frame.

Divergence is expected for some parameter sets and is reported, never caught: `diverged` is
set when any state or recorded value is non-finite or exceeds `divergence_bound` in absolute
value at a frame boundary. After that the run's values are meaningless and must be ignored.

`make_simulate` returns the function for one parameter set; `make_batch_simulate` vmaps and
jits it. Each distinct card compiles separately, so run all parameter sets of one card in a
few large, fixed-size batches (brief §7.4). This module does not compute statistics.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from delssome_fm.config import SimConfig
from delssome_fm.sim.observe import bw_derivative, bw_initial, bw_signal
from delssome_fm.spec.compile import CompiledModel
from delssome_fm.spec.ops import JaxOps

Array = Any


class SimOutput(NamedTuple):
    """
    observed: (T, N)  recorded signal (BOLD or the downsampled observable), T = n_frames
    diverged: ()      bool; when True, `observed` must not be used
    """

    observed: Array
    diverged: Array


def steps_per_frame(cfg: SimConfig) -> int:
    n = round(cfg.tr / cfg.dt)
    if n < 1 or abs(n * cfg.dt - cfg.tr) > 1e-9 * cfg.tr:
        raise ValueError(f"TR {cfg.tr} s is not a whole number of steps of dt {cfg.dt} s")
    return n


def make_simulate(model: CompiledModel, cfg: SimConfig,
                  dtype: Any = jnp.float32) -> Callable[..., SimOutput]:
    """simulate(key, x0, theta, psi, sc) for one parameter set.

    key:   PRNG key; the run is a deterministic function of it
    x0:    (N, V) initial state
    theta: (N, P) regional parameters, in card.regional_params order
    psi:   (K,)   global parameters, in card.global_params order
    sc:    (N, N) structural connectivity as the simulator should see it (already rescaled)
    """
    ops = JaxOps(dtype)
    rhs = model.rhs(ops)
    observe = model.observe(ops)
    n_steps = steps_per_frame(cfg)
    hemo = model.card.observable.observation == "balloon_windkessel"
    V, P, K = model.card.n_states, len(model.card.regional_params), len(model.card.global_params)
    dt = jnp.asarray(cfg.dt, dtype)
    noise_scale = jnp.asarray(1.0 / cfg.dt ** 0.5, dtype)
    bound = jnp.asarray(cfg.divergence_bound, dtype)

    def simulate(key: Array, x0: Array, theta: Array, psi: Array, sc: Array) -> SimOutput:
        N = x0.shape[0]
        _check_shapes(x0, theta, psi, sc, N, V, P, K)
        x0, theta, psi, sc = (jnp.asarray(a, dtype) for a in (x0, theta, psi, sc))

        def step(carry, _):
            x, cx, h, ch, k = carry
            k, sub = jax.random.split(k)
            nu = jax.random.normal(sub, x.shape, dtype) * noise_scale
            if hemo:
                h, ch = kahan_add(h, ch, dt * bw_derivative(h, observe(x)))
            x, cx = kahan_add(x, cx, dt * rhs(x, nu, theta, psi, sc))
            return (x, cx, h, ch, k), None

        def frame(carry, _):
            state, diverged = carry
            state, _ = jax.lax.scan(step, state, None, length=n_steps)
            x, h = state[0], state[2]
            y = bw_signal(h) if hemo else observe(x)
            bad = (~jnp.all(jnp.isfinite(x)) | ~jnp.all(jnp.isfinite(y))
                   | jnp.any(jnp.abs(x) > bound) | jnp.any(jnp.abs(y) > bound))
            return (state, diverged | bad), y

        h0 = bw_initial(N, dtype)
        init = ((x0, jnp.zeros_like(x0), h0, jnp.zeros_like(h0), key), jnp.asarray(False))
        (_, diverged), frames = jax.lax.scan(frame, init, None,
                                              length=cfg.burn_in_frames + cfg.n_frames)
        return SimOutput(observed=frames[cfg.burn_in_frames:], diverged=diverged)

    return simulate


def kahan_add(total: Array, compensation: Array, increment: Array) -> tuple[Array, Array]:
    """total + increment with Kahan compensation; returns (new total, new compensation)."""
    y = increment - compensation
    t = total + y
    return t, (t - total) - y


def make_batch_simulate(model: CompiledModel, cfg: SimConfig,
                        dtype: Any = jnp.float32) -> Callable[..., SimOutput]:
    """jit(vmap(simulate)): every argument gains a leading batch axis B."""
    return jax.jit(jax.vmap(make_simulate(model, cfg, dtype)))


def rescale_sc(sc: Array, maximum: float = 0.02) -> Array:
    """SC as the original DELSSOME simulated it: sc / max(sc) * 0.02 (SI S3)."""
    top = jnp.max(sc)
    return sc / top * maximum


def _check_shapes(x0, theta, psi, sc, N, V, P, K) -> None:
    expected = {"x0": ((N, V), x0.shape), "theta": ((N, P), theta.shape),
                "psi": ((K,), psi.shape), "sc": ((N, N), sc.shape)}
    for name, (want, got) in expected.items():
        if tuple(got) != want:
            raise ValueError(f"simulate: {name} must have shape {want}, found {tuple(got)}")
