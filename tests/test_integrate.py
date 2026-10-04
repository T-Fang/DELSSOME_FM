"""sim/integrate.py and sim/observe.py against analytic results.

- Ornstein-Uhlenbeck (brief §4.6): the stationary variance and lag-1 autocorrelation of an
  Euler-Maruyama OU process are known in closed form.
- A coupled linear network: its stationary covariance solves a discrete Lyapunov equation
  (scipy), which checks coupling and noise together.
- Noise-free decay: x_n = (1 - theta dt)^n x_0 exactly.
- Balloon-Windkessel: the steady state under constant input is known in closed form.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from delssome_fm.config import SimConfig
from delssome_fm.sim import observe
from delssome_fm.sim.integrate import make_batch_simulate, make_simulate, rescale_sc
from delssome_fm.spec.card import Channel, Coef, ModelCard, Observable, Parameter
from delssome_fm.spec.compile import compile_card

DT, TR = 0.001, 0.72
STEPS = 720


def _cfg(n_frames, burn_in=10, bound=1e6):
    return SimConfig(dt=DT, tr=TR, n_frames=n_frames, burn_in_frames=burn_in,
                     divergence_bound=bound, fcd_window=83, fcd_bins=10000)


def _ou_card(coupled: bool = False) -> ModelCard:
    channels, params = (), [Parameter("theta", "regional"), Parameter("sigma", "regional")]
    if coupled:
        channels = (Channel("x", (1.0,), "identity", False, Coef(param="G"), (), (0,)),)
        params.append(Parameter("G", "global"))
    return ModelCard(name="ou", states=("x",), linear=((Coef(param="theta"),),), bias=(None,),
                     nonlinear=(None,), noise=(Coef(param="sigma"),), channels=channels,
                     observable=Observable(0, None, "direct"), parameters=tuple(params))


def test_ou_stationary_variance_and_autocorrelation():
    theta, sigma, N, B, T = 1.0, 0.5, 68, 4, 1200
    sim = make_batch_simulate(compile_card(_ou_card()), _cfg(T, burn_in=20))
    keys = jax.random.split(jax.random.key(0), B)
    out = sim(keys, jnp.zeros((B, N, 1)), jnp.tile(jnp.array([theta, sigma]), (B, N, 1)),
              jnp.zeros((B, 0)), jnp.zeros((B, N, N)))
    assert not bool(out.diverged.any())
    x = np.asarray(out.observed, dtype=np.float64)          # (B, T, N)
    a = 1.0 - theta * DT
    var_em = sigma ** 2 * DT / (1.0 - a ** 2)              # exact for Euler-Maruyama
    rho = a ** STEPS                                        # autocorrelation over one TR
    # effective sample size of B*N series of length T with lag-1 correlation rho
    n_eff = B * N * T * (1 - rho) / (1 + rho)
    assert x.var() == pytest.approx(var_em, rel=6 * np.sqrt(2 / n_eff))
    lag1 = np.mean([np.corrcoef(x[b, :-1, i], x[b, 1:, i])[0, 1]
                    for b in range(B) for i in range(N)])
    assert lag1 == pytest.approx(rho, abs=6 / np.sqrt(B * N * T))
    assert abs(x.mean()) < 6 * np.sqrt(var_em / n_eff)


def test_coupled_network_matches_discrete_lyapunov():
    """x_{n+1} = M x_n + sigma sqrt(dt) xi with M = I + dt (-theta I + G C)."""
    theta, sigma, G, N, B, T = 1.5, 0.3, 0.8, 3, 16, 1200
    C = np.array([[0.0, 1.0, 0.2], [1.0, 0.0, 0.5], [0.2, 0.5, 0.0]])
    sim = make_batch_simulate(compile_card(_ou_card(coupled=True)), _cfg(T, burn_in=20))
    out = sim(jax.random.split(jax.random.key(1), B), jnp.zeros((B, N, 1)),
              jnp.tile(jnp.array([theta, sigma]), (B, N, 1)), jnp.full((B, 1), G),
              jnp.tile(jnp.asarray(C), (B, 1, 1)))
    x = np.asarray(out.observed, dtype=np.float64).reshape(-1, N)
    M = np.eye(N) + DT * (-theta * np.eye(N) + G * C)
    P = solve_discrete_lyapunov(M, sigma ** 2 * DT * np.eye(N))
    cov = np.cov(x.T)
    np.testing.assert_allclose(cov, P, rtol=0.08, atol=0.08 * np.abs(P).max())
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert corr == pytest.approx(P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]), abs=0.05)


def test_noise_free_decay_is_exact_euler():
    theta, n_frames = 2.0, 3
    sim = make_simulate(compile_card(_ou_card()), _cfg(n_frames, burn_in=0))
    out = sim(jax.random.key(0), jnp.ones((2, 1)), jnp.array([[theta, 0.0]] * 2),
              jnp.zeros((0,)), jnp.zeros((2, 2)))
    expected = (1 - theta * DT) ** (STEPS * np.arange(1, n_frames + 1))
    np.testing.assert_allclose(np.asarray(out.observed)[:, 0], expected, rtol=2e-4)


def test_same_key_same_run_different_key_different_run():
    sim = make_simulate(compile_card(_ou_card()), _cfg(5, burn_in=0))
    args = (jnp.zeros((4, 1)), jnp.array([[1.0, 0.5]] * 4), jnp.zeros((0,)), jnp.zeros((4, 4)))
    a = sim(jax.random.key(7), *args).observed
    b = sim(jax.random.key(7), *args).observed
    c = sim(jax.random.key(8), *args).observed
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.allclose(np.asarray(a), np.asarray(c))


def test_divergence_is_flagged_not_raised():
    sim = make_batch_simulate(compile_card(_ou_card()), _cfg(5, burn_in=0))
    theta = jnp.array([[[1.0, 0.1]], [[-50.0, 0.1]]])  # leak vs growth at rate 50 /s
    out = sim(jax.random.split(jax.random.key(0), 2), jnp.ones((2, 1, 1)), theta,
              jnp.zeros((2, 0)), jnp.zeros((2, 1, 1)))
    assert out.diverged.tolist() == [False, True]


def test_shape_mismatch_raises():
    sim = make_simulate(compile_card(_ou_card()), _cfg(2, burn_in=0))
    with pytest.raises(ValueError, match="theta"):
        sim(jax.random.key(0), jnp.zeros((3, 1)), jnp.zeros((3, 5)), jnp.zeros((0,)),
            jnp.zeros((3, 3)))


def test_tr_must_be_whole_number_of_steps():
    with pytest.raises(ValueError, match="whole number"):
        make_simulate(compile_card(_ou_card()), SimConfig(0.001, 0.7205, 2, 0, 1e6, 83, 10000))


def test_rescale_sc():
    sc = jnp.array([[0.0, 5.0], [10.0, 0.0]])
    np.testing.assert_allclose(np.asarray(rescale_sc(sc)), [[0, 0.01], [0.02, 0]])


# ---------------------------------------------------------------- Balloon-Windkessel


def _bw_steady_state(u):
    """Steady state of the original equations in (z, f, v, q), as deviations from rest."""
    f = 1 + u / observe.GAMMA
    v = f ** observe.ALPHA
    q = f * (1 - (1 - observe.RHO) ** (1 / f)) / observe.RHO / v ** (1 / observe.ALPHA - 1)
    return np.array([0.0, f - 1, v - 1, q - 1])


def _original_bw(state, u):
    """The original code's equations in (z, f, v, q) (tzeng CBIG_pMFM.py:193-198)."""
    z, f, v, q = state.T
    k, g, tau, a, rho = observe.KAPPA, observe.GAMMA, observe.TAU, observe.ALPHA, observe.RHO
    return np.stack([u - k * z - g * (f - 1), z, (f - v ** (1 / a)) / tau,
                     (f / rho * (1 - (1 - rho) ** (1 / f)) - q * v ** (1 / a - 1)) / tau], -1)


def _original_bold(state):
    z, f, v, q = state.T
    return 100 / observe.RHO * observe.V0 * (observe.K1 * (1 - q) + observe.K2 * (1 - q / v)
                                              + observe.K3 * (1 - v))


def test_deviation_form_equals_original_equations():
    rng = np.random.default_rng(0)
    orig = np.stack([rng.normal(0, 0.05, 50), 1 + rng.normal(0, 0.1, 50),
                     1 + rng.normal(0, 0.05, 50), 1 + rng.normal(0, 0.05, 50)], -1)
    dev = orig - np.array([0.0, 1.0, 1.0, 1.0])
    u = rng.uniform(0, 0.5, 50)
    with jax.enable_x64(True):
        got = np.asarray(observe.bw_derivative(jnp.asarray(dev), jnp.asarray(u)))
        bold = np.asarray(observe.bw_signal(jnp.asarray(dev)))
    np.testing.assert_allclose(got, _original_bw(orig, u), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(bold, _original_bold(orig), rtol=1e-9, atol=1e-12)


def test_float32_deviation_form_keeps_small_fluctuations():
    """The reason for the deviation form: a float32 step of size 1e-9 must not vanish."""
    dev = jnp.asarray([[0.0, 1e-6, 1e-6, 1e-6]], jnp.float32)
    d = observe.bw_derivative(dev, jnp.asarray([1e-6], jnp.float32))
    stepped = dev + jnp.float32(5e-4) * d
    # z = 0, so F does not move; V and Q must, by a step of ~1e-9 on values of ~1e-6
    assert bool(jnp.all(d[:, 2:] != 0)) and bool(jnp.all(stepped[:, 2:] != dev[:, 2:]))


def test_balloon_windkessel_steady_state_under_constant_input():
    u_np = np.array([0.0, 0.2, 0.5])
    with jax.enable_x64(True):
        u = jnp.asarray(u_np)
        def step(h, _):
            return h + 1e-3 * observe.bw_derivative(h, u), None
        h, _ = jax.lax.scan(step, observe.bw_initial(3, jnp.float64), None, length=200_000)
        h = np.asarray(h)
        bold = np.asarray(observe.bw_signal(jnp.asarray(h)))
    for i, ui in enumerate(u_np):
        np.testing.assert_allclose(h[i], _bw_steady_state(ui), rtol=1e-6, atol=1e-9)
    assert bold[0] == pytest.approx(0.0, abs=1e-12)   # rest gives zero BOLD
    assert bold[2] > bold[1] > 0                      # more input, more BOLD


def test_bold_constants_match_original_code():
    assert observe.K1 == pytest.approx(4.3 * 28.265 * 3 * 0.0331 * 0.34)
    assert observe.K2 == pytest.approx(0.47 * 110 * 0.0331 * 0.34)
    assert observe.K3 == pytest.approx(0.53)


def test_kahan_keeps_increments_below_float32_resolution():
    """Adding 1e-8 to 1.0 a million times: plain float32 stays at 1, compensated reaches 1.01."""
    from delssome_fm.sim.integrate import kahan_add

    def body(carry, _):
        plain, total, comp = carry
        total, comp = kahan_add(total, comp, jnp.float32(1e-8))
        return (plain + jnp.float32(1e-8), total, comp), None

    one = jnp.float32(1.0)
    (plain, total, _), _ = jax.lax.scan(jax.jit(body), (one, one, jnp.float32(0)), None,
                                        length=1_000_000)
    assert float(plain) == 1.0
    assert float(total) == pytest.approx(1.01, rel=1e-6)
