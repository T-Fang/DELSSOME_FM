"""sim/linear.py against values computed another way.

- A coupled Ornstein-Uhlenbeck network: its stationary covariance under the Euler map is
  written out by hand from theta, G and C, and solved by scipy.
- A damped rotation: the Euler map's decay rate is |1 + (-a + i omega) dt| in closed form,
  weaker than a by about omega^2 dt / 2.
- Balloon-Windkessel: the steady state must zero observe.bw_derivative.
- Against the simulator (a different code path): FC of a long simulation of a coupled
  network observed through Balloon-Windkessel.
- The surrogate: its covariances at lag 0 and lag 1 frame are those of the linearised map.
- A gain beyond the instability is reported as unstable.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from delssome_fm.config import SimConfig
from delssome_fm.sim import observe
from delssome_fm.sim.integrate import make_simulate
from delssome_fm.sim.linear import LinearModel, analytic_fc, bw_steady_state, surrogate
from delssome_fm.sim.summary import functional_connectivity
from delssome_fm.spec.card import Channel, Coef, ModelCard, Observable, Parameter
from delssome_fm.spec.compile import compile_card

DT, TR, N = 0.001, 0.72, 5


def _cfg(n_frames=1200):
    return SimConfig(dt=DT, tr=TR, n_frames=n_frames, burn_in_frames=50, divergence_bound=1e6,
                     fcd_window=83, fcd_bins=10000)


def _ou(observation="direct") -> ModelCard:
    return ModelCard(
        name="ou", states=("x",), linear=((Coef(param="theta"),),), bias=(None,), nonlinear=(None,),
        noise=(Coef(param="sigma"),),
        channels=(Channel("x", (1.0,), "identity", False, Coef(param="G"), (), (0,)),),
        observable=Observable(0, None, observation),
        parameters=(Parameter("theta", "regional"), Parameter("sigma", "regional"),
                    Parameter("G", "global")))


def _network(seed=0):
    rng = np.random.default_rng(seed)
    sc = rng.uniform(0.2, 1.0, (N, N))
    sc = (sc + sc.T) / 2
    np.fill_diagonal(sc, 0.0)
    theta = np.column_stack([rng.uniform(2.0, 3.0, N), rng.uniform(0.5, 1.5, N)])
    return theta, sc


def _upper(m):
    return m[np.triu_indices(len(m), 1)]


def test_coupled_ou_matches_the_hand_written_lyapunov_solution():
    theta, sc = _network()
    G = 0.6
    lin = LinearModel(compile_card(_ou())).linearise(np.zeros((N, 1)), theta, np.array([G]), sc,
                                                     DT)
    assert lin.status == "stable"
    M = np.eye(N) + DT * (-np.diag(theta[:, 0]) + G * sc)          # dx = (-theta x + G C x) dt + ...
    cov = solve_discrete_lyapunov(M, DT * np.diag(theta[:, 1] ** 2))
    sd = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(analytic_fc(lin), _upper(cov / np.outer(sd, sd)), rtol=1e-9)
    np.testing.assert_allclose(lin.decay, -np.log(np.abs(np.linalg.eigvals(M)).max()) / DT,
                               rtol=1e-9)
    np.testing.assert_allclose(lin.rest, 0.0, atol=1e-12)


def test_euler_step_weakens_an_oscillating_modes_damping():
    a, omega = 0.2, 12.0
    card = ModelCard(
        name="rotation", states=("x", "y"),
        linear=((Coef(a), Coef(-omega)), (Coef(omega), Coef(a))), bias=(None, None),
        nonlinear=(None, None), noise=(Coef(param="sigma"), Coef(param="sigma")), channels=(),
        observable=Observable(0, None, "direct"), parameters=(Parameter("sigma", "regional"),))
    lin = LinearModel(compile_card(card)).linearise(np.zeros((N, 2)), np.ones((N, 1)), np.zeros(0),
                                                    np.zeros((N, N)), DT)
    exact = -np.log(abs(1.0 + (-a + 1j * omega) * DT)) / DT
    np.testing.assert_allclose(lin.decay, exact, rtol=1e-9)
    assert abs(lin.decay - (a - omega ** 2 * DT / 2)) < 1e-3 and lin.decay < 0.7 * a


def test_bw_steady_state_zeroes_the_derivative():
    u = np.array([-0.15, 0.0, 0.05, 0.8, 3.0])
    with jax.enable_x64(True):
        d = observe.bw_derivative(jnp.asarray(bw_steady_state(u)), jnp.asarray(u))
    np.testing.assert_allclose(np.asarray(d), 0.0, atol=1e-12)


def test_analytic_fc_matches_a_long_simulation_through_balloon_windkessel():
    theta, sc = _network(1)
    theta[:, 1] *= 0.01                              # small noise: BW stays near its rest
    G, cfg = 0.9, _cfg(6000)
    model = compile_card(_ou("balloon_windkessel"))
    lin = LinearModel(model).linearise(np.zeros((N, 1)), theta, np.array([G]), sc, DT)
    assert lin.status == "stable" and lin.step.shape == (5 * N, 5 * N)
    out = jax.jit(make_simulate(model, cfg))(jax.random.key(3), np.zeros((N, 1)), theta,
                                             np.array([G]), sc)
    assert not bool(out.diverged)
    simulated = _upper(np.asarray(functional_connectivity(out.observed)))
    analytic = analytic_fc(lin)
    assert analytic.min() > 0.1                      # a coupled network: there is something to match
    assert np.abs(simulated - analytic).max() < 0.08


def test_surrogate_has_the_maps_covariances():
    theta, sc = _network(2)
    cfg = _cfg(40000)
    lin = LinearModel(compile_card(_ou())).linearise(np.zeros((N, 1)), theta, np.array([0.5]),
                                                     sc, DT)
    y = surrogate(lin, cfg, np.random.default_rng(5))
    assert y.shape == (cfg.n_frames, N)
    hop = np.linalg.matrix_power(lin.step, 720)
    cov0, cov1 = lin.read @ lin.cov @ lin.read.T, lin.read @ hop @ lin.cov @ lin.read.T
    scale = np.sqrt(np.outer(np.diag(cov0), np.diag(cov0)))
    assert np.abs(y.T @ y / len(y) - cov0).max() < 0.03 * scale.max()
    assert np.abs(y[1:].T @ y[:-1] / (len(y) - 1) - cov1).max() < 0.03 * scale.max()
    again = surrogate(lin, cfg, np.random.default_rng(5))
    np.testing.assert_array_equal(y, again)


def test_gain_beyond_the_instability_is_unstable():
    theta, sc = _network()
    lin = LinearModel(compile_card(_ou())).linearise(np.zeros((N, 1)), theta, np.array([5.0]), sc, DT)
    assert lin.status == "unstable" and lin.decay < 0 and lin.cov is None
    with pytest.raises(ValueError, match="stable"):
        analytic_fc(lin)


def test_fallback_finds_the_state_the_map_settles_at(monkeypatch):
    """Bistable dx = x - x^3 (growth, cubic gate) started at 0.5. The first search is made to
    return the unstable point 0, which the map leaves; the fallback runs the map, which
    settles at +1 (f'(1) = -2)."""
    from delssome_fm.spec.card import Gate, Nonlinear
    card = ModelCard(
        name="bistable", states=("x",), linear=((Coef(-1.0),),), bias=(None,),
        nonlinear=(Nonlinear(-1.0, Gate("quadratic", weights=(1.0,)), "identity",
                             (Coef(1.0),), None),),
        noise=(Coef(param="sigma"),), channels=(), observable=Observable(0, None, "direct"),
        parameters=(Parameter("sigma", "regional"),))
    model = LinearModel(compile_card(card))
    search = model.resting_state

    def unstable_first(x0, theta, psi, sc, newton_only=False):
        return search(x0, theta, psi, sc, newton_only) if newton_only else np.zeros(N)

    monkeypatch.setattr(model, "resting_state", unstable_first)
    lin = model.linearise(np.full((N, 1), 0.5), np.ones((N, 1)), np.zeros(0), np.zeros((N, N)), DT)
    assert lin.status == "stable"
    np.testing.assert_allclose(lin.rest, 1.0, atol=1e-9)
    np.testing.assert_allclose(lin.decay, -np.log(1.0 - 2.0 * DT) / DT, rtol=1e-6)
