"""Build step 6 pieces: the cost, FIC's w_IE solver, the mapping of the original parameter
vectors onto our cards, and the gate's pass criterion.

The mapping is checked against the original code's own equations (transcribed from tzeng
DELSSOME_plus/scripts/models/CBIG_pMFM.py, CBIG_pFIC.py, CBIG_Hopf.py) evaluated on the
original parameter layout, so a wrong shift, column order or w_IE would show here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pathlib import Path

from delssome_fm.config import ReproduceConfig, load_config
from delssome_fm.sim import fic
from delssome_fm.sim.cost import delssome_cost, fc_cost, ks_cost
from delssome_fm.sim.reproduce import Original, card_inputs, judge
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.ops import NumpyOps
from delssome_fm.spec.reference import load_reference

N = 68
RNG = np.random.default_rng(11)
SC = RNG.uniform(0, 1, (N, N)) * (RNG.uniform(size=(N, N)) < 0.5)
SC = (SC + SC.T) / 2
np.fill_diagonal(SC, 0)
SC = SC / SC.max() * 0.02


def _vector(p1, p2, G, sigma):
    return np.concatenate([p1, p2, [G], sigma])


# ---------------------------------------------------------------- cost


def test_cost_matches_original_formulas():
    a, b = RNG.uniform(-0.2, 0.9, (N, N)), RNG.uniform(-0.2, 0.9, (N, N))
    a, b = (a + a.T) / 2, (b + b.T) / 2
    iu = np.triu_indices(N, 1)
    with jax.enable_x64(True):
        corr, mean = (float(v) for v in fc_cost(jnp.asarray(a), jnp.asarray(b)))
    assert corr == pytest.approx(1 - np.corrcoef(a[iu], b[iu])[0, 1], rel=1e-5)
    assert mean == pytest.approx(abs(a[iu].mean() - b[iu].mean()), rel=1e-5)
    h1, h2 = RNG.integers(0, 50, 1000), RNG.integers(0, 50, 1000)
    c1, c2 = np.cumsum(h1) * 3.0, np.cumsum(h2)  # scale must not matter
    ks = float(ks_cost(jnp.asarray(c1), jnp.asarray(c2)))
    assert ks == pytest.approx(np.max(np.abs(c1 / c1[-1] - c2 / c2[-1])), rel=1e-5)
    total = delssome_cost(jnp.asarray(a), jnp.asarray(c1), jnp.asarray(b), jnp.asarray(c2)).total
    assert float(total) == pytest.approx(corr + mean + ks, rel=1e-6)


def test_identical_inputs_cost_zero():
    a = np.corrcoef(RNG.normal(size=(N, 200)))
    c = np.cumsum(RNG.integers(0, 9, 100)).astype(float)
    cost = delssome_cost(jnp.asarray(a), jnp.asarray(c), jnp.asarray(a), jnp.asarray(c))
    assert float(cost.total) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- FIC solver


def test_w_ie_puts_the_fic_card_at_its_fixed_point():
    """With w_IE solved, S_E = S_E_ave and S_I = S_I_ave in every region is a fixed point of
    the FIC card (no noise): the solver and the card agree on the model."""
    w_ee, w_ei, G = RNG.uniform(1, 8, N), RNG.uniform(1, 4, N), 1.7
    w_ie = fic.solve_w_ie(w_ee, w_ei, G, SC)
    s_e, _ = fic.excitatory_steady_state()
    # S_I_ave from step 4 of the solver: invert by solving the I population at S_E_ave
    model = compile_card(load_reference("fic"))
    rhs = model.rhs(NumpyOps())
    theta = np.stack([w_ee, w_ei, np.zeros(N), w_ie], axis=1)
    # find S_I by integrating the I equation alone at fixed S_E (fast: tau_I = 10 ms)
    s_i = np.full(N, 0.1433408985)
    for _ in range(4000):
        x = np.stack([np.full(N, s_e), s_i], axis=1)
        s_i = s_i + 1e-4 * rhs(x, np.zeros((N, 2)), theta, np.array([G]), SC)[:, 1]
    x = np.stack([np.full(N, s_e), s_i], axis=1)
    d = rhs(x, np.zeros((N, 2)), theta, np.array([G]), SC)
    np.testing.assert_allclose(d, 0.0, atol=1e-6)
    assert s_e == pytest.approx(0.0641 * 3 / (1 + 0.0641 * 3), rel=1e-3)  # the solver's comment


# ---------------------------------------------------------------- mapping onto the cards


def _original_mfm(S, w, I0, G, sigma_unused):
    a, b, d, tau, J, gam = 270.0, 108.0, 0.154, 0.1, 0.2609, 0.641
    I = w * J * S + G * J * (SC @ S) + I0
    r = (a * I - b) / (1 - np.exp(-d * (a * I - b)))
    return -S / tau + (1 - S) * gam * r


def _original_fic(SE, SI, wEE, wEI, G, wIE):
    I0, aE, bE, dE, tE, WE = 0.382, 310.0, 125.0, 0.16, 0.1, 1.0
    aI, bI, dI, tI, WI, J, gam = 615.0, 177.0, 0.087, 0.01, 0.7, 0.15, 0.641
    IE = WE * I0 + wEE * J * SE + G * J * (SC @ SE) - wIE * SI
    II = WI * I0 + wEI * J * SE - SI
    rE = (aE * IE - bE) / (1 - np.exp(-dE * (aE * IE - bE)))
    rI = (aI * II - bI) / (1 - np.exp(-dI * (aI * II - bI)))
    return -SE / tE + (1 - SE) * gam * rE, -SI / tI + rI


def _original_hopf(x, y, a, omega, G):
    rs = SC.sum(1)
    dx = (a - x ** 2 - y ** 2) * x - omega * y + G * (SC @ x - rs * x)
    dy = (a - x ** 2 - y ** 2) * y + omega * x + G * (SC @ y - rs * y)
    return dx, dy


def _card_rhs(name, vector, x):
    theta, psi = card_inputs(name, vector, SC)
    return compile_card(load_reference(name)).rhs(NumpyOps())(x, np.zeros_like(x), theta, psi, SC)


def test_mfm_mapping_matches_original_equations():
    w, I0, sigma = RNG.uniform(0, 1, N), RNG.uniform(0, 0.5, N), RNG.uniform(5e-4, 1e-2, N)
    S = RNG.uniform(0.05, 0.5, N)
    got = _card_rhs("mfm", _vector(w, I0, 1.9, sigma), S[:, None])
    np.testing.assert_allclose(got[:, 0], _original_mfm(S, w, I0, 1.9, sigma), rtol=1e-10)


def test_fic_mapping_matches_original_equations():
    wEE, wEI, sigma = RNG.uniform(1, 9, N), RNG.uniform(1, 4, N), RNG.uniform(5e-4, 1e-2, N)
    G = 2.0
    SE, SI = RNG.uniform(0.1, 0.4, N), RNG.uniform(0.05, 0.2, N)
    got = _card_rhs("fic", _vector(wEE, wEI, G, sigma), np.stack([SE, SI], 1))
    wIE = fic.solve_w_ie(wEE, wEI, G, SC)
    dSE, dSI = _original_fic(SE, SI, wEE, wEI, G, wIE)
    np.testing.assert_allclose(got, np.stack([dSE, dSI], 1), rtol=1e-9)


def test_hopf_mapping_matches_original_equations():
    a, om, sigma = RNG.uniform(-1, 0, N), RNG.uniform(0, 12.5, N), RNG.uniform(5e-4, 1e-2, N)
    x, y = RNG.normal(0, 0.1, N), RNG.normal(0, 0.1, N)
    got = _card_rhs("hopf", _vector(a, om, 2.7, sigma), np.stack([x, y], 1))
    np.testing.assert_allclose(got, np.stack(_original_hopf(x, y, a, om, 2.7), 1),
                               rtol=1e-10, atol=1e-14)


def test_noise_column_is_sigma_for_every_model():
    for name, col in (("mfm", 2), ("fic", 2), ("hopf", 2)):
        sigma = RNG.uniform(5e-4, 1e-2, N)
        theta, _ = card_inputs(name, _vector(RNG.uniform(1, 2, N), RNG.uniform(0.1, 0.4, N),
                                             1.0, sigma), SC)
        np.testing.assert_array_equal(theta[:, col], sigma)


# ---------------------------------------------------------------- the pass criterion


def _cfg():
    """The real thresholds (configs/reproduce.yaml), three models as in the gate."""
    real = load_config(Path(__file__).resolve().parents[1] / "configs" / "reproduce.yaml",
                       ReproduceConfig)
    assert len(real.models) == 3
    return real


def _synthetic(shift=0.0, outliers=0, seed=0):
    rng = np.random.default_rng(seed)
    true = {c: rng.uniform(0.1, 0.4, 50) for c in ("corr", "mean", "ks")}
    noise_sd = 0.02
    rec = {c: v + rng.normal(0, noise_sd, 50) for c, v in true.items()}
    ours = {c: v[:, None] + shift + rng.normal(0, noise_sd, (50, 20)) for c, v in true.items()}
    for i in range(outliers):
        rec["corr"][i] += 10 * noise_sd
    ours["total"] = ours["corr"] + ours["mean"] + ours["ks"]
    ours["n_valid"] = np.full((50, 20), 3)
    return Original(params=np.zeros((50, 205)), **rec), ours


def test_gate_false_failure_rate_is_about_five_percent():
    """Under the null (recorded costs and ours from one noise distribution) the whole gate,
    3 models x 4 components, should fail a perfect implementation about 5% of the time."""
    cfg, n = _cfg(), 300
    fails = sum(not all(r.passed for m in range(3)
                        for r in judge(cfg, *_synthetic(seed=1000 * k + m)))
                for k in range(n))
    assert fails / n < 0.10  # Monte Carlo of the calibration: ~4% expected


def test_judge_passes_a_faithful_reimplementation():
    original, ours = _synthetic(seed=1)
    assert all(r.passed for r in judge(_cfg(), original, ours))


def test_judge_fails_on_systematic_bias():
    original, ours = _synthetic(shift=0.03)
    assert not all(r.passed for r in judge(_cfg(), original, ours))


def test_judge_tolerates_four_outliers_but_not_five():
    original, ours = _synthetic(outliers=4, seed=1)
    corr = next(r for r in judge(_cfg(), original, ours) if r.component == "corr")
    assert corr.misses == 4
    original, ours = _synthetic(outliers=5, seed=1)
    corr = next(r for r in judge(_cfg(), original, ours) if r.component == "corr")
    assert corr.misses == 5 and not corr.passed


def test_judge_counts_fully_diverged_sets_as_misses():
    original, ours = _synthetic()
    for c in ("corr", "mean", "ks", "total"):
        ours[c][:5, 0] = np.nan
    assert all(r.misses >= 5 and not r.passed for r in judge(_cfg(), original, ours))
