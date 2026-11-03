"""sim/reference_eval.py: re-expressing a reference card in the corpus convention keeps its
equations (the same dx/dt for the same physical parameter values) and makes the multipliers
positive and centred on 1."""

import numpy as np
import pytest

from delssome_fm.sim.reference_eval import actual_values, in_convention
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.ops import NumpyOps
from delssome_fm.spec.reference import load_reference

N = 68


@pytest.mark.parametrize("name", ["mfm", "fic", "hopf"])
def test_rescaled_card_has_the_same_equations(name):
    card = load_reference(name)
    rng = np.random.default_rng(0)
    P, K = len(card.regional_params), len(card.global_params)
    # 5 parameter sets with a negative parameter (like MFM's drive and Hopf's a)
    sign = np.array([-1.0 if j == 1 else 1.0 for j in range(P)])
    theta = sign * rng.uniform(0.2, 2.0, (5, N, P))
    psi = rng.uniform(0.5, 3.0, (5, K))
    new_card, new_theta, new_psi, medians = in_convention(card, theta, psi)
    assert (new_theta > 0).all() and (new_psi > 0).all()
    assert abs(np.median(np.log10(new_theta))) < 0.2           # centred on 1
    x, nu = rng.uniform(0, 0.5, (N, card.n_states)), rng.normal(size=(N, card.n_states))
    sc = rng.uniform(0, 0.02, (N, N))
    old_rhs = compile_card(card).rhs(NumpyOps())
    new_rhs = compile_card(new_card).rhs(NumpyOps())
    for s in range(5):
        np.testing.assert_allclose(new_rhs(x, nu, new_theta[s], new_psi[s], sc),
                                   old_rhs(x, nu, theta[s], psi[s], sc), rtol=1e-10, atol=1e-12)



@pytest.mark.parametrize("name", ["mfm", "fic", "hopf"])
def test_actual_values_card_has_the_same_equations_and_unit_scales(name):
    from delssome_fm.spec.card import Coef
    card = load_reference(name)
    rng = np.random.default_rng(1)
    P, K = len(card.regional_params), len(card.global_params)
    theta, psi = rng.uniform(-2.0, 2.0, (3, N, P)), rng.uniform(0.5, 3.0, (3, K))
    new_card, new_theta, new_psi, factors = actual_values(card, theta, psi)
    scales = []

    def collect(x):
        import dataclasses
        if isinstance(x, Coef):
            scales.append(x.scale) if x.param else None
        elif isinstance(x, tuple):
            [collect(v) for v in x]
        elif dataclasses.is_dataclass(x):
            [collect(getattr(x, f.name)) for f in dataclasses.fields(x)]
    collect(new_card)
    assert np.allclose(np.abs(scales), 1.0)
    x, nu = rng.uniform(0, 0.5, (N, card.n_states)), rng.normal(size=(N, card.n_states))
    sc = rng.uniform(0, 0.02, (N, N))
    old_rhs, new_rhs = compile_card(card).rhs(NumpyOps()), compile_card(new_card).rhs(NumpyOps())
    for s in range(3):
        np.testing.assert_allclose(new_rhs(x, nu, new_theta[s], new_psi[s], sc),
                                   old_rhs(x, nu, theta[s], psi[s], sc), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", ["mfm", "fic", "hopf"])
def test_scrambled_draw_keeps_each_map_mean_and_sd_and_drops_its_order(name):
    from delssome_fm.sim.reference_eval import variant_draw
    from delssome_fm.spec.sampler import positive_params
    card = load_reference(name)
    rng = np.random.default_rng(1)
    P, K = len(card.regional_params), len(card.global_params)
    theta = np.sort(rng.uniform(0.5, 2.0, (3, N, P)), axis=1)   # structured: sorted over regions
    psi = rng.uniform(0.5, 3.0, (3, K))
    t, p = variant_draw(card, theta, psi, "scrambled", 1, np.random.default_rng(7))
    assert t.shape == (N, P) and np.array_equal(p, psi[1])
    np.testing.assert_allclose(t.mean(0), theta[1].mean(0), atol=4 * theta[1].std(0).max() / N ** 0.5)
    np.testing.assert_allclose(t.std(0), theta[1].std(0), rtol=0.3)
    assert all(abs(np.corrcoef(t[:, k], theta[1][:, k])[0, 1]) < 0.4 for k in range(P))
    fixed = [k for k, q in enumerate(card.regional_params) if q in positive_params(card)]
    assert (t[:, fixed] >= 0).all()


def test_random_draw_is_the_corpus_draw_and_unknown_draws_raise():
    from delssome_fm.sim.reference_eval import variant_draw
    from delssome_fm.spec.sampler import draw_parameters
    card = load_reference("fic")
    theta, psi = np.ones((2, N, 4)), np.ones((2, 1))
    t, p = variant_draw(card, theta, psi, "random", 0, np.random.default_rng(5))
    t2, p2 = draw_parameters(card, np.random.default_rng(5), 1, N, "sinh")
    np.testing.assert_array_equal(t, t2[0])
    np.testing.assert_array_equal(p, p2[0])
    with pytest.raises(ValueError, match="variant must be one of"):
        variant_draw(card, theta, psi, "permuted", 0, np.random.default_rng(5))
    with pytest.raises(ValueError, match="2 fitted sets"):
        variant_draw(card, theta, psi, "scrambled", 2, np.random.default_rng(5))


def test_shuffled_draw_permutes_whole_regions_and_solved_variants_need_a_constraint():
    from delssome_fm.sim.reference_eval import variant_draw
    card = load_reference("mfm")
    rng = np.random.default_rng(2)
    theta, psi = rng.uniform(0.5, 2.0, (2, N, 3)), rng.uniform(0.5, 3.0, (2, 1))
    t, p = variant_draw(card, theta, psi, "shuffled", 1, np.random.default_rng(3))
    order = np.argsort(t[:, 0])
    np.testing.assert_array_equal(t[order], theta[1][np.argsort(theta[1][:, 0])])  # same rows
    assert not np.array_equal(t, theta[1]) and np.array_equal(p, psi[1])
    with pytest.raises(ValueError, match="implicit constraint"):
        variant_draw(card, theta, psi, "scrambled_solved", 0, np.random.default_rng(3))
    solve = lambda th, ps: th + 1.0                                     # noqa: E731
    t2, _ = variant_draw(card, theta, psi, "scrambled_solved", 0, np.random.default_rng(3), solve)
    t3, _ = variant_draw(card, theta, psi, "scrambled", 0, np.random.default_rng(3), solve)
    np.testing.assert_allclose(t2, t3 + 1.0)                     # solve applied, scrambled ignores it


@pytest.mark.cluster
def test_fic_solver_reproduces_the_fitted_w_ie():
    from delssome_fm.config import ReproduceConfig, load_config
    from delssome_fm.sim.reference_eval import fic_solver, fitted_sets
    theta, psi, sc = fitted_sets(load_config("configs/reproduce.yaml", ReproduceConfig), "fic")
    card, theta, psi, factors = actual_values(load_reference("fic"), theta, psi)
    solve = fic_solver(card, factors, sc)
    for s in (0, 17, 49):
        np.testing.assert_allclose(solve(theta[s], psi[s]), theta[s], rtol=1e-6)
