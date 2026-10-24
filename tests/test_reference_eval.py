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
