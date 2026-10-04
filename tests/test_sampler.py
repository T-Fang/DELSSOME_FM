"""spec/sampler.py: determinism, the rules of generation.md §5, the free-parameter assignment
of §4, and the slot frequencies of §2."""

import collections

import numpy as np
import pytest

from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.sampler import (MULTIPLIER_LOG10, N_FREE, V_PROBS, draw_parameters,
                                      sample_card)

CARDS = [sample_card(123, i) for i in range(400)]


def _nominal_L(card):
    V = card.n_states
    L = np.zeros((V, V))
    for v in range(V):
        for u in range(V):
            c = card.linear[v][u]
            if c is not None:
                L[v, u] = c.value if c.param is None else c.scale
    return L


def test_same_seed_and_index_give_the_same_card():
    assert sample_card(5, 17) == sample_card(5, 17)
    assert sample_card(5, 17) != sample_card(5, 18)
    assert sample_card(5, 17) != sample_card(6, 17)


def test_every_sampled_card_compiles():
    for card in CARDS[:100]:
        compile_card(card)  # includes the DAG round-trip assertion


def test_r1_dissipative_unless_quadratic_gate():
    for card in CARDS:
        quadratic = any(t is not None and t.gate.kind == "quadratic" for t in card.nonlinear)
        if not quadratic:
            assert np.linalg.eigvals(_nominal_L(card)).real.min() > 0, card.name


def test_l_diagonal_always_present():
    for card in CARDS:
        assert all(card.linear[v][v] is not None for v in range(card.n_states))


def test_channel_sources_are_one_hot_and_gains_global():
    for card in CARDS:
        for ch in card.channels:
            assert sorted(ch.source) == [0.0] * (card.n_states - 1) + [1.0]
            assert ch.gain.param in card.global_params
        assert len(card.global_params) == len(card.channels)


def test_free_parameter_assignment():
    """P = n + 1 with n in 1..4 (sigma is the +1), unless no variable is noisy."""
    for card in CARDS:
        regional = card.regional_params
        noisy = [c for c in card.noise if c is not None]
        assert all(c.param == "sigma" for c in noisy)  # one shared sigma
        n = len(regional) - (1 if noisy else 0)
        assert 1 <= n <= max(N_FREE)
        assert ("sigma" in regional) == bool(noisy)


def test_scales_carry_the_nominal_magnitude():
    """Free slots keep their sampled constant as the scale: 8 decades, both signs."""
    scales = [c.scale for card in CARDS for row in card.linear for c in row
              if c is not None and c.param is not None]
    logs = np.log10(np.abs(scales))
    assert logs.min() >= -4 and logs.max() <= 4 and logs.max() - logs.min() > 4


def test_number_of_state_variables_follows_the_table():
    counts = collections.Counter(min(c.n_states, 4) for c in CARDS)
    for k, p in zip((1, 2, 3, 4), V_PROBS):
        assert counts[k] / len(CARDS) == pytest.approx(p, abs=0.07)


def test_nonlinear_term_frequency():
    terms = [t is not None for c in CARDS for t in c.nonlinear]
    assert np.mean(terms) == pytest.approx(0.85, abs=0.05)


def test_every_nonlinear_input_is_nonconstant():
    for card in CARDS:
        for v, t in enumerate(card.nonlinear):
            if t is None:
                continue
            channel_in = any(v in ch.into_input for ch in card.channels)
            assert any(w is not None for w in t.weights) or t.drive is not None or channel_in


def test_draw_parameters_are_log_uniform_multipliers():
    card = CARDS[0]
    theta, psi = draw_parameters(card, np.random.default_rng(0), 50, 68)
    assert theta.shape == (50, 68, len(card.regional_params))
    assert psi.shape == (50, len(card.global_params))
    lo, hi = (10.0 ** b for b in MULTIPLIER_LOG10)
    assert theta.min() >= lo and theta.max() <= hi
    assert np.log10(theta).mean() == pytest.approx(0.0, abs=0.05)
