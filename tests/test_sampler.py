"""spec/sampler.py: determinism, the rules of generation.md §5, the free-parameter assignment
of §4, and the slot frequencies of §2."""

import collections

import numpy as np
import pytest

from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.sampler import (COUPLING_RATIO_LOG10, GAIN_LOG10, MULTIPLIER_LOG10,
                                      N_FREE, SC_ROW_SUM, SINH_SCALE, SINH_Z, V_PROBS,
                                      draw_parameters, sample_card)

CARDS = [sample_card(123, 0, i) for i in range(400)]


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
    assert sample_card(5, 0, 17) == sample_card(5, 0, 17)
    assert sample_card(5, 0, 17) != sample_card(5, 0, 18)
    assert sample_card(5, 0, 17) != sample_card(6, 0, 17)
    assert sample_card(5, 0, 17) != sample_card(5, 1, 17)  # another stream (split)


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


def test_coupling_gain_is_relative_to_the_competing_term():
    for card in CARDS:
        for ch in card.channels:
            if ch.into_derivative:
                v = ch.into_derivative[0]
                c = card.linear[v][v]
                reference = abs(c.value if c.param is None else c.scale)
            else:
                t = card.nonlinear[ch.into_input[0]]
                vals = [abs(w.value if w.param is None else w.scale) for w in t.weights
                        if w is not None]
                if t.drive is not None:
                    vals.append(abs(t.drive.value if t.drive.param is None else t.drive.scale))
                reference = max(vals, default=1.0)
            ratio = np.log10(ch.gain.scale * SC_ROW_SUM / reference)
            assert COUPLING_RATIO_LOG10[0] - 1e-9 <= ratio <= COUPLING_RATIO_LOG10[1] + 1e-9


@pytest.mark.cluster
def test_sc_row_sum_constant_matches_the_training_groups(data_cfg):
    from conftest import data_available
    if not data_available():
        pytest.skip("HCP-YA data not reachable")
    from delssome_fm.sim.corpus import split_scs
    assert split_scs(data_cfg, "train").sum(axis=2).mean() == pytest.approx(SC_ROW_SUM, abs=0.005)



# ---------------------------------------------------------------- the sinh policy (pilot)

SINH = [sample_card(123, 0, i, "sinh") for i in range(400)]


def _slots(card):
    """Every coefficient slot as (where, param name or None, sign of a constant)."""
    def key(c):
        return None if c is None else (c.param, None if c.param else np.sign(c.value))
    out = [(("L", v, u), key(c)) for v, row in enumerate(card.linear) for u, c in enumerate(row)]
    out += [(("beta", v), key(c)) for v, c in enumerate(card.bias)]
    for v, t in enumerate(card.nonlinear):
        if t is not None:
            out += [(("gain", v), np.sign(t.gain)), (("gate", v), t.gate.kind, t.transfer)]
            out += [(("w", v, u), key(c)) for u, c in enumerate(t.weights)]
            out += [(("I", v), key(t.drive))]
    return out


def test_sinh_policy_samples_the_same_equations():
    """Same slots, free parameters, transfers and constant signs; only magnitudes differ."""
    for a, b in zip(CARDS, SINH):
        assert _slots(a) == _slots(b) and a.parameters == b.parameters
        assert a.observable == b.observable and len(a.channels) == len(b.channels)
        assert [s is None for s in a.noise] == [s is None for s in b.noise]
    assert all(c.scale == 1.0 for card in SINH for c in card.noise if c is not None)
    assert all(ch.gain.scale == 1.0 for card in SINH for ch in card.channels)


def test_sinh_constants_are_sinh_uniform():
    """|c| = S sinh(|z|), |z| ~ U(0, Z): at most 1e4, ~5% below S sinh(1), log-uniform above."""
    values = np.abs([c.value for card in SINH for row in card.linear for c in row
                     if c is not None and c.param is None])
    z = np.arcsinh(values / SINH_SCALE) / SINH_Z
    assert values.max() <= 1e4 * (1 + 1e-9)
    assert np.histogram(z, bins=4, range=(0, 1))[0].min() > 0.15 * len(z)


def test_sinh_free_scales_are_signs():
    """A free L diagonal keeps the card's leak/growth sign in its scale; others have 1."""
    for a, b in zip(CARDS, SINH):
        for v in range(b.n_states):
            for u in range(b.n_states):
                c = b.linear[v][u]
                if c is not None and c.param is not None:
                    assert c.scale == (np.sign(a.linear[v][u].scale) if u == v else 1.0)


def test_sinh_draws_share_a_mean_and_relative_spread_per_draw():
    card = next(c for c in SINH if len(c.regional_params) >= 3)
    theta, psi = draw_parameters(card, np.random.default_rng(0), 400, 68, "sinh")
    assert theta.shape == (400, 68, len(card.regional_params))
    diagonal = {card.linear[v][v].param for v in range(card.n_states)}
    for j, p in enumerate(card.regional_params):
        x = theta[..., j]
        mean, sd = x.mean(1), x.std(1)
        assert (sd <= 1.5 * np.abs(mean) + 1e-12).mean() > 0.9      # kappa <= 1, so SD ~ |mean|
        if p == "sigma" or p in diagonal:
            assert (x > 0).all()
        else:
            assert 0.35 < (mean < 0).mean() < 0.65                   # the draw's sign
            assert ((x > 0).any(1) & (x < 0).any(1)).mean() > 0.1  # some draws cross zero
    assert np.abs(np.arcsinh(theta / SINH_SCALE)).max() <= SINH_Z * 1.2
    assert (np.log10(psi) >= GAIN_LOG10[0] - 1e-9).all()
    assert (np.log10(psi) <= GAIN_LOG10[1] + 1e-9).all()


def test_sinh_cards_compile_and_unknown_policies_raise():
    for card in SINH[:50]:
        compile_card(card)
    with pytest.raises(ValueError, match="policy"):
        sample_card(123, 0, 0, "flat")
    with pytest.raises(ValueError, match="policy"):
        draw_parameters(CARDS[0], np.random.default_rng(0), 1, 68, "flat")
