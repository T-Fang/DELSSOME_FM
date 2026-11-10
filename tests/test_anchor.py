"""spec/anchor.py and the anchored prior of spec/sampler.py: a solved model rests at its
resting state, there its coupling ratio is the effective loop, and nominal levels stay in
their window."""

from types import SimpleNamespace

import numpy as np
import pytest

from delssome_fm.spec.anchor import (DIRECT_RATIO_LOG10, INPUT_LOG10, INPUT_RANGE, RATE_LIMIT,
                                     RATIO_LOG10, REST_LOG10, REST_UNIT, gate, is_stable,
                                     transfer)
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.ops import NumpyOps
from delssome_fm.spec.sampler import (COUPLING_RATIO_LOG10, LEVEL_WINDOW_LOG10, SC_ROW_SUM,
                                      SINH_SCALE, draw_level_spread, gain_site, rate_params,
                                      sample_card, sample_model)

FLOOR = 0.1
MODELS = [sample_model(123, 0, i, "sinh", FLOOR, "anchored") for i in range(300)]
N = 4
SC = np.full((N, N), SC_ROW_SUM / (N - 1)) * (1 - np.eye(N))      # every row sum = SC_ROW_SUM


def _value(c, sample):
    """A coefficient's nominal value: the constant, or scale x the slot's nominal level."""
    if c is None:
        return 0.0
    if c.param is None:
        return c.value
    return c.scale * sample.nominal[sample.card.regional_params.index(c.param)]


def _drift(sample, x):
    """dx/dt (N, V) of the compiled card at the homogeneous state x, nominal values, no noise."""
    card = sample.card
    theta = np.broadcast_to(np.nan_to_num(sample.nominal, nan=1.0), (N, len(sample.nominal)))
    state = np.broadcast_to(x, (N, card.n_states)).astype(np.float64)
    rhs = compile_card(card).rhs(NumpyOps())
    return np.asarray(rhs(state, np.zeros_like(state), theta, sample.gains, SC))


@pytest.mark.parametrize("name", ["identity", "logistic", "wong_wang"])
def test_transfer_values_and_slopes(name):
    for u in (-30.0, -3.0, -1e-5, 0.0, 1e-5, 0.7, 5.0, 30.0):
        value, slope = transfer(name, u)
        h = 1e-5
        numeric = (transfer(name, u + h)[0] - transfer(name, u - h)[0]) / (2 * h)
        assert slope == pytest.approx(numeric, rel=1e-5, abs=1e-9)
        expected = {"identity": u, "logistic": 1 / (1 + np.exp(-u)),
                    "wong_wang": 1.0 if u == 0 else u / -np.expm1(-u)}[name]
        assert value == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_most_models_are_anchored_at_a_stable_point_and_all_compile():
    status = [m.anchor for m in MODELS]
    assert set(status) <= {"stable", "unstable", "unsolved"}
    assert status.count("stable") > 0.5 * len(MODELS)       # the rest: mostly rates > RATE_LIMIT
    assert status.count("unsolved") < 0.05 * len(MODELS)
    for m in MODELS[:60]:
        compile_card(m.card)
    for m in MODELS:
        assert (m.rest is None) == (m.anchor == "unsolved")
        assert m.gains.shape == (len(m.card.global_params),) and (m.gains > 0).all()
        assert np.isfinite(m.gains).all()


def test_an_anchored_card_rests_at_its_resting_state():
    """End to end: the compiled card's drift vanishes at x*, every region alike."""
    checked = 0
    for m in MODELS[:150]:
        if m.rest is None:
            continue
        card = m.card
        L = np.array([[abs(_value(c, m)) for c in row] for row in card.linear])
        scale = L @ np.abs(m.rest) + np.array([abs(_value(b, m)) for b in card.bias]) + 1e-300
        assert (np.abs(_drift(m, m.rest)) <= 1e-5 * scale).all(), card.name
        checked += 1
    assert checked > 100


def test_the_resting_point_is_inside_its_priors_and_the_ratio_is_the_effective_loop():
    sites = set()
    for m in MODELS:
        if m.anchor == "unsolved":
            continue
        card, x = m.card, m.rest
        inputs = np.zeros(card.n_states)                     # resting input of direct channels
        for ch, G in zip(card.channels, m.gains):
            src = int(np.argmax(ch.source))
            for v in ch.into_input:
                inputs[v] += (not ch.diffusive) * G * SC_ROW_SUM * transfer(ch.transfer, x[src])[0]
        loop, g_of = {}, {}
        rate = [abs(_value(card.linear[v][v], m)) for v in range(card.n_states)]
        for v, t in enumerate(card.nonlinear):
            if t is None:
                continue
            assert any(c is not None for c in (*t.weights, t.drive))    # u^v can be placed
            g = {"kind": t.gate.kind, "state": t.gate.state, "weights": t.gate.weights}
            g_of[v] = g
            if t.gate.kind == "one_minus_self":
                assert REST_UNIT[0] <= x[v] <= REST_UNIT[1]
            u = sum(_value(w, m) * x[k] for k, w in enumerate(t.weights)) + _value(t.drive, m)
            u += inputs[v]
            if t.transfer == "identity":
                assert 10 ** INPUT_LOG10[0] * (1 - 1e-6) <= abs(u) <= 10 ** INPUT_LOG10[1] * (1 + 1e-6)
            else:
                lo, hi = INPUT_RANGE[t.transfer]
                assert lo - 1e-6 <= u <= hi + 1e-6
            phi, slope = transfer(t.transfer, u)
            loop[v] = abs(t.gain * gate(g, x, v) * slope)
            dg = {"one": 0.0, "one_minus_self": -1.0, "state": float(t.gate.state == v),
                  "quadratic": 2 * t.gate.weights[v] * x[v] if t.gate.weights else 0.0}[t.gate.kind]
            feedback = t.gain * (dg * phi + gate(g, x, v) * slope * _value(t.weights[v], m))
            rate[v] = abs(_value(card.linear[v][v], m) - feedback)
        for k, (ch, G) in enumerate(zip(card.channels, m.gains)):
            site, v = gain_site(card, card.global_params[k])
            sites.add(site)
            slope = transfer(ch.transfer, x[int(np.argmax(ch.source))])[1]
            through = loop[v] if site == "input" else 1.0
            ratio = through * G * SC_ROW_SUM * slope / rate[v]
            lo, hi = RATIO_LOG10 if ch.diffusive else DIRECT_RATIO_LOG10
            assert lo - 1e-6 <= np.log10(ratio) <= hi + 1e-6, (card.name, ratio)
            sign = np.sign(card.nonlinear[v].gain * gate(g_of[v], x, v)) if site == "input" else 1.0
            assert m.ratios[k] == pytest.approx(sign * ratio, rel=1e-6)   # what the sampler reports
    assert sites == {"input", "derivative"} and RATIO_LOG10 == COUPLING_RATIO_LOG10
    free = np.abs([m.rest[v] for m in MODELS if m.rest is not None
                   for v, t in enumerate(m.card.nonlinear)
                   if t is not None and t.gate.kind != "one_minus_self"])
    assert 10 ** REST_LOG10[0] <= free.min() and free.max() <= 10 ** REST_LOG10[1]


def test_stability_is_the_sign_of_the_local_jacobian():
    def draft(rate, gain=0.0):
        channel = {"source": 0, "transfer": "identity", "diffusive": False, "gain": gain,
                   "into_input": [], "into_derivative": [0]}
        return SimpleNamespace(V=1, L=np.array([[rate]]), bias=[None], nonlinear=[None],
                               channels=[channel])
    rest = np.zeros(1)
    assert is_stable(draft(2.0), rest, SC_ROW_SUM)
    assert not is_stable(draft(-2.0), rest, SC_ROW_SUM)
    assert not is_stable(draft(2 * RATE_LIMIT), rest, SC_ROW_SUM)     # too fast for the step
    # neighbours are clamped, so a direct channel does not change the region's own stability
    assert is_stable(draft(2.0, gain=100.0), rest, SC_ROW_SUM)


def test_anchoring_is_deterministic_and_leaves_the_other_priors_alone():
    a, b = sample_model(123, 0, 7, "sinh", FLOOR, "anchored"), MODELS[7]
    assert a.card == b.card and a.anchor == b.anchor
    np.testing.assert_array_equal(a.rest, b.rest)
    np.testing.assert_array_equal(a.gains, b.gains)
    np.testing.assert_array_equal(a.nominal, b.nominal)
    flat = sample_model(123, 0, 7, "sinh", FLOOR)
    assert flat.anchor == "none" and flat.rest is None and flat.gains is None
    assert flat.card == sample_card(123, 0, 7, "sinh", FLOOR)
    # a constant slot's value is in the card; a regional slot's nominal value is reported
    for m in MODELS[:40]:
        assert m.nominal.shape == (len(m.card.regional_params),)
        named = dict(zip(m.card.regional_params, m.nominal))
        assert all(np.isfinite(v) == (p != "sigma") for p, v in named.items())
        assert (m.nominal[rate_params(m.card)] >= FLOOR).all()


def test_nominal_levels_stay_in_their_window():
    seen = 0
    for m in MODELS[:80]:
        card = m.card
        if len(card.regional_params) < 2:
            continue
        z, kappa = draw_level_spread(card, np.random.default_rng(0), 400, FLOOR, "anchored",
                                     m.nominal)
        z_flat, kappa_flat = draw_level_spread(card, np.random.default_rng(0), 400, FLOOR,
                                               "anchored")
        np.testing.assert_array_equal(kappa, kappa_flat)
        level = SINH_SCALE * np.sinh(z)
        for j, p in enumerate(card.regional_params):
            if p == "sigma":
                np.testing.assert_array_equal(z[:, j], z_flat[:, j])        # the full range
                continue
            ratio = np.log10(level[:, j] / m.nominal[j])                     # same sign: > 0
            floor = np.log10(FLOOR / m.nominal[j]) if rate_params(card)[j] else -np.inf
            assert max(LEVEL_WINDOW_LOG10[0], floor) - 1e-6 <= ratio.min()
            assert ratio.max() <= LEVEL_WINDOW_LOG10[1] + 1e-6
            assert ratio.max() - ratio.min() > 0.5 or floor > LEVEL_WINDOW_LOG10[0]
            seen += 1
    assert seen > 50
    with pytest.raises(ValueError, match="nominal must have shape"):
        draw_level_spread(MODELS[0].card, np.random.default_rng(0), 2, FLOOR, "anchored",
                          np.ones(9))
