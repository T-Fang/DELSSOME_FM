"""spec/: the five reference cards, the compiler's round trip, and walk B against independent
code paths.

Independent references used here:
- the published equations (DELSSOME SI S1, S7, S8; Wilson & Cowan 1972), written out by hand
  in NumPy with the published constants, not via the template or SymPy;
- SymPy's own lambdify of the canonical expressions, with the coupling evaluated as an
  explicit double sum over the Agg edge expression rather than as a matrix-vector product.
"""

import math

import numpy as np
import pytest
import sympy as sp

from delssome_fm.spec.card import (CardError, Channel, Coef, Gate, ModelCard, Nonlinear,
                                   Observable, Parameter, card_from_yaml, card_to_yaml)
from delssome_fm.spec.compile import (CompileError, _assert_round_trip, canonicalize,
                                      compile_card)
from delssome_fm.spec.dag import (NODE_TYPES, N_FEATURES, Dag, Node, depth_from_roots,
                                  node_features)
from delssome_fm.spec.ops import JaxOps, NumpyOps
from delssome_fm.spec.reference import REFERENCE_NAMES, load_reference

N = 7
RNG = np.random.default_rng(12345)


@pytest.fixture(scope="module", params=REFERENCE_NAMES)
def compiled(request):
    return compile_card(load_reference(request.param))


def _inputs(model, rng):
    V = model.card.n_states
    x = rng.uniform(0.05, 0.6, size=(N, V))
    nu = rng.normal(size=(N, V))
    sc = rng.uniform(0, 0.02, size=(N, N))
    np.fill_diagonal(sc, 0)
    return x, nu, sc


# ---------------------------------------------------------------- reference cards


def test_reference_cards_compile_with_expected_sizes(compiled):
    card = compiled.card
    expected = {"linear": (1, 2, 1), "mfm": (1, 3, 1), "fic": (2, 4, 1),
                "wilson_cowan": (2, 3, 1), "hopf": (2, 3, 1), "mpr": (2, 3, 1),
                "jansen_rit": (6, 2, 1)}[card.name]
    assert (card.n_states, len(card.regional_params), len(card.global_params)) == expected
    assert 10 <= compiled.dag.n_nodes <= 120


def test_reference_cards_yaml_round_trip(compiled):
    card = compiled.card
    assert card_from_yaml(card_to_yaml(card)) == card


def test_agg_depth_matches_architecture(compiled):
    """architecture.md §2: the coupling sits at depth 2 in Linear and Hopf and deeper (under a
    transfer function) in MFM, FIC and Wilson-Cowan."""
    depth = depth_from_roots(compiled.dag)
    agg = [int(depth[a]) for a in compiled.dag.agg_nodes]
    if compiled.card.name in ("linear", "hopf", "mpr", "jansen_rit"):  # injected at dx/dt
        assert set(agg) == {2}
    else:
        assert min(agg) >= 5


# ---------------------------------------------------------------- published equations


def _published_mfm(x, nu, theta, psi, sc):
    J, a, b, d, gam, tau = 0.2609, 270.0, 108.0, 0.154, 0.641, 0.1
    w, I_card, sigma = theta.T
    i_ext = I_card + b / a  # the card's drive is I - b/a
    S = x[:, 0]
    cur = w * J * S + psi[0] * J * (sc @ S) + i_ext
    H = (a * cur - b) / (1 - np.exp(-d * (a * cur - b)))
    return (-S / tau + gam * (1 - S) * H + sigma * nu[:, 0])[:, None]


def _published_fic(x, nu, theta, psi, sc):
    aE, aI, bE, bI, dE, dI = 310.0, 615.0, 125.0, 177.0, 0.16, 0.087
    tE, tI, gam, wII, WE, WI, I0, J = 0.1, 0.01, 0.641, 1.0, 1.0, 0.7, 0.382, 0.15
    wEE, wEI, sigma, wIE = theta.T
    SE, SI = x[:, 0], x[:, 1]
    IE = WE * I0 + wEE * J * SE + psi[0] * J * (sc @ SE) - wIE * SI
    II = WI * I0 + wEI * J * SE - wII * SI

    def H(cur, a, b, d):
        return (a * cur - b) / (1 - np.exp(-d * (a * cur - b)))

    dSE = -SE / tE + (1 - SE) * gam * H(IE, aE, bE, dE) + sigma * nu[:, 0]
    dSI = -SI / tI + H(II, aI, bI, dI) + sigma * nu[:, 1]
    return np.stack([dSE, dSI], axis=-1)


def _published_hopf(x, nu, theta, psi, sc):
    a, w, sigma = theta.T
    z = x[:, 0] + 1j * x[:, 1]
    coupling = psi[0] * (sc @ z - sc.sum(1) * z)
    dz = (a + 1j * w) * z - np.abs(z) ** 2 * z + coupling
    return np.stack([dz.real + sigma * nu[:, 0], dz.imag + sigma * nu[:, 1]], axis=-1)


def _published_linear(x, nu, theta, psi, sc):
    inv_tau, sigma = theta.T
    X = x[:, 0]
    return (-X * inv_tau + psi[0] * (sc @ X) + sigma * nu[:, 0])[:, None]


def _published_wilson_cowan(x, nu, theta, psi, sc):
    c2, c3, c4, ae, the, ai, thi, Q, te, ti = 12, 15, 3, 1.3, 4.0, 2.0, 3.7, 0.0, 0.01, 0.01
    c1, P_card, sigma = theta.T
    P = P_card + the  # the card's drive is P - theta_e
    exc, inh = x[:, 0], x[:, 1]

    def S(z, a, th):
        return 1 / (1 + np.exp(-a * (z - th)))

    dE = (-exc + (1 - exc) * S(c1 * exc - c2 * inh + P + psi[0] * (sc @ exc), ae, the)) / te
    dI = (-inh + (1 - inh) * S(c3 * exc - c4 * inh + Q, ai, thi)) / ti
    return np.stack([dE + sigma * nu[:, 0], dI + sigma * nu[:, 1]], axis=-1)


def _published_mpr(x, nu, theta, psi, sc):
    tau, delta = 0.02, 1.0
    eta, J, sigma = theta.T
    r, v = x[:, 0], x[:, 1]
    dr = (delta / (np.pi * tau) + 2 * r * v) / tau
    dv = (v ** 2 + eta + J * tau * r - (np.pi * tau * r) ** 2 + psi[0] * (sc @ r)) / tau
    return np.stack([dr, dv + sigma * nu[:, 1]], axis=-1)


def _published_jansen_rit(x, nu, theta, psi, sc):
    A, B, a, b, e0, v0, r, C = 3.25, 22.0, 100.0, 50.0, 2.5, 6.0, 0.56, 135.0
    C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
    p, sigma = theta.T
    y0, y1, y2, y3, y4, y5 = x.T

    def S(v):
        return 2 * e0 / (1 + np.exp(r * (v0 - v)))

    coupling = psi[0] * (sc @ (y1 - y2))
    return np.stack([
        y3, y4, y5,
        A * a * S(y1 - y2) - 2 * a * y3 - a ** 2 * y0,
        A * a * (p + C2 * S(C1 * y0) + coupling) - 2 * a * y4 - a ** 2 * y1
        + A * a * sigma * nu[:, 4],
        B * b * C4 * S(C3 * y0) - 2 * b * y5 - b ** 2 * y2,
    ], axis=-1)


PUBLISHED = {
    "mpr": (_published_mpr, lambda r: np.stack([r.uniform(-6, -4, N), r.uniform(10, 20, N),
                                                 r.uniform(0.01, 0.1, N)], 1), (0.01,)),
    "jansen_rit": (_published_jansen_rit,
                   lambda r: np.stack([r.uniform(120, 320, N), r.uniform(1, 5, N)], 1), (0.1,)),
    "mfm": (_published_mfm, lambda r: np.stack([r.uniform(0.5, 1.5, N), r.uniform(-0.1, 0.1, N),
                                                 r.uniform(0.001, 0.01, N)], 1), (0.5,)),
    "fic": (_published_fic, lambda r: np.stack([r.uniform(2, 6, N), r.uniform(0.5, 2, N),
                                                 r.uniform(0.001, 0.01, N),
                                                 r.uniform(1, 3, N)], 1), (2.0,)),
    "hopf": (_published_hopf, lambda r: np.stack([r.uniform(-0.1, 0.1, N),
                                                   r.uniform(0.2, 0.6, N),
                                                   r.uniform(0.01, 0.05, N)], 1), (0.3,)),
    "linear": (_published_linear, lambda r: np.stack([r.uniform(0.5, 2, N),
                                                       r.uniform(0.01, 0.1, N)], 1), (0.4,)),
    "wilson_cowan": (_published_wilson_cowan,
                     lambda r: np.stack([r.uniform(10, 20, N), r.uniform(-2, 2, N),
                                         r.uniform(0.01, 0.1, N)], 1), (1.5,)),
}


def test_rhs_matches_published_equations(compiled):
    reference, draw_theta, psi = PUBLISHED[compiled.card.name]
    rng = np.random.default_rng(7)
    x, nu, sc = _inputs(compiled, rng)
    theta, psi = draw_theta(rng), np.array(psi)
    got = compiled.rhs(NumpyOps())(x, nu, theta, psi, sc)
    np.testing.assert_allclose(got, reference(x, nu, theta, psi, sc), rtol=1e-11, atol=1e-11)


def test_rhs_matches_sympy_lambdify_with_explicit_neighbour_sum(compiled):
    """A different code path: SymPy evaluates the canonical expressions, and each channel is
    sum_j of its Agg edge expression, not walk B's matvec."""
    card, sym = compiled.card, compiled.symbols
    rng = np.random.default_rng(3)
    x, nu, sc = _inputs(compiled, rng)
    theta = rng.uniform(0.1, 1.0, size=(N, len(card.regional_params)))
    psi = rng.uniform(0.1, 1.0, size=len(card.global_params))
    params = [sym.params[n] for n in card.regional_params + card.global_params]
    channels = []
    for c, edge in zip(card.channels, compiled.edges):
        used = [u for u, m in enumerate(c.source) if m != 0]
        f = sp.lambdify([*sym.state, *[sym.neighbour[(c.name, u)] for u in used],
                         sym.edge[c.name], *params], edge, "numpy")
        a = np.zeros(N)
        for i in range(N):
            for j in range(N):
                a[i] += f(*x[i], *x[j, used], sc[i, j], *theta[i], *psi)
        channels.append(a)
    args = [*sym.state, *sym.noise, *params, *[sym.channel[c.name] for c in card.channels]]
    expected = np.stack([
        np.broadcast_to(sp.lambdify(args, e, "numpy")(*x.T, *nu.T, *theta.T, *psi, *channels),
                        (N,)) for e in compiled.derivs], axis=-1)
    got = compiled.rhs(NumpyOps())(x, nu, theta, psi, sc)
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)


def test_observable(compiled):
    obs = compiled.card.observable
    x = np.random.default_rng(1).normal(size=(N, compiled.card.n_states))
    expected = x[:, obs.plus] - (0 if obs.minus is None else x[:, obs.minus])
    np.testing.assert_allclose(compiled.observe(NumpyOps())(x), expected, rtol=1e-15)


def test_jax_backend_matches_numpy_under_jit(compiled):
    import jax

    rng = np.random.default_rng(5)
    x, nu, sc = _inputs(compiled, rng)
    theta = rng.uniform(0.1, 1.0, size=(N, len(compiled.card.regional_params)))
    psi = rng.uniform(0.1, 1.0, size=len(compiled.card.global_params))
    ref = compiled.rhs(NumpyOps())(x, nu, theta, psi, sc)
    f = jax.jit(compiled.rhs(JaxOps()))
    got = np.asarray(f(*(a.astype(np.float32) for a in (x, nu, theta, psi, sc))))
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=1e-4 * np.abs(ref).max())


# ---------------------------------------------------------------- DAG features


def test_node_features_layout(compiled):
    dag = compiled.dag
    X = node_features(dag)
    assert X.shape == (dag.n_nodes, N_FEATURES) == (dag.n_nodes, 21)
    assert np.all(X[:, :15].sum(1) == 1)
    types = [n.type for n in dag.nodes]
    for m, (node, row) in enumerate(zip(dag.nodes, X)):
        extra = row[15:]
        if node.type == "Const":
            assert extra[0] == np.sign(node.value)
            assert extra[1] == pytest.approx(np.clip(np.log10(abs(node.value)), -4, 4))
            assert not extra[2:].any()
        elif node.type == "Par":
            assert extra[2] == (1 if node.scope == "regional" else -1) and extra[[0, 1, 3, 4, 5]].sum() == 0
        elif node.type == "Var":
            assert extra[3] == (-1 if node.neighbour else 1)
        elif node.type == "Out":
            assert extra[4:].sum() == 1
        else:
            assert not extra.any()
    assert types.count("Deriv") == compiled.card.n_states and types.count("Out") == 1
    assert types.count("Agg") == len(compiled.card.channels)
    # every EdgeAttr and neighbour Var sits under an Agg
    under_agg = set()
    for a in dag.agg_nodes:
        stack = [a]
        while stack:
            m = stack.pop()
            under_agg.add(m)
            stack.extend(dag.nodes[m].children)
    for m, node in enumerate(dag.nodes):
        if node.type == "EdgeAttr" or (node.type == "Var" and node.neighbour):
            assert m in under_agg


def test_par_nodes_follow_theta_and_psi_order(compiled):
    dag, card = compiled.dag, compiled.card
    assert [dag.nodes[m].param for m in dag.regional_par_nodes] == list(card.regional_params)
    assert [dag.nodes[m].param for m in dag.global_par_nodes] == list(card.global_params)


def test_no_zero_constants_reach_the_dag(compiled):
    assert all(n.value != 0 for n in compiled.dag.nodes if n.type == "Const")


def test_constant_feature_by_hand():
    dag = Dag(nodes=(Node("Const", value=-250.0), Node("Const", value=3e-7),
                     Node("Add", children=(0, 1)), Node("Deriv", children=(2,), state=0),
                     Node("Var", state=0, neighbour=False),
                     Node("Out", children=(4,), observation="direct")),
              deriv_roots=(3,), out_root=5, agg_nodes=(), regional_par_nodes=(),
              global_par_nodes=())
    X = node_features(dag)
    assert X[0, 15:17].tolist() == [-1.0, pytest.approx(math.log10(250))]
    assert X[1, 15:17].tolist() == [1.0, -4.0]  # log10(3e-7) = -6.5, clipped
    assert X[5, 19:21].tolist() == [0.0, 1.0]
    assert NODE_TYPES.index("Out") == 14 and X[5, 14] == 1


# ---------------------------------------------------------------- round trip and canonical form


def test_canonicalize_is_idempotent(compiled):
    for e in (*compiled.derivs, *compiled.edges, compiled.observable):
        assert canonicalize(e) == e


def test_round_trip_detects_a_tampered_dag():
    m = compile_card(load_reference("mfm"))
    nodes = list(m.dag.nodes)
    k = next(i for i, n in enumerate(nodes) if n.type == "Const")
    nodes[k] = Node("Const", value=nodes[k].value * 1.01)
    bad = Dag(nodes=tuple(nodes), deriv_roots=m.dag.deriv_roots, out_root=m.dag.out_root,
              agg_nodes=m.dag.agg_nodes, regional_par_nodes=m.dag.regional_par_nodes,
              global_par_nodes=m.dag.global_par_nodes)
    with pytest.raises(CompileError, match="round trip"):
        _assert_round_trip(m.card, bad, m.symbols, list(m.derivs), list(m.edges), m.observable)


def test_wong_wang_shares_its_input_node():
    """u appears twice in u / (1 - exp(-u)); the DAG must hold it once."""
    m = compile_card(load_reference("mfm"))
    adds = [n for n in m.dag.nodes if n.type == "Add"]
    u_nodes = [i for i, n in enumerate(m.dag.nodes)
               if n.type == "Add" and any(m.dag.nodes[c].type == "Agg" for c in n.children)]
    assert len(u_nodes) == 1 and adds
    parents = [i for i, n in enumerate(m.dag.nodes) if u_nodes[0] in n.children]
    assert len(parents) == 2  # the WW numerator product and the Neg inside exp


# ---------------------------------------------------------------- card validation


def _one_var(**kw):
    base = dict(name="t", states=("x",), linear=((Coef(value=1.0),),), bias=(None,),
                nonlinear=(Nonlinear(gain=1.0, gate=Gate("one_minus_self"),
                                     transfer="logistic", weights=(Coef(param="w"),),
                                     drive=None),),
                noise=(Coef(param="s"),), channels=(),
                observable=Observable(0, None, "direct"),
                parameters=(Parameter("w", "regional"), Parameter("s", "regional")))
    base.update(kw)
    return ModelCard(**base)


def test_valid_minimal_card_compiles():
    compile_card(_one_var())


@pytest.mark.parametrize("change,match", [
    (dict(nonlinear=(Nonlinear(1.0, Gate("one"), "identity", (Coef(param="w"),), None),)),
     "R2"),
    (dict(nonlinear=(Nonlinear(1.0, Gate("quadratic", weights=(2.0,)), "identity",
                               (Coef(param="w"),), None),)), "R3"),
    (dict(parameters=(Parameter("w", "regional"), Parameter("s", "regional"),
                      Parameter("z", "regional"))), "not used"),
    (dict(parameters=(Parameter("w", "global"), Parameter("s", "regional"))), "regional"),
    (dict(observable=Observable(0, 0, "direct")), "two different"),
    (dict(channels=(Channel("c", (1.0,), "identity", False, Coef(param="s"), (0,), ()),)),
     "global"),
    (dict(channels=(Channel("c", (0.0,), "identity", False, Coef(value=1.0), (0,), ()),)),
     "not all zero"),
])
def test_card_rules_raise(change, match):
    with pytest.raises(CardError, match=match):
        _one_var(**change)


def test_card_yaml_rejects_unknown_keys_and_string_numbers():
    text = card_to_yaml(load_reference("linear"))
    with pytest.raises(CardError, match="unknown keys"):
        card_from_yaml(text.replace("name: linear", "name: linear\nextra: 1"))
    with pytest.raises(CardError, match="1.0e-4"):
        card_from_yaml(text.replace("bias: [null]", "bias: [1e-4]"))


# ---------------------------------------------------------------- Wong-Wang evaluation


@pytest.mark.parametrize("ops", [NumpyOps(), JaxOps()], ids=["numpy", "jax-float32"])
def test_wong_wang_kernel_is_finite_and_accurate_near_zero(ops):
    u = np.array([0.0, 3e-8, -3e-8, 1e-4, -1e-4, 0.5, -0.5, 5.0, -5.0, 80.0, -80.0])
    got = np.asarray(ops.wong_wang(np.asarray(u, dtype=ops.dtype)), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = np.where(u == 0, 1.0, u / -np.expm1(-u))  # float64, limit 1 at u = 0
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, ref, rtol=2e-6, atol=1e-30)


def test_walk_b_fuses_wong_wang_but_the_dag_keeps_primitives():
    for name, n_ww in (("mfm", 1), ("fic", 2), ("hopf", 0), ("wilson_cowan", 0)):
        m = compile_card(load_reference(name))
        assert sum(i.op == "wong_wang" for i in m.program.instrs) == n_ww
        assert all(n.type in NODE_TYPES for n in m.dag.nodes)


def test_fic_rhs_finite_where_the_current_crosses_threshold():
    """u_E = 0 exactly: the literal subtree is 0/0, the fused kernel gives the limit."""
    m = compile_card(load_reference("fic"))
    # choose S_I so that u_E = 7.44 wEE S_E - 49.6 wIE S_I - 1.0528 = 0 with no coupling
    wEE, wIE, SE = 3.0, 1.5, 0.2
    SI = (7.44 * wEE * SE - 1.0528) / (49.6 * wIE)
    x = np.array([[SE, SI]])
    out = m.rhs(NumpyOps())(x, np.zeros((1, 2)), np.array([[wEE, 2.0, 0.0, wIE]]),
                            np.array([0.0]), np.zeros((1, 1)))
    expected_E = -10 * SE + 4.00625 * (1 - SE) * 1.0
    assert np.all(np.isfinite(out)) and out[0, 0] == pytest.approx(expected_E, rel=1e-6)
