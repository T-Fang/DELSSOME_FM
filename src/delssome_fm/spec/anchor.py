"""The anchored prior (generation.md §14, pilot): draw a model's resting point first, then
solve the coefficients that put the model there.

Why. A channel into a transfer input u^v couples regions at the rate

    gamma_v g_v(x*) Phi_v'(u*_v) G f_c'(p*) x (SC row sum)

against the rate r_v at which x^v returns to rest, and g_v(x*), Phi_v'(u*_v) and r_v depend on
the state x* the model rests in and on the transfer input u*_v there. Coefficients drawn
independently leave them to chance, and in most draws the gate or the slope is near zero
(generation.md §13). Here x* and u* are drawn and the rest follows.

    r_v = | L_vv - gamma_v ( dg_v/dx^v Phi_v + g_v Phi_v' w_vv ) |      (`decay_rates`)

is the leak with the term's own feedback on its variable; r_v = |L_vv| without a term.

The resting point is that of one region whose neighbours all rest at x* too, with every
region's connectome row sum at its mean (`row_sum`): the noise-free template at nominal
values. `anchor_draft` changes the sampler's draft in place:

    1. x*_v    +-10^U(-2, 1); U(0.05, 0.95) under the gate 1 - x^v (the gate stays positive)
    2. gains of channels at the derivative   G = ratio r_v / (row_sum f_c'(p*))
    3. a variable with no nonlinear term     its bias is solved; without a bias its x*_v is
                                             solved from its (linear) equation instead
    4. a variable with a nonlinear term
         u*_v       logistic U(-3, 3), Wong-Wang U(-3, 5), identity +-10^U(-1, 1)
         weights    w_vu = c_vu / x*^u, c_vu the weight's contribution to u*_v, |c| = 10^U(-2, 1)
                    with the sign and uniform number of the flat prior (c itself if x*^u = 0)
         gamma_v    from the balance  0 = -sum_u L_vu x*^u + beta_v + gamma_v g_v Phi_v + D_v
         gains of channels into u^v
                    G = ratio r_v / (row_sum |gamma_v g_v Phi_v'| f_c'(p*))
         I_v        from  u*_v = sum_u w_vu x*^u + I_v + U_v ; without a drive the weights are
                    rescaled together to meet it
       (D_v, U_v: the resting input of direct channels at the derivative, into u^v)
    5. the resting point must be stable, and slow enough for the simulation step, for a
       region whose neighbours are clamped at x*: every eigenvalue of the finite-difference
       Jacobian has a real part in (-RATE_LIMIT, 0) and an imaginary part below RATE_LIMIT;
       otherwise 1-5 are repeated with a new x*, u*

Gains, gamma and the weights depend on each other (r_v needs gamma_v and w_vv; a direct
channel's resting input needs its gain), so 2 and 4 are iterated to a fixed point; an attempt
that does not converge counts as unsolved.

`ratio` is the coupling ratio the card stream already drew, log10 ~ U(-2, 1); for both
injection sites it is the effective loop at the resting point. A direct channel's ratio is
mapped to log10 ~ U(-2, 0) (`channel_ratio`): at a ratio of 1 the resting network loses
stability. Every other coefficient (L, beta of a variable with a term, gate weights, noise)
keeps its drawn value.

Status: "stable"; "unstable" (solved, but no stable point in ANCHOR_ATTEMPTS: the last
solution is kept); "unsolved" (no attempt could be solved: the draft keeps its drawn
coefficients, gains follow the loop rule at nominal values, and there is no resting state).
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np

REST_LOG10 = (-2.0, 1.0)             # |x*_v|
REST_UNIT = (0.05, 0.95)             # x*_v under the gate 1 - x^v
INPUT_RANGE = {"logistic": (-3.0, 3.0), "wong_wang": (-3.0, 5.0)}   # u*_v
INPUT_LOG10 = (-1.0, 1.0)            # |u*_v|, identity transfer
CONTRIBUTION_LOG10 = (-2.0, 1.0)     # |w_vu x*^u|
RATIO_LOG10 = (-2.0, 1.0)            # the card stream's coupling ratio (sampler)
DIRECT_RATIO_LOG10 = (-2.0, 0.0)     # the same uniform number, for a direct channel
RATE_LIMIT = 1000.0                  # 1/s: half of Euler's stability limit at dt = 1 ms
ANCHOR_ATTEMPTS = 200
GAIN_ITERATIONS = 50
STATUSES = ("none", "stable", "unstable", "unsolved")


def transfer(name: str, u: float) -> tuple[float, float]:
    """(Phi(u), Phi'(u)) of a canonical transfer function (generation.md §3)."""
    if name == "identity":
        return float(u), 1.0
    if name == "logistic":
        e = float(np.exp(-abs(u)))                     # no overflow for large |u|
        p = 1.0 / (1.0 + e) if u >= 0 else e / (1.0 + e)
        return p, p * (1.0 - p)
    if name == "wong_wang":
        a = abs(u)
        if a < 1e-4:                                   # series around 0
            return 1.0 + u / 2.0, 0.5 + u / 6.0
        e = float(np.exp(-a))
        value, slope = a / (1.0 - e), (1.0 - e - a * e) / (1.0 - e) ** 2     # at |u|
        # H(u) - H(-u) = u, so H(-a) = H(a) - a and H'(-a) = 1 - H'(a)
        return (value, slope) if u > 0 else (value - a, 1.0 - slope)
    raise ValueError(f"unknown transfer {name!r}")


def gate(g: dict, x: np.ndarray, v: int) -> float:
    """The value at state x of the gate dict of the sampler's draft."""
    if g["kind"] == "one":
        return 1.0
    if g["kind"] == "one_minus_self":
        return float(1.0 - x[v])
    if g["kind"] == "state":
        return float(x[g["state"]])
    if g["kind"] == "quadratic":
        return float(np.dot(g["weights"], x ** 2))
    raise ValueError(f"unknown gate {g['kind']!r}")


def local_rhs(d, x: np.ndarray, rest: np.ndarray, row_sum: float) -> np.ndarray:
    """dx/dt (V,) of one region at state x in the noise-free draft, its neighbours clamped at
    `rest` and its connectome row sum `row_sum`."""
    inputs = np.zeros(d.V)
    out = -np.nan_to_num(d.L) @ x
    for ch in d.channels:
        src = ch["source"]
        a = ch["gain"] * row_sum * (transfer(ch["transfer"], rest[src])[0]
                                    - ch["diffusive"] * transfer(ch["transfer"], x[src])[0])
        for v in ch["into_input"]:
            inputs[v] += a
        for v in ch["into_derivative"]:
            out[v] += a
    for v in range(d.V):
        if d.bias[v] is not None:
            out[v] += d.bias[v]
        t = d.nonlinear[v]
        if t is not None:
            u = sum(w * x[k] for k, w in enumerate(t["weights"]) if w is not None)
            u += (t["drive"] or 0.0) + inputs[v]
            out[v] += t["gain"] * gate(t["gate"], x, v) * transfer(t["transfer"], u)[0]
    return out


def channel_ratio(ch: dict) -> float:
    """The channel's effective loop at rest: its drawn ratio, mapped to DIRECT_RATIO_LOG10
    for a direct channel (module docstring)."""
    if ch["diffusive"]:
        return ch["ratio"]
    (lo, hi), (dlo, dhi) = RATIO_LOG10, DIRECT_RATIO_LOG10
    return float(10.0 ** (dlo + (np.log10(ch["ratio"]) - lo) * (dhi - dlo) / (hi - lo)))


def decay_rates(d, x: np.ndarray, u_star: dict[int, float]) -> np.ndarray:
    """r_v (V,) of the module docstring at state x; u_star: the transfer input of each
    variable with a term."""
    r = np.empty(d.V)
    for v, t in enumerate(d.nonlinear):
        r[v] = d.L[v, v]
        if t is not None:
            g, kind = gate(t["gate"], x, v), t["gate"]["kind"]
            dg = {"one": 0.0, "one_minus_self": -1.0,
                  "state": 1.0 if kind == "state" and t["gate"]["state"] == v else 0.0,
                  "quadratic": 2.0 * t["gate"]["weights"][v] * x[v] if kind == "quadratic"
                  else 0.0}[kind]
            phi, slope = transfer(t["transfer"], u_star[v])
            with np.errstate(over="ignore", invalid="ignore"):
                r[v] -= t["gain"] * (dg * phi + g * slope * (t["weights"][v] or 0.0))
    return np.abs(r)


def is_stable(d, rest: np.ndarray, row_sum: float) -> bool:
    """True if `rest` is a stable resting point of `local_rhs` that the simulation step
    resolves: every eigenvalue of its finite-difference Jacobian has a real part in
    (-RATE_LIMIT, 0) and an imaginary part below RATE_LIMIT."""
    J = np.empty((d.V, d.V))
    for k in range(d.V):
        h = 1e-6 * max(abs(rest[k]), 1e-3)
        e = np.zeros(d.V)
        e[k] = h
        J[:, k] = (local_rhs(d, rest + e, rest, row_sum)
                   - local_rhs(d, rest - e, rest, row_sum)) / (2 * h)
    if not np.isfinite(J).all():
        return False
    eig = np.linalg.eigvals(J)
    return bool(eig.real.max() < 0 and eig.real.min() > -RATE_LIMIT
                and np.abs(eig.imag).max() < RATE_LIMIT)


def anchor_draft(d, rng: np.random.Generator, row_sum: float,
                 contribution: Callable[[float], float]) -> tuple[np.ndarray | None, str]:
    """Anchor the draft in place (module docstring). d: the sampler's draft, whose channels
    carry the coupling `ratio` of the card stream. contribution(w): the flat-prior weight w
    moved to CONTRIBUTION_LOG10. Returns (x* (V,) or None, status)."""
    drawn = (copy.deepcopy(d.nonlinear), list(d.bias))
    kept = None
    for _ in range(ANCHOR_ATTEMPTS):
        d.nonlinear, d.bias = copy.deepcopy(drawn[0]), list(drawn[1])
        rest = _solve(d, rng, row_sum, contribution)
        if rest is None:
            continue
        if is_stable(d, rest, row_sum):
            return rest, "stable"
        kept = (copy.deepcopy(d.nonlinear), list(d.bias), [ch["gain"] for ch in d.channels], rest)
    if kept is not None:
        d.nonlinear, d.bias = kept[0], kept[1]
        for ch, g in zip(d.channels, kept[2]):
            ch["gain"] = g
        return kept[3], "unstable"
    d.nonlinear, d.bias = drawn
    for ch in d.channels:                              # the loop rule at nominal values
        v = (ch["into_derivative"] or ch["into_input"])[0]
        through = abs(d.nonlinear[v]["gain"]) if ch["into_input"] else 1.0
        ch["gain"] = float(channel_ratio(ch) * abs(d.L[v, v]) / (row_sum * through))
    return None, "unsolved"


def _solve(d, rng: np.random.Generator, row_sum: float,
           contribution: Callable[[float], float]) -> np.ndarray | None:
    """One attempt: draw x* and u*, solve the draft's coefficients; None if it cannot be
    solved (a zero gate, transfer or balance; a singular linear part; a non-finite value).
    Consumes the same random numbers whatever the outcome."""
    V, L = d.V, np.nan_to_num(d.L)
    sign = rng.choice([-1.0, 1.0], size=V)
    magnitude = 10.0 ** rng.uniform(*REST_LOG10, size=V)
    unit = rng.uniform(*REST_UNIT, size=V)
    u_unit, u_sign = rng.uniform(size=V), rng.choice([-1.0, 1.0], size=V)
    x = np.array([unit[v] if t is not None and t["gate"]["kind"] == "one_minus_self"
                  else sign[v] * magnitude[v] for v, t in enumerate(d.nonlinear)])

    def channel_input(v: int, site: str) -> float:
        """Resting input of the direct channels injected at `site` of variable v."""
        return sum(ch["gain"] * row_sum * transfer(ch["transfer"], x[ch["source"]])[0]
                   for ch in d.channels if v in ch[site] and not ch["diffusive"])

    def set_gain(ch: dict, rate: float, through: float) -> bool:
        slope = transfer(ch["transfer"], x[ch["source"]])[1]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            ch["gain"] = float(channel_ratio(ch) * rate / np.float64(row_sum * through * slope))
        return bool(np.isfinite(ch["gain"]) and ch["gain"] > 0)

    # variables with neither a term nor a bias rest where their linear equation puts them;
    # the gains of channels into them need only |L_vv|
    solved = [v for v in range(V) if d.nonlinear[v] is None and d.bias[v] is None]
    for ch in d.channels:                              # start of the gain iteration
        v = (ch["into_derivative"] or ch["into_input"])[0]
        if not set_gain(ch, abs(L[v, v]), abs(d.nonlinear[v]["gain"]) if ch["into_input"] else 1.0):
            return None
    for _ in range(2 if solved else 0):                # second pass: gains at the solved x*
        A, b = L[np.ix_(solved, solved)].copy(), np.zeros(len(solved))
        for i, v in enumerate(solved):
            b[i] = -sum(L[v, u] * x[u] for u in range(V) if u not in solved)
            for ch in d.channels:
                if v not in ch["into_derivative"] or ch["diffusive"]:
                    continue
                if ch["transfer"] == "identity" and ch["source"] in solved:
                    A[i, solved.index(ch["source"])] -= ch["gain"] * row_sum
                else:
                    b[i] += ch["gain"] * row_sum * transfer(ch["transfer"], x[ch["source"]])[0]
        if abs(np.linalg.det(A)) < 1e-12 * np.abs(A).max() ** len(solved):
            return None
        x[solved] = np.linalg.solve(A, b)
        for ch in d.channels:
            if ch["into_derivative"] and ch["into_derivative"][0] in solved:
                if not set_gain(ch, abs(L[ch["into_derivative"][0], ch["into_derivative"][0]]), 1.0):
                    return None
    if not np.isfinite(x).all():
        return None

    u_star = {}
    for v, t in enumerate(d.nonlinear):
        if t is None:
            continue
        if t["transfer"] == "identity":
            u_star[v] = u_sign[v] * 10.0 ** (INPUT_LOG10[0] + (INPUT_LOG10[1] - INPUT_LOG10[0])
                                             * u_unit[v])
        else:
            lo, hi = INPUT_RANGE[t["transfer"]]
            u_star[v] = lo + (hi - lo) * u_unit[v]
    drawn = (copy.deepcopy(d.nonlinear), list(d.bias))
    through = {}
    for _ in range(GAIN_ITERATIONS):
        d.nonlinear, d.bias = copy.deepcopy(drawn[0]), list(drawn[1])
        for v, t in enumerate(d.nonlinear):
            balance = float(L[v] @ x) - channel_input(v, "into_derivative")
            if t is None:
                if d.bias[v] is not None:
                    d.bias[v] = balance
                    if d.bias[v] == 0:
                        return None
                continue
            t["weights"] = [None if w is None else contribution(w) / (x[u] if x[u] != 0 else 1.0)
                            for u, w in enumerate(t["weights"])]
            g, (phi, slope) = gate(t["gate"], x, v), transfer(t["transfer"], u_star[v])
            balance -= d.bias[v] or 0.0
            if g == 0 or phi == 0 or balance == 0:
                return None
            t["gain"] = balance / (g * phi)
            through[v] = abs(t["gain"] * g * slope)
            wx = sum(w * x[u] for u, w in enumerate(t["weights"]) if w is not None)
            target = u_star[v] - channel_input(v, "into_input")
            if t["drive"] is not None:
                t["drive"] = target - wx
            elif wx != 0:
                t["weights"] = [None if w is None else w * target / wx for w in t["weights"]]
            else:
                return None
            values = [t["gain"], t["drive"], *t["weights"]]
            if not all(np.isfinite(c) and c != 0 for c in values if c is not None):
                return None
        rates = decay_rates(d, x, u_star)
        if not np.isfinite(rates).all():
            return None
        before = np.array([ch["gain"] for ch in d.channels])
        for ch in d.channels:
            v = (ch["into_derivative"] or ch["into_input"])[0]
            if v in solved:
                continue
            if not set_gain(ch, rates[v], through[v] if ch["into_input"] else 1.0):
                return None
        after = np.array([ch["gain"] for ch in d.channels])
        if np.allclose(after, before, rtol=1e-10, atol=0.0):
            break
    else:
        return None
    residual = local_rhs(d, x, x, row_sum)
    scale = np.abs(L) @ np.abs(x) + np.array([abs(b or 0.0) for b in d.bias]) + 1e-300
    return x if np.isfinite(residual).all() and (np.abs(residual) <= 1e-6 * scale).all() else None
