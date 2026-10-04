"""Procedural sampling of synthetic model cards (generation.md §2, §4, §5), and of their
free-parameter values.

`sample_card(seed, index)` draws one ModelCard deterministically from (seed, index). It
applies, before any simulation:

    R1  dissipativity: min Re eig(L) > 0 with nominal values, unless some variable has a
        quadratic gate to supply saturation (resample L otherwise)
    R2  no identity transfer with gate 1 (resample the gate)
    R3  the first nonzero quadratic-gate weight is 1
    two channels must differ in source, delta or injection (resample the channel)
    a difference observable needs structurally distinguishable variables (else single)

and the free-parameter assignment of §4: candidates are the present L, beta, w and I slots,
n ~ U{1..4} of them become regional parameters; sigma is one shared regional parameter for
every noisy variable; each channel gain is a global parameter. Every coefficient, free or
not, gets a nominal constant c, log10|c| ~ U[-4, 4] with the slot's sign. A free slot is
stored as scale x parameter with scale = c, and `draw_parameters` gives the dimensionless
multipliers Theta, Psi ~ 10^U(-1, 1) (decided 2026-10-05).

Coupling gains are the one exception to the flat constant prior (decided 2026-10-05). Drawn
independently over 8 decades, the coupling was usually negligible next to the leak: in a
60-candidate pilot, 6 of 7 usable kept models had mean FC ~ 0, so their FC carried no SC.
A channel's nominal gain is instead relative to the term it competes with at its injection
site:

    G_c = 10^U(-2, 1) * reference / SC_ROW_SUM
    reference = |L_vv| of the target v           (injected at the derivative, D)
              = largest |w_vu| or |I_v| of u^v    (injected into u^v, U; 1 if neither)

so coupling ranges from 1% to 10x the competing term for a typical region.

Probabilities the design documents leave open are module constants below, marked "choice".
This module does not simulate or screen (sim/corpus.py does).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from delssome_fm.spec.card import (GATES, TRANSFERS, Channel, Coef, Gate, ModelCard, Nonlinear,
                                   Observable, Parameter)

# generation.md §2
V_CHOICES = ((1,), (2,), (3,), (4, 5, 6))
V_PROBS = (0.30, 0.45, 0.15, 0.10)
P_OFF_DIAGONAL = 0.5
P_BIAS = 0.5                     # choice: "present or absent"
P_NONLINEAR = 0.85
P_WEIGHT = 0.5
P_DRIVE = 0.5                    # choice: "present or absent"
QUADRATIC_SCHEMES = (("isotropic", 0.4), ("one_hot", 0.3), ("free", 0.3))
P_TWO_CHANNELS = 0.15
P_CHANNEL_IDENTITY = 0.85
P_DIFFUSIVE = 0.5                # choice
P_INJECT_INPUT = 0.5             # choice: U (into u^v) vs D (derivative), when u^v exists
P_NOISELESS = 0.1
P_DIFFERENCE_OBSERVABLE = 0.1
P_BALLOON_WINDKESSEL = 0.5       # choice: "independent of everything else"
P_LEAK = 0.8                     # choice: L diagonal positive (leak) vs negative (growth)
N_FREE = (1, 2, 3, 4)            # §4 step 3
LOG10_RANGE = (-4.0, 4.0)        # §4 constants
MULTIPLIER_LOG10 = (-1.0, 1.0)   # Theta, Psi ~ 10^U(-1, 1)
COUPLING_RATIO_LOG10 = (-2.0, 1.0)
# Mean row sum of the 64 HCP-YA training group SCs rescaled to max 0.02 (range over groups
# 0.356-0.365; tests/test_sampler.py checks it against the data on the cluster).
SC_ROW_SUM = 0.360
MAX_ATTEMPTS = 1000


class SamplerError(RuntimeError):
    """A rule could not be satisfied in MAX_ATTEMPTS draws: a sampler bug, never data."""


@dataclass
class _Draft:
    """Mutable working copy; a frozen ModelCard is built from it at the end."""

    V: int
    L: np.ndarray                                  # (V, V) nominal values, nan = absent
    bias: list = field(default_factory=list)       # nominal value or None
    nonlinear: list = field(default_factory=list)  # dict or None
    noise: list = field(default_factory=list)      # nominal scale or None
    channels: list = field(default_factory=list)   # dicts


def _constant(rng: np.random.Generator, sign: float | None = None) -> float:
    """log10|c| ~ U[-4, 4]; sign given, or +-1 with equal probability."""
    s = sign if sign is not None else rng.choice([-1.0, 1.0])
    return float(s * 10.0 ** rng.uniform(*LOG10_RANGE))


def sample_card(seed: int, index: int) -> ModelCard:
    """The `index`-th synthetic card of the corpus generated with `seed`."""
    rng = np.random.default_rng([seed, index])
    V = int(rng.choice(V_CHOICES[rng.choice(len(V_CHOICES), p=V_PROBS)]))
    d = _Draft(V=V, L=np.full((V, V), np.nan))
    d.nonlinear = [_sample_nonlinear(rng, v, V) if rng.random() < P_NONLINEAR else None
                   for v in range(V)]
    d.bias = [_constant(rng) if rng.random() < P_BIAS else None for _ in range(V)]
    d.noise = [None if rng.random() < P_NOISELESS else _constant(rng, 1.0) for _ in range(V)]
    _sample_channels(rng, d)
    _ensure_inputs(d)
    _sample_L(rng, d)
    _set_gains(rng, d)
    observable = _sample_observable(rng, d)
    return _assemble(rng, d, observable, name=f"syn_{seed}_{index}")


def _sample_nonlinear(rng: np.random.Generator, v: int, V: int) -> dict:
    transfer = TRANSFERS[rng.integers(len(TRANSFERS))]
    for _ in range(MAX_ATTEMPTS):
        kind = GATES[rng.integers(len(GATES))]
        if not (transfer == "identity" and kind == "one"):  # R2
            break
    else:
        raise SamplerError("could not satisfy R2")
    gate = {"kind": kind, "state": None, "weights": None}
    if kind == "state":
        gate["state"] = int(rng.integers(V))
    elif kind == "quadratic":
        gate["weights"] = _quadratic_weights(rng, V)
    return {"gain": _constant(rng), "gate": gate, "transfer": transfer,
            "weights": [_constant(rng) if rng.random() < P_WEIGHT else None for _ in range(V)],
            "drive": _constant(rng) if rng.random() < P_DRIVE else None}


def _quadratic_weights(rng: np.random.Generator, V: int) -> tuple[float, ...]:
    names, probs = zip(*QUADRATIC_SCHEMES)
    scheme = names[rng.choice(len(names), p=probs)]
    if scheme == "isotropic":
        w = np.ones(V)
    elif scheme == "one_hot":
        w = np.zeros(V)
        w[rng.integers(V)] = 1.0
    else:
        w = np.array([_constant(rng) for _ in range(V)])
    first = w[np.flatnonzero(w)[0]]
    return tuple(float(x) for x in w / first)  # R3


def _sample_channels(rng: np.random.Generator, d: _Draft) -> None:
    n = 2 if rng.random() < P_TWO_CHANNELS else 1
    signatures = set()
    for c in range(n):
        for _ in range(MAX_ATTEMPTS):
            ch = _one_channel(rng, d, c)
            sig = (ch["source"], ch["diffusive"], tuple(ch["into_input"]),
                   tuple(ch["into_derivative"]))
            if sig not in signatures:  # two channels must differ
                signatures.add(sig)
                d.channels.append(ch)
                break
        else:
            raise SamplerError("could not draw distinct channels")


def _one_channel(rng: np.random.Generator, d: _Draft, c: int) -> dict:
    target = int(rng.integers(d.V))
    into_input = d.nonlinear[target] is not None and rng.random() < P_INJECT_INPUT
    if rng.random() < P_CHANNEL_IDENTITY:
        transfer = "identity"
    else:
        transfer = ("logistic", "wong_wang")[rng.integers(2)]
    return {"name": f"c{c}", "source": int(rng.integers(d.V)), "transfer": transfer,
            "diffusive": bool(rng.random() < P_DIFFUSIVE), "gain": None,  # _set_gains
            "into_input": [target] if into_input else [],
            "into_derivative": [] if into_input else [target]}


def _ensure_inputs(d: _Draft) -> None:
    """A nonlinear term whose u^v has no weight, drive or channel would be the constant
    Phi(0), which for the identity transfer erases the term: give it the self weight."""
    for v, term in enumerate(d.nonlinear):
        if term is None:
            continue
        has_channel = any(v in ch["into_input"] for ch in d.channels)
        if all(w is None for w in term["weights"]) and term["drive"] is None and not has_channel:
            term["weights"][v] = 1.0


def _sample_L(rng: np.random.Generator, d: _Draft) -> None:
    quadratic = any(t is not None and t["gate"]["kind"] == "quadratic" for t in d.nonlinear)
    for _ in range(MAX_ATTEMPTS):
        L = np.full((d.V, d.V), np.nan)
        for v in range(d.V):
            L[v, v] = _constant(rng, 1.0 if rng.random() < P_LEAK else -1.0)
            for u in range(d.V):
                if u != v and rng.random() < P_OFF_DIAGONAL:
                    L[v, u] = _constant(rng)
        if quadratic or np.min(np.linalg.eigvals(np.nan_to_num(L)).real) > 0:  # R1
            d.L = L
            return
    raise SamplerError("could not satisfy R1")


def _set_gains(rng: np.random.Generator, d: _Draft) -> None:
    """Coupling gain relative to the competing term at the injection site (module docstring)."""
    for ch in d.channels:
        if ch["into_derivative"]:
            reference = abs(d.L[ch["into_derivative"][0], ch["into_derivative"][0]])
        else:
            t = d.nonlinear[ch["into_input"][0]]
            inputs = [abs(w) for w in t["weights"] if w is not None]
            inputs += [abs(t["drive"])] if t["drive"] is not None else []
            reference = max(inputs, default=1.0)
        ratio = 10.0 ** rng.uniform(*COUPLING_RATIO_LOG10)
        ch["gain"] = float(ratio * reference / SC_ROW_SUM)


def _sample_observable(rng: np.random.Generator, d: _Draft) -> tuple[int, int | None]:
    plus = int(rng.integers(d.V))
    if d.V > 1 and rng.random() < P_DIFFERENCE_OBSERVABLE:
        minus = int(rng.choice([u for u in range(d.V) if u != plus]))
        if _distinguishable(d, plus, minus):
            return plus, minus
    return plus, None


def _distinguishable(d: _Draft, a: int, b: int) -> bool:
    """generation.md §2: the difference of two variables that share a transfer function, a
    gate and the pattern of their row of L is near zero."""
    def signature(v):
        t = d.nonlinear[v]
        term = None if t is None else (t["transfer"], t["gate"]["kind"])
        return term, tuple(~np.isnan(d.L[v]))
    return signature(a) != signature(b)


def _assemble(rng: np.random.Generator, d: _Draft, observable: tuple[int, int | None],
              name: str) -> ModelCard:
    V = d.V
    candidates = [("L", v, u) for v in range(V) for u in range(V) if not np.isnan(d.L[v, u])]
    candidates += [("beta", v, None) for v in range(V) if d.bias[v] is not None]
    for v, t in enumerate(d.nonlinear):
        if t is not None:
            candidates += [("w", v, u) for u in range(V) if t["weights"][u] is not None]
            candidates += [("I", v, None)] if t["drive"] is not None else []
    n = min(int(rng.choice(N_FREE)), len(candidates))
    free = {candidates[i] for i in rng.choice(len(candidates), size=n, replace=False)}

    def coef(value, slot):
        if value is None:
            return None
        if slot in free:
            kind, v, u = slot
            return Coef(param=f"{kind}_{v}" + ("" if u is None else f"_{u}"), scale=value)
        return Coef(value=value)

    nonlinear = []
    for v, t in enumerate(d.nonlinear):
        if t is None:
            nonlinear.append(None)
            continue
        g = t["gate"]
        nonlinear.append(Nonlinear(
            gain=t["gain"], gate=Gate(g["kind"], g["state"], g["weights"]),
            transfer=t["transfer"],
            weights=tuple(coef(t["weights"][u], ("w", v, u)) for u in range(V)),
            drive=coef(t["drive"], ("I", v, None))))
    noisy = any(s is not None for s in d.noise)
    params = [Parameter(f"{k}_{v}" + ("" if u is None else f"_{u}"), "regional")
              for k, v, u in sorted(free, key=str)]
    params += [Parameter("sigma", "regional")] if noisy else []
    params += [Parameter(f"G_{ch['name']}", "global") for ch in d.channels]
    return ModelCard(
        name=name, states=tuple(f"x{v}" for v in range(V)),
        linear=tuple(tuple(coef(None if np.isnan(d.L[v, u]) else float(d.L[v, u]),
                                ("L", v, u)) for u in range(V)) for v in range(V)),
        bias=tuple(coef(d.bias[v], ("beta", v, None)) for v in range(V)),
        nonlinear=tuple(nonlinear),
        noise=tuple(None if s is None else Coef(param="sigma", scale=s) for s in d.noise),
        channels=tuple(Channel(
            name=ch["name"], source=tuple(1.0 if u == ch["source"] else 0.0 for u in range(V)),
            transfer=ch["transfer"], diffusive=ch["diffusive"],
            gain=Coef(param=f"G_{ch['name']}", scale=ch["gain"]),
            into_input=tuple(ch["into_input"]), into_derivative=tuple(ch["into_derivative"]))
            for ch in d.channels),
        observable=Observable(observable[0], observable[1],
                              "balloon_windkessel" if rng.random() < P_BALLOON_WINDKESSEL
                              else "direct"),
        parameters=tuple(params))


def draw_parameters(card: ModelCard, rng: np.random.Generator, n_draws: int,
                    n_regions: int) -> tuple[np.ndarray, np.ndarray]:
    """returns: theta (n_draws, N, P), psi (n_draws, K), multipliers 10^U(-1, 1)."""
    P, K = len(card.regional_params), len(card.global_params)
    theta = 10.0 ** rng.uniform(*MULTIPLIER_LOG10, size=(n_draws, n_regions, P))
    psi = 10.0 ** rng.uniform(*MULTIPLIER_LOG10, size=(n_draws, K))
    return theta, psi
