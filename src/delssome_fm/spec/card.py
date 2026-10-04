"""The model card: the one description of a circuit model, from which everything else derives.

A card fills the slots of the template in docs/generation.md §1. For state variable v of
region i:

    dx^v/dt = -sum_u L[v,u] x^u + beta_v + gamma_v g_v(x) Phi_v(u^v)
              + sum_{c : v in D_c} A_c + sigma_v nu^v
    u^v     = sum_u w[v,u] x^u + I_v + sum_{c : v in U_c} A_c
    A_c,i   = G_c sum_j C_ij ( f_c(x^{s_c}_j) - delta_c f_c(x^{s_c}_i) )

Every coefficient slot holds a `Coef`: a fixed constant, or `scale` times a named free
parameter. Several slots may name the same parameter, which is how hand-written cards tie
coefficients (Hopf's a and omega). A slot that is absent (None) produces no term.

This module defines the dataclasses, checks the template's structural rules when a card is
constructed, and converts cards to and from YAML without loss. It does not build equations
(compile.py) or sample cards (sampler.py).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TRANSFERS: tuple[str, ...] = ("identity", "logistic", "wong_wang")
GATES: tuple[str, ...] = ("one", "one_minus_self", "state", "quadratic")
OBSERVATIONS: tuple[str, ...] = ("balloon_windkessel", "direct")
SCOPES: tuple[str, ...] = ("regional", "global")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class CardError(ValueError):
    """A card violates the template. Raised on construction, never caught internally."""


@dataclass(frozen=True)
class Coef:
    """A fixed constant (`value`), or `scale` times the free parameter named `param`."""

    value: float | None = None
    param: str | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if (self.value is None) == (self.param is None):
            raise CardError(f"Coef needs exactly one of value or param, found {self}")
        if self.value is not None and (not math.isfinite(self.value) or self.value == 0):
            raise CardError(f"constant must be finite and nonzero (mask the slot instead), "
                            f"found {self.value}")
        if not math.isfinite(self.scale) or self.scale == 0:
            raise CardError(f"parameter scale must be finite and nonzero, found {self.scale}")


@dataclass(frozen=True)
class Gate:
    """g_v. `one`: 1. `one_minus_self`: 1 - x^v. `state`: x^u with u = `state`.
    `quadratic`: sum_u weights[u] (x^u)^2, first nonzero weight fixed to 1 (rule R3)."""

    kind: str
    state: int | None = None
    weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Nonlinear:
    """The optional term gamma_v g_v Phi_v(u^v), u^v = sum_u w[u] x^u + I_v + (U-channels)."""

    gain: float                         # gamma_v: always a fixed constant, never free
    gate: Gate
    transfer: str                       # Phi_v, one of TRANSFERS
    weights: tuple[Coef | None, ...]    # w[v, :], length V
    drive: Coef | None                  # I_v


@dataclass(frozen=True)
class Channel:
    """A coupling channel A_c, declared once and injected into input or derivative slots."""

    name: str
    source: int                         # s_c: the one-hot m^c, index of the transmitted state
    transfer: str                       # f_c, one of TRANSFERS
    diffusive: bool                     # delta_c: False = direct, True = diffusive
    gain: Coef                          # G_c: a global parameter (or a constant)
    into_input: tuple[int, ...]         # U_c: variables whose u^v receives A_c
    into_derivative: tuple[int, ...]    # D_c: variables whose dx^v/dt receives A_c


@dataclass(frozen=True)
class Observable:
    """o_i = x^plus (minus None) or x^plus - x^minus, seen through `observation`."""

    plus: int
    minus: int | None
    observation: str                    # one of OBSERVATIONS


@dataclass(frozen=True)
class Parameter:
    name: str
    scope: str                          # "regional" (a column of Theta) or "global" (of Psi)


@dataclass(frozen=True)
class ModelCard:
    """
    states:     (V,)    state-variable names, for humans only
    linear:     (V, V)  L, entering as -sum_u L[v,u] x^u; diagonal always present
    bias:       (V,)    beta_v
    nonlinear:  (V,)    the nonlinear term of each variable, or None
    noise:      (V,)    sigma_v, or None for a noiseless variable
    parameters: the free parameters; regional ones in order are the P columns of Theta,
                global ones in order are the K entries of Psi
    """

    name: str
    states: tuple[str, ...]
    linear: tuple[tuple[Coef | None, ...], ...]
    bias: tuple[Coef | None, ...]
    nonlinear: tuple[Nonlinear | None, ...]
    noise: tuple[Coef | None, ...]
    channels: tuple[Channel, ...]
    observable: Observable
    parameters: tuple[Parameter, ...]

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def regional_params(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters if p.scope == "regional")

    @property
    def global_params(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters if p.scope == "global")


# ---------------------------------------------------------------- validation


def _validate(card: ModelCard) -> None:
    V = len(card.states)
    where = f"card '{card.name}'"
    if not _NAME.match(card.name):
        raise CardError(f"{where}: name must match {_NAME.pattern}")
    if V < 1:
        raise CardError(f"{where}: needs at least one state variable")
    _check_lengths(card, V, where)
    for v in range(V):
        if card.linear[v][v] is None:
            raise CardError(f"{where}: L[{v},{v}] must be present (the diagonal is never masked)")
        if card.nonlinear[v] is not None:
            _check_nonlinear(card.nonlinear[v], V, f"{where} variable {v}")
    _check_channels(card, V, where)
    obs = card.observable
    if obs.observation not in OBSERVATIONS:
        raise CardError(f"{where}: observation must be one of {OBSERVATIONS}")
    if not 0 <= obs.plus < V or (obs.minus is not None and not 0 <= obs.minus < V):
        raise CardError(f"{where}: observable indexes a missing state variable")
    if obs.minus == obs.plus:
        raise CardError(f"{where}: a difference observable needs two different variables")
    _check_parameters(card, where)


def _check_lengths(card: ModelCard, V: int, where: str) -> None:
    if len(card.linear) != V or any(len(row) != V for row in card.linear):
        raise CardError(f"{where}: L must be {V}x{V}")
    for field in ("bias", "nonlinear", "noise"):
        if len(getattr(card, field)) != V:
            raise CardError(f"{where}: {field} must have {V} entries, "
                            f"found {len(getattr(card, field))}")


def _check_nonlinear(term: Nonlinear, V: int, where: str) -> None:
    if term.transfer not in TRANSFERS:
        raise CardError(f"{where}: transfer must be one of {TRANSFERS}, found {term.transfer!r}")
    if not math.isfinite(term.gain) or term.gain == 0:
        raise CardError(f"{where}: gain gamma must be finite and nonzero")
    if len(term.weights) != V:
        raise CardError(f"{where}: w must have {V} entries")
    gate = term.gate
    if gate.kind not in GATES:
        raise CardError(f"{where}: gate must be one of {GATES}, found {gate.kind!r}")
    if gate.kind == "state" and not (gate.state is not None and 0 <= gate.state < V):
        raise CardError(f"{where}: a 'state' gate needs a valid state index")
    if gate.kind != "state" and gate.state is not None:
        raise CardError(f"{where}: only a 'state' gate takes a state index")
    if gate.kind == "quadratic":
        w = gate.weights
        if w is None or len(w) != V or not any(w):
            raise CardError(f"{where}: a quadratic gate needs {V} weights, not all zero")
        if next(s for s in w if s != 0) != 1:
            raise CardError(f"{where}: first nonzero quadratic weight must be 1 (rule R3)")
    elif gate.weights is not None:
        raise CardError(f"{where}: only a quadratic gate takes weights")
    if term.transfer == "identity" and gate.kind == "one":
        raise CardError(f"{where}: identity transfer with gate 1 is affine; omit the term "
                        "and use L, beta or a derivative injection instead (rule R2)")


def _check_channels(card: ModelCard, V: int, where: str) -> None:
    names = [c.name for c in card.channels]
    if len(set(names)) != len(names):
        raise CardError(f"{where}: channel names must be unique, found {names}")
    signatures = set()
    for c in card.channels:
        cw = f"{where} channel '{c.name}'"
        if not _NAME.match(c.name):
            raise CardError(f"{cw}: name must match {_NAME.pattern}")
        if not 0 <= c.source < V:
            raise CardError(f"{cw}: source {c.source} is not a state variable")
        if c.transfer not in TRANSFERS:
            raise CardError(f"{cw}: transfer must be one of {TRANSFERS}")
        if not c.into_input and not c.into_derivative:
            raise CardError(f"{cw}: is not injected anywhere")
        for v in c.into_input + c.into_derivative:
            if not 0 <= v < V:
                raise CardError(f"{cw}: injection target {v} is not a state variable")
        for v in c.into_input:
            if card.nonlinear[v] is None:
                raise CardError(f"{cw}: injects into u^{v}, but variable {v} has no "
                                "nonlinear term")
        signature = (c.source, c.diffusive, tuple(sorted(c.into_input)),
                     tuple(sorted(c.into_derivative)))
        if signature in signatures:
            raise CardError(f"{cw}: duplicates another channel's source, delta and injection")
        signatures.add(signature)


def _iter_coefs(card: ModelCard):
    """(slot description, Coef, the scope the slot requires)."""
    V = len(card.states)
    for v in range(V):
        for u in range(V):
            yield f"L[{v},{u}]", card.linear[v][u], "regional"
        yield f"beta[{v}]", card.bias[v], "regional"
        yield f"sigma[{v}]", card.noise[v], "regional"
        term = card.nonlinear[v]
        if term is not None:
            for u in range(V):
                yield f"w[{v},{u}]", term.weights[u], "regional"
            yield f"I[{v}]", term.drive, "regional"
    for c in card.channels:
        yield f"G[{c.name}]", c.gain, "global"


def _check_parameters(card: ModelCard, where: str) -> None:
    scopes = {}
    for p in card.parameters:
        if not _NAME.match(p.name):
            raise CardError(f"{where}: parameter name {p.name!r} must match {_NAME.pattern}")
        if p.scope not in SCOPES:
            raise CardError(f"{where}: parameter {p.name} scope must be one of {SCOPES}")
        if p.name in scopes:
            raise CardError(f"{where}: parameter {p.name} declared twice")
        scopes[p.name] = p.scope
    used = set()
    for slot, coef, required in _iter_coefs(card):
        if coef is None or coef.param is None:
            continue
        if coef.param not in scopes:
            raise CardError(f"{where}: {slot} uses undeclared parameter {coef.param!r}")
        if scopes[coef.param] != required:
            raise CardError(f"{where}: {slot} needs a {required} parameter, but "
                            f"{coef.param!r} is {scopes[coef.param]}")
        used.add(coef.param)
    unused = [n for n in scopes if n not in used]
    if unused:
        raise CardError(f"{where}: declared parameters {unused} are not used by any slot")


# ---------------------------------------------------------------- YAML


def card_to_dict(card: ModelCard) -> dict[str, Any]:
    """Plain-data form of a card, the inverse of `card_from_dict`."""
    return {
        "name": card.name,
        "states": list(card.states),
        "parameters": [{"name": p.name, "scope": p.scope} for p in card.parameters],
        "linear": [[_coef_out(c) for c in row] for row in card.linear],
        "bias": [_coef_out(c) for c in card.bias],
        "nonlinear": [_nonlinear_out(t) for t in card.nonlinear],
        "noise": [_coef_out(c) for c in card.noise],
        "channels": [{
            "name": c.name, "source": c.source, "transfer": c.transfer,
            "diffusive": c.diffusive, "gain": _coef_out(c.gain),
            "into_input": list(c.into_input), "into_derivative": list(c.into_derivative),
        } for c in card.channels],
        "observable": {"plus": card.observable.plus, "minus": card.observable.minus,
                       "observation": card.observable.observation},
    }


def card_from_dict(raw: dict[str, Any]) -> ModelCard:
    """Build a card from plain data, raising CardError on any missing or unknown key."""
    _keys(raw, {"name", "states", "parameters", "linear", "bias", "nonlinear", "noise",
                "channels", "observable"}, "card")
    params = tuple(Parameter(**_keys(p, {"name", "scope"}, "parameter"))
                   for p in raw["parameters"])
    channels = []
    for c in raw["channels"]:
        _keys(c, {"name", "source", "transfer", "diffusive", "gain", "into_input",
                  "into_derivative"}, "channel")
        channels.append(Channel(
            name=c["name"], source=_int(c["source"], "channel source"), transfer=c["transfer"],
            diffusive=_bool(c["diffusive"], "channel diffusive"),
            gain=_coef_in(c["gain"], f"channel {c['name']} gain", allow_none=False),
            into_input=tuple(_int(v, "into_input") for v in c["into_input"]),
            into_derivative=tuple(_int(v, "into_derivative") for v in c["into_derivative"])))
    obs = _keys(raw["observable"], {"plus", "minus", "observation"}, "observable")
    return ModelCard(
        name=raw["name"],
        states=tuple(raw["states"]),
        linear=tuple(tuple(_coef_in(c, "linear") for c in row) for row in raw["linear"]),
        bias=tuple(_coef_in(c, "bias") for c in raw["bias"]),
        nonlinear=tuple(_nonlinear_in(t) for t in raw["nonlinear"]),
        noise=tuple(_coef_in(c, "noise") for c in raw["noise"]),
        channels=tuple(channels),
        observable=Observable(plus=_int(obs["plus"], "observable plus"),
                              minus=None if obs["minus"] is None
                              else _int(obs["minus"], "observable minus"),
                              observation=obs["observation"]),
        parameters=params,
    )


def card_to_yaml(card: ModelCard) -> str:
    return yaml.safe_dump(card_to_dict(card), sort_keys=False, default_flow_style=None)


def card_from_yaml(text: str) -> ModelCard:
    return card_from_dict(yaml.safe_load(text))


def load_card(path: Path) -> ModelCard:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"card file not found: {path}")
    return card_from_yaml(path.read_text())


def _coef_out(c: Coef | None) -> Any:
    if c is None:
        return None
    if c.param is None:
        return c.value
    return {"param": c.param} if c.scale == 1 else {"param": c.param, "scale": c.scale}


def _coef_in(raw: Any, where: str, allow_none: bool = True) -> Coef | None:
    if raw is None:
        if not allow_none:
            raise CardError(f"{where}: a coefficient is required here")
        return None
    if isinstance(raw, bool) or isinstance(raw, str):
        raise CardError(f"{where}: expected a number or {{param: ...}}, found {raw!r} "
                        "(note: YAML reads 1e-4 as a string; write 1.0e-4)")
    if isinstance(raw, (int, float)):
        return Coef(value=float(raw))
    _keys(raw, {"param", "scale"}, where, optional={"scale"})
    scale = raw.get("scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise CardError(f"{where}: scale must be a number, found {scale!r}")
    return Coef(param=raw["param"], scale=float(scale))


def _nonlinear_out(t: Nonlinear | None) -> Any:
    if t is None:
        return None
    gate: dict[str, Any] = {"kind": t.gate.kind}
    if t.gate.state is not None:
        gate["state"] = t.gate.state
    if t.gate.weights is not None:
        gate["weights"] = list(t.gate.weights)
    return {"gain": t.gain, "gate": gate, "transfer": t.transfer,
            "weights": [_coef_out(c) for c in t.weights], "drive": _coef_out(t.drive)}


def _nonlinear_in(raw: Any) -> Nonlinear | None:
    if raw is None:
        return None
    _keys(raw, {"gain", "gate", "transfer", "weights", "drive"}, "nonlinear")
    g = _keys(raw["gate"], {"kind", "state", "weights"}, "gate", optional={"state", "weights"})
    gain = raw["gain"]
    if isinstance(gain, bool) or not isinstance(gain, (int, float)):
        raise CardError(f"nonlinear gain must be a number, found {gain!r}")
    return Nonlinear(
        gain=float(gain),
        gate=Gate(kind=g["kind"],
                  state=None if g.get("state") is None else _int(g["state"], "gate state"),
                  weights=None if g.get("weights") is None
                  else tuple(float(s) for s in g["weights"])),
        transfer=raw["transfer"],
        weights=tuple(_coef_in(c, "w") for c in raw["weights"]),
        drive=_coef_in(raw["drive"], "drive"),
    )


def _keys(raw: Any, required: set[str], where: str, optional: set[str] = frozenset()) -> dict:
    if not isinstance(raw, dict):
        raise CardError(f"{where}: expected a mapping, found {raw!r}")
    missing = (required - optional) - set(raw)
    unknown = set(raw) - required
    if missing or unknown:
        raise CardError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return raw


def _int(raw: Any, where: str) -> int:
    if type(raw) is not int:
        raise CardError(f"{where}: expected an integer, found {raw!r}")
    return raw


def _bool(raw: Any, where: str) -> bool:
    if type(raw) is not bool:
        raise CardError(f"{where}: expected true or false, found {raw!r}")
    return raw
