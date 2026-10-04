"""Compile a ModelCard into the equation DAG and the simulator's right-hand side, from one tree.

    ModelCard -> SymPy expressions -> canonicalise -+-> walk A: DAG nodes + edges (dag.py)
                                                    +-> walk B: program over the ops interface
    then: rebuild the expressions from the DAG and assert they equal the canonical ones.

This is the invariant of brief §2.1: the DAG the encoder reads and the integrator that runs
come from the same canonical SymPy trees, in the same `compile_card` call, and every compile
checks the round trip. A compile that fails raises CompileError; nothing here is optional.

Expressions. For each state variable v, the canonical expression is the full right-hand side
dx^v/dt including the noise term sigma_v * nu_v, with each coupling channel appearing as a
symbol A_c. The observable is a second kind of expression. Each channel additionally has an
edge expression G_c * C_ij * (f_c(p^c_j) - delta_c f_c(p^c_i)), p^c = sum_u m^c_u x^u, which
is what the Agg node sums over neighbours j.

Walk B evaluates the derivative trees with nu as an input. Feeding nu = xi / sqrt(dt), with xi
standard normal, makes `x + dt * rhs` exactly one Euler-Maruyama step; feeding nu = 0 gives
the deterministic drift. Channels are evaluated once per step outside the per-region trees,
as G_c * (C @ f_c(p^c) - delta_c * rowsum(C) * f_c(p^c)) (brief §6.2).

This module does not integrate (sim/integrate.py) and does not sample cards (sampler.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sympy as sp
from sympy.core.sorting import default_sort_key

from delssome_fm.spec.card import Channel, Coef, ModelCard, Nonlinear
from delssome_fm.spec.dag import Dag, Node
from delssome_fm.spec.ops import Array, Ops


class CompileError(RuntimeError):
    """A card could not be compiled, or the DAG round trip failed. Always a bug or a bad card."""


# ---------------------------------------------------------------- symbols


@dataclass(frozen=True)
class Symbols:
    """The SymPy symbols of one card. Internal names start with '__', which parameter names
    cannot (card.py requires them to start with a letter), so the two never collide."""

    state: tuple[sp.Symbol, ...]                 # x^v_i
    noise: tuple[sp.Symbol, ...]                 # nu^v_i
    params: dict[str, sp.Symbol]
    channel: dict[str, sp.Symbol]                # A_c, the per-region coupling input
    edge: dict[str, sp.Symbol]                   # C_ij, one symbol per channel
    neighbour: dict[tuple[str, int], sp.Symbol]  # x^u_j, per (channel, u) with m^c_u != 0

    @staticmethod
    def for_card(card: ModelCard) -> "Symbols":
        V = card.n_states
        return Symbols(
            state=tuple(sp.Symbol(f"__x{v}", real=True) for v in range(V)),
            noise=tuple(sp.Symbol(f"__nu{v}", real=True) for v in range(V)),
            params={p.name: sp.Symbol(p.name, real=True) for p in card.parameters},
            channel={c.name: sp.Symbol(f"__A_{c.name}", real=True) for c in card.channels},
            edge={c.name: sp.Symbol(f"__C_{c.name}", real=True) for c in card.channels},
            neighbour={(c.name, u): sp.Symbol(f"__xn_{c.name}_{u}", real=True)
                       for c in card.channels for u, m in enumerate(c.source) if m != 0},
        )


# ---------------------------------------------------------------- card -> sympy


def number(value: float) -> sp.Expr:
    """A constant as SymPy sees it: integral values become Integers so that 1.0 * x folds to
    x; everything else is a Float carrying the exact binary64 value."""
    v = float(value)
    if v.is_integer() and abs(v) < 2 ** 53:
        return sp.Integer(int(v))
    return sp.Float(v)


def _add(*xs: sp.Expr) -> sp.Expr:
    return sp.Add(*xs, evaluate=False)


def _mul(*xs: sp.Expr) -> sp.Expr:
    return sp.Mul(*xs, evaluate=False)


def _neg(x: sp.Expr) -> sp.Expr:
    return sp.Mul(sp.Integer(-1), x, evaluate=False)


def _inv(x: sp.Expr) -> sp.Expr:
    return sp.Pow(x, sp.Integer(-1), evaluate=False)


def _sq(x: sp.Expr) -> sp.Expr:
    return sp.Pow(x, sp.Integer(2), evaluate=False)


def _exp(x: sp.Expr) -> sp.Expr:
    return sp.exp(x, evaluate=False)


def transfer(name: str, u: sp.Expr) -> sp.Expr:
    """The canonical transfer functions of generation.md §3, written in DAG primitives:
    logistic(u) = Inv(1 + Exp(Neg u)),  wong_wang(u) = u * Inv(1 + Neg(Exp(Neg u)))."""
    if name == "identity":
        return u
    if name == "logistic":
        return _inv(_add(sp.Integer(1), _exp(_neg(u))))
    if name == "wong_wang":
        return _mul(u, _inv(_add(sp.Integer(1), _neg(_exp(_neg(u))))))
    raise CompileError(f"unknown transfer function {name!r}")


def _factors(c: Coef, sym: Symbols) -> list[sp.Expr]:
    """A coefficient as a list of multiplicative factors (constant, or scale and parameter)."""
    if c.param is None:
        return [number(c.value)]
    return [number(c.scale), sym.params[c.param]]


def _gate(term: Nonlinear, v: int, sym: Symbols) -> sp.Expr:
    g = term.gate
    if g.kind == "one":
        return sp.Integer(1)
    if g.kind == "one_minus_self":
        return _add(sp.Integer(1), _neg(sym.state[v]))
    if g.kind == "state":
        return sym.state[g.state]
    return _add(*[_mul(number(s), _sq(sym.state[u])) for u, s in enumerate(g.weights) if s != 0])


def channel_source(c: Channel, sym: Symbols, neighbour: bool) -> sp.Expr:
    """p^c = sum_u m^c_u x^u, of region i (neighbour=False) or of neighbour j."""
    terms = [_mul(number(m), sym.neighbour[(c.name, u)] if neighbour else sym.state[u])
             for u, m in enumerate(c.source) if m != 0]
    return terms[0] if len(terms) == 1 else _add(*terms)


def build_expressions(card: ModelCard, sym: Symbols) -> tuple[list, list, sp.Expr]:
    """The template of generation.md §1 written out in SymPy, unevaluated.

    Expressions are built with evaluate=False so that SymPy's automatic rewriting (which
    distributes numbers over sums and pulls constants out of exponentials) cannot change the
    template's structure; `canonicalize` then applies exactly the brief's normalisation.

    returns: (derivative expression per state variable,
              edge expression per channel,
              observable expression)
    """
    V = card.n_states
    x = sym.state
    into_input = {v: [sym.channel[c.name] for c in card.channels if v in c.into_input]
                  for v in range(V)}
    into_deriv = {v: [sym.channel[c.name] for c in card.channels if v in c.into_derivative]
                  for v in range(V)}
    derivs = []
    for v in range(V):
        terms = [_mul(sp.Integer(-1), *_factors(card.linear[v][u], sym), x[u])
                 for u in range(V) if card.linear[v][u] is not None]
        if card.bias[v] is not None:
            terms.append(_mul(*_factors(card.bias[v], sym)))
        term = card.nonlinear[v]
        if term is not None:
            u_terms = [_mul(*_factors(term.weights[u], sym), x[u])
                       for u in range(V) if term.weights[u] is not None]
            if term.drive is not None:
                u_terms.append(_mul(*_factors(term.drive, sym)))
            u_v = _add(*u_terms, *into_input[v])
            terms.append(_mul(number(term.gain), _gate(term, v, sym),
                              transfer(term.transfer, u_v)))
        terms.extend(into_deriv[v])
        if card.noise[v] is not None:
            terms.append(_mul(*_factors(card.noise[v], sym), sym.noise[v]))
        derivs.append(_add(*terms))
    edges = []
    for c in card.channels:
        f_nbr = transfer(c.transfer, channel_source(c, sym, neighbour=True))
        f_self = transfer(c.transfer, channel_source(c, sym, neighbour=False))
        diff = _add(f_nbr, _neg(f_self)) if c.diffusive else f_nbr
        edges.append(_mul(*_factors(c.gain, sym), sym.edge[c.name], diff))
    obs = card.observable
    observable = x[obs.plus] if obs.minus is None else _add(x[obs.plus], _neg(x[obs.minus]))
    return derivs, edges, observable


# ---------------------------------------------------------------- canonicalisation


def canonicalize(expr: sp.Expr) -> sp.Expr:
    """The normal form both walks traverse. Bottom-up, and nothing else:

    - constant folding: the numbers among the operands of one Add (Mul) are summed
      (multiplied) into a single constant; integral values become Integers
    - removal of zeros: x + 0 -> x, 0 * x -> 0, 1 * x -> x
    - term ordering: operands sorted by SymPy's default_sort_key
    - negation has one form, Mul(-1, X) with X a single expression (the DAG's Neg node);
      a product with coefficient -1 and several factors becomes Mul(-1, Mul(factors))

    Products are not distributed over sums and nested products are not flattened, so the
    template's structure (gamma * gate * Phi(u), the shared u inside Phi) survives. The
    function is idempotent; the round trip relies on that.
    """
    if expr.is_Number:
        return number(float(expr))
    if expr.is_Symbol:
        return expr
    args = [canonicalize(a) for a in expr.args]
    if expr.is_Add:
        return _canonical_add(args)
    if expr.is_Mul:
        return _canonical_mul(args)
    if expr.is_Pow:
        return sp.Pow(args[0], args[1], evaluate=False)
    if isinstance(expr, (sp.exp, sp.log)):
        return expr.func(args[0], evaluate=False)
    raise CompileError(f"cannot canonicalise {type(expr).__name__}: {expr}")


def _canonical_add(args: list[sp.Expr]) -> sp.Expr:
    flat = []
    for a in args:
        flat.extend(a.args if a.is_Add else [a])
    total = sum(float(a) for a in flat if a.is_Number)
    terms = [a for a in flat if not a.is_Number]
    if total != 0:
        terms.append(number(total))
    if not terms:
        return sp.Integer(0)
    if len(terms) == 1:
        return terms[0]
    return sp.Add(*sorted(terms, key=default_sort_key), evaluate=False)


def _canonical_mul(args: list[sp.Expr]) -> sp.Expr:
    coef = 1.0
    factors = []
    for a in args:
        if a.is_Number:
            coef *= float(a)
        elif _is_neg(a):  # pull a nested negation's sign into this product's coefficient
            coef = -coef
            factors.append(a.args[1])
        else:
            factors.append(a)
    if coef == 0:
        return sp.Integer(0)
    if not factors:
        return number(coef)
    factors.sort(key=default_sort_key)
    body = factors[0] if len(factors) == 1 else sp.Mul(*factors, evaluate=False)
    if coef == 1:
        return body
    if coef == -1:
        return sp.Mul(sp.Integer(-1), body, evaluate=False)
    return sp.Mul(number(coef), *factors, evaluate=False)


def _is_neg(e: sp.Expr) -> bool:
    return e.is_Mul and len(e.args) == 2 and e.args[0] == -1


# ---------------------------------------------------------------- walk A: DAG


class _DagWalk:
    """Walk A. Emits nodes in post-order, so children always precede parents.

    Identical subexpressions share one node (memoised on the SymPy expression), except
    constants: each occurrence of a constant gets its own Const node, so that two unrelated
    subtrees are not joined through a shared literal such as 1.
    """

    def __init__(self, card: ModelCard, sym: Symbols, edges: list[sp.Expr]) -> None:
        self.nodes: list[Node] = []
        self.memo: dict[sp.Expr, int] = {}
        self.sym = sym
        self.lookup: dict[sp.Symbol, Node | tuple] = {}
        for v, s in enumerate(sym.state):
            self.lookup[s] = Node("Var", state=v, neighbour=False)
        for v, s in enumerate(sym.noise):
            self.lookup[s] = Node("Noise", state=v)
        for p in card.parameters:
            self.lookup[sym.params[p.name]] = Node("Par", param=p.name, scope=p.scope)
        for c, e in zip(card.channels, edges):
            self.lookup[sym.edge[c.name]] = Node("EdgeAttr", channel=c.name)
            for u, m in enumerate(c.source):
                if m != 0:
                    self.lookup[sym.neighbour[(c.name, u)]] = Node(
                        "Var", state=u, neighbour=True, channel=c.name)
            self.lookup[sym.channel[c.name]] = ("Agg", c.name, e)

    def add(self, node: Node) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def walk(self, expr: sp.Expr) -> int:
        if expr.is_Number:
            return self.add(Node("Const", value=float(expr)))
        if expr in self.memo:
            return self.memo[expr]
        m = self._emit(expr)
        self.memo[expr] = m
        return m

    def _emit(self, expr: sp.Expr) -> int:
        if expr.is_Symbol:
            target = self.lookup.get(expr)
            if target is None:
                raise CompileError(f"unknown symbol {expr} in expression")
            if isinstance(target, tuple):  # a channel: Agg over its edge expression
                _, name, edge = target
                return self.add(Node("Agg", children=(self.walk(edge),), channel=name))
            return self.add(target)
        if expr.is_Add:
            return self.add(Node("Add", children=tuple(self.walk(a) for a in expr.args)))
        if expr.is_Mul:
            if _is_neg(expr):
                return self.add(Node("Neg", children=(self.walk(expr.args[1]),)))
            return self.add(Node("Mul", children=tuple(self.walk(a) for a in expr.args)))
        if expr.is_Pow:
            base, exp = expr.args
            if exp == -1:
                return self.add(Node("Inv", children=(self.walk(base),)))
            if exp == 2:
                return self.add(Node("Sq", children=(self.walk(base),)))
            if exp == -2:
                sq = self.add(Node("Sq", children=(self.walk(base),)))
                return self.add(Node("Inv", children=(sq,)))
            raise CompileError(f"power {exp} in {expr} has no DAG primitive (no Pow node)")
        if isinstance(expr, sp.exp):
            return self.add(Node("Exp", children=(self.walk(expr.args[0]),)))
        if isinstance(expr, sp.log):
            return self.add(Node("Log", children=(self.walk(expr.args[0]),)))
        raise CompileError(f"no DAG primitive for {type(expr).__name__}: {expr}")


def _walk_dag(card: ModelCard, sym: Symbols, derivs: list, edges: list,
              observable: sp.Expr) -> Dag:
    w = _DagWalk(card, sym, edges)
    roots = [w.add(Node("Deriv", children=(w.walk(e),), state=v)) for v, e in enumerate(derivs)]
    out = w.add(Node("Out", children=(w.walk(observable),),
                     observation=card.observable.observation))
    par_index = {}
    for m, node in enumerate(w.nodes):
        if node.type == "Par":
            par_index[node.param] = m
    missing = [p.name for p in card.parameters if p.name not in par_index]
    if missing:
        raise CompileError(f"parameters {missing} vanished during canonicalisation")
    agg = {node.channel: m for m, node in enumerate(w.nodes) if node.type == "Agg"}
    if set(agg) != {c.name for c in card.channels}:
        raise CompileError(f"channels {sorted({c.name for c in card.channels} - set(agg))} "
                           "do not appear in any equation")
    return Dag(nodes=tuple(w.nodes), deriv_roots=tuple(roots), out_root=out,
               agg_nodes=tuple(agg[c.name] for c in card.channels),
               regional_par_nodes=tuple(par_index[n] for n in card.regional_params),
               global_par_nodes=tuple(par_index[n] for n in card.global_params))


# ---------------------------------------------------------------- round trip


def dag_to_sympy(dag: Dag, sym: Symbols) -> tuple[list, dict[str, sp.Expr], sp.Expr]:
    """Rebuild (derivative expressions, edge expression per channel, observable) from a DAG."""
    vals: list[sp.Expr] = []
    edges: dict[str, sp.Expr] = {}
    for node in dag.nodes:
        ch = [vals[c] for c in node.children]
        t = node.type
        if t == "Const":
            e = number(node.value)
        elif t == "Par":
            e = sym.params[node.param]
        elif t == "Var":
            e = (sym.neighbour[(node.channel, node.state)] if node.neighbour
                 else sym.state[node.state])
        elif t == "Noise":
            e = sym.noise[node.state]
        elif t == "EdgeAttr":
            e = sym.edge[node.channel]
        elif t == "Agg":
            edges[node.channel] = ch[0]
            e = sym.channel[node.channel]
        elif t == "Add":
            e = _add(*ch)
        elif t == "Mul":
            e = _mul(*ch)
        elif t == "Neg":
            e = _neg(ch[0])
        elif t == "Inv":
            e = _inv(ch[0])
        elif t == "Sq":
            e = _sq(ch[0])
        elif t == "Exp":
            e = _exp(ch[0])
        elif t == "Log":
            e = sp.log(ch[0], evaluate=False)
        elif t in ("Deriv", "Out"):
            e = ch[0]
        else:
            raise CompileError(f"cannot rebuild node type {t}")
        vals.append(e)
    return [vals[r] for r in dag.deriv_roots], edges, vals[dag.out_root]


def _assert_round_trip(card: ModelCard, dag: Dag, sym: Symbols, derivs: list, edges: list,
                       observable: sp.Expr) -> None:
    r_derivs, r_edges, r_obs = dag_to_sympy(dag, sym)
    for v, (a, b) in enumerate(zip(r_derivs, derivs)):
        if a != b:
            raise CompileError(f"{card.name}: DAG round trip changed dx{v}/dt\n"
                               f"  canonical: {b}\n  rebuilt:   {a}")
    for c, e in zip(card.channels, edges):
        if r_edges.get(c.name) != e:
            raise CompileError(f"{card.name}: DAG round trip changed channel {c.name}\n"
                               f"  canonical: {e}\n  rebuilt:   {r_edges.get(c.name)}")
    if r_obs != observable:
        raise CompileError(f"{card.name}: DAG round trip changed the observable")


# ---------------------------------------------------------------- walk B: program


@dataclass(frozen=True)
class Instr:
    """One step of a straight-line program. `args` index earlier instructions.

    op:    const | state | noise | par_regional | par_global | channel |
           add | mul | neg | inv | sq | exp | log
    index: state / noise / parameter column / channel position, where relevant
    """

    op: str
    args: tuple[int, ...] = ()
    value: float | None = None
    index: int | None = None


@dataclass(frozen=True)
class Program:
    """
    instrs:          straight-line code, evaluated per region (arrays of shape (N,))
    derivs:          (V,) instruction index of dx^v/dt
    observable:      instruction index of o_i
    channel_source:  per channel, instruction index of f_c(p^c_i)
    channel_gain:    per channel, instruction index of G_c
    channel_diffusive: per channel, delta_c
    """

    instrs: tuple[Instr, ...]
    derivs: tuple[int, ...]
    observable: int
    channel_source: tuple[int, ...]
    channel_gain: tuple[int, ...]
    channel_diffusive: tuple[bool, ...]


class _ProgramWalk:
    """Walk B. Same traversal rules as walk A (post-order, sorted args, the same mapping of
    SymPy forms to primitives), emitting instructions instead of nodes."""

    def __init__(self, card: ModelCard, sym: Symbols) -> None:
        self.instrs: list[Instr] = []
        self.memo: dict[sp.Expr, int] = {}
        self.lookup: dict[sp.Symbol, Instr] = {}
        for v, s in enumerate(sym.state):
            self.lookup[s] = Instr("state", index=v)
        for v, s in enumerate(sym.noise):
            self.lookup[s] = Instr("noise", index=v)
        for p, name in enumerate(card.regional_params):
            self.lookup[sym.params[name]] = Instr("par_regional", index=p)
        for k, name in enumerate(card.global_params):
            self.lookup[sym.params[name]] = Instr("par_global", index=k)
        for i, c in enumerate(card.channels):
            self.lookup[sym.channel[c.name]] = Instr("channel", index=i)

    def add(self, ins: Instr) -> int:
        self.instrs.append(ins)
        return len(self.instrs) - 1

    def walk(self, expr: sp.Expr) -> int:
        if expr in self.memo:
            return self.memo[expr]
        m = self._emit(expr)
        self.memo[expr] = m
        return m

    def _emit(self, expr: sp.Expr) -> int:
        if expr.is_Number:
            return self.add(Instr("const", value=float(expr)))
        if expr.is_Symbol:
            if expr not in self.lookup:
                raise CompileError(f"symbol {expr} cannot appear in a per-region program")
            return self.add(self.lookup[expr])
        if expr.is_Add or expr.is_Mul:
            if _is_neg(expr):
                return self.add(Instr("neg", args=(self.walk(expr.args[1]),)))
            idx = [self.walk(a) for a in expr.args]
            op = "add" if expr.is_Add else "mul"
            acc = idx[0]
            for j in idx[1:]:
                acc = self.add(Instr(op, args=(acc, j)))
            return acc
        if expr.is_Pow:
            base, exp = expr.args
            if exp == -1:
                return self.add(Instr("inv", args=(self.walk(base),)))
            if exp == 2:
                return self.add(Instr("sq", args=(self.walk(base),)))
            if exp == -2:
                sq = self.add(Instr("sq", args=(self.walk(base),)))
                return self.add(Instr("inv", args=(sq,)))
            raise CompileError(f"power {exp} in {expr} has no primitive")
        if isinstance(expr, sp.exp):
            return self.add(Instr("exp", args=(self.walk(expr.args[0]),)))
        if isinstance(expr, sp.log):
            return self.add(Instr("log", args=(self.walk(expr.args[0]),)))
        raise CompileError(f"no primitive for {type(expr).__name__}: {expr}")


def _walk_program(card: ModelCard, sym: Symbols, derivs: list,
                  observable: sp.Expr) -> Program:
    w = _ProgramWalk(card, sym)
    d = tuple(w.walk(e) for e in derivs)
    obs = w.walk(observable)
    source = tuple(w.walk(canonicalize(transfer(c.transfer, channel_source(c, sym, False))))
                   for c in card.channels)
    gain = tuple(w.walk(canonicalize(_mul(*_factors(c.gain, sym)))) for c in card.channels)
    return Program(instrs=tuple(w.instrs), derivs=d, observable=obs, channel_source=source,
                   channel_gain=gain, channel_diffusive=tuple(c.diffusive for c in card.channels))


def run_program(prog: Program, ops: Ops, x: Array, nu: Array, theta: Array, psi: Array,
                channels: list[Array] | None, outputs: tuple[int, ...]) -> list[Array]:
    """Evaluate `outputs` of `prog`. Shapes: x, nu (N, V); theta (N, P); psi (K,);
    channels: list of (N,) or None when the outputs do not depend on channels."""
    needed = _closure(prog, outputs)
    vals: dict[int, Array] = {}
    for m in sorted(needed):
        ins = prog.instrs[m]
        op = ins.op
        if op == "const":
            v = ops.const(ins.value)
        elif op == "state":
            v = ops.column(x, ins.index)
        elif op == "noise":
            v = ops.column(nu, ins.index)
        elif op == "par_regional":
            v = ops.column(theta, ins.index)
        elif op == "par_global":
            v = psi[ins.index]
        elif op == "channel":
            if channels is None:
                raise CompileError("program reads a channel, but no channels were supplied")
            v = channels[ins.index]
        elif op == "add":
            v = ops.add(vals[ins.args[0]], vals[ins.args[1]])
        elif op == "mul":
            v = ops.mul(vals[ins.args[0]], vals[ins.args[1]])
        else:
            v = getattr(ops, op)(vals[ins.args[0]])
        vals[m] = v
    return [vals[i] for i in outputs]


def _closure(prog: Program, outputs: tuple[int, ...]) -> set[int]:
    """The instructions `outputs` depend on, so that e.g. the observable can be evaluated
    without supplying the channel inputs only the derivatives read."""
    needed, stack = set(), list(outputs)
    while stack:
        m = stack.pop()
        if m not in needed:
            needed.add(m)
            stack.extend(prog.instrs[m].args)
    return needed


# ---------------------------------------------------------------- the compiled model


@dataclass(frozen=True)
class CompiledModel:
    """Everything derived from one card. `derivs`, `edges` and `observable` are the canonical
    SymPy expressions both walks traversed."""

    card: ModelCard
    symbols: Symbols
    derivs: tuple[sp.Expr, ...]
    edges: tuple[sp.Expr, ...]
    observable: sp.Expr
    dag: Dag
    program: Program

    def rhs(self, ops: Ops) -> Callable[[Array, Array, Array, Array, Array], Array]:
        """dx/dt as a function of (x, nu, theta, psi, sc).

        x, nu: (N, V)   state and noise input (nu = 0 for the deterministic drift)
        theta: (N, P)   regional parameters
        psi:   (K,)     global parameters
        sc:    (N, N)   structural connectivity, C_ij
        returns: (N, V)
        """
        prog = self.program

        def rhs(x: Array, nu: Array, theta: Array, psi: Array, sc: Array) -> Array:
            channels = None
            if prog.channel_source:
                outs = run_program(prog, ops, x, nu, theta, psi, None,
                                   prog.channel_source + prog.channel_gain)
                n = len(prog.channel_source)
                channels = []
                for f, g, diffusive in zip(outs[:n], outs[n:], prog.channel_diffusive):
                    f = ops.broadcast(f, ops.column(x, 0))
                    a = ops.matvec(sc, f)
                    if diffusive:
                        a = ops.add(a, ops.neg(ops.mul(ops.rowsum(sc), f)))
                    channels.append(ops.mul(g, a))
            derivs = run_program(prog, ops, x, nu, theta, psi, channels, prog.derivs)
            like = ops.column(x, 0)
            return ops.stack([ops.broadcast(d, like) for d in derivs])

        return rhs

    def observe(self, ops: Ops) -> Callable[[Array], Array]:
        """The observable o_i as a function of the state x (N, V); returns (N,)."""
        prog = self.program

        def observe(x: Array) -> Array:
            (o,) = run_program(prog, ops, x, None, None, None, None, (prog.observable,))
            return ops.broadcast(o, ops.column(x, 0))

        return observe


def compile_card(card: ModelCard) -> CompiledModel:
    """Card -> canonical SymPy -> DAG (walk A) and program (walk B), round trip asserted."""
    sym = Symbols.for_card(card)
    raw_derivs, raw_edges, raw_obs = build_expressions(card, sym)
    derivs = [canonicalize(e) for e in raw_derivs]
    edges = [canonicalize(e) for e in raw_edges]
    observable = canonicalize(raw_obs)
    _check_noise_additive(card, sym, derivs)
    dag = _walk_dag(card, sym, derivs, edges, observable)
    program = _walk_program(card, sym, derivs, observable)
    _assert_round_trip(card, dag, sym, derivs, edges, observable)
    return CompiledModel(card=card, symbols=sym, derivs=tuple(derivs), edges=tuple(edges),
                         observable=observable, dag=dag, program=program)


def _check_noise_additive(card: ModelCard, sym: Symbols, derivs: list) -> None:
    """Euler-Maruyama via nu = xi / sqrt(dt) is exact only for additive noise."""
    state = set(sym.state) | set(sym.channel.values())
    for v, e in enumerate(derivs):
        for u, nu in enumerate(sym.noise):
            coef = sp.diff(e.doit(), nu)
            if u != v and coef != 0:
                raise CompileError(f"{card.name}: noise nu{u} enters dx{v}/dt")
            if coef.free_symbols & (state | set(sym.noise)):
                raise CompileError(f"{card.name}: noise in dx{v}/dt is not additive: {coef}")
