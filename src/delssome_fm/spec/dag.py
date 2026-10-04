"""The equation DAG the template encoder reads: node types, nodes, edges, feature encoding.

The DAG is emitted by compile.py (walk A). This module only defines its vocabulary and data
structure and turns nodes into the M x 21 feature matrix X_dag of docs/architecture.md:

    columns 1-15   one-hot node type (NODE_TYPES)
    column  16     sign(c) in {-1, 0, +1}                     Const only
    column  17     clip(log10|c|, -4, 4)                      Const only
    column  18     parameter scope: +1 regional, -1 global    Par only
    column  19     variable scope: +1 self, -1 neighbour      Var only
    columns 20-21  observation one-hot (Balloon-Windkessel, direct)   Out only

This layout was fixed by the project lead on 2026-10-05; it replaces architecture.md §1's
"1 parameter index, 1 scope tag" with parameter scope and variable scope. Which Par node is
which parameter is not a feature: it is recorded in `Dag.regional_par_nodes` and
`Dag.global_par_nodes`, which the encoder uses for the kappa_p and lambda_kappa readouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NODE_TYPES: tuple[str, ...] = (
    "Add",       # n-ary sum
    "Mul",       # n-ary product
    "Neg",       # unary negation; subtraction is Add(a, Neg(b))
    "Inv",       # reciprocal; division is Mul(a, Inv(b))
    "Exp",
    "Log",       # reserved: no template slot produces it yet
    "Sq",        # square
    "Agg",       # sum over neighbours j, one per coupling channel
    "Const",     # numeric constant
    "Par",       # free parameter, regional or global
    "Var",       # state variable, of this region (self) or of neighbour j
    "EdgeAttr",  # the connectivity weight C_ij; appears only under Agg
    "Noise",     # nu^v_i
    "Deriv",     # root: dx^v/dt, one per state variable
    "Out",       # root: the observable
)
N_FEATURES = len(NODE_TYPES) + 6
OBSERVATION_COLUMNS: tuple[str, ...] = ("balloon_windkessel", "direct")
LOG10_CLIP = 4.0

_UNARY = {"Neg", "Inv", "Exp", "Log", "Sq", "Agg", "Deriv", "Out"}
_NARY = {"Add", "Mul"}
_LEAVES = {"Const", "Par", "Var", "EdgeAttr", "Noise"}


@dataclass(frozen=True)
class Node:
    """One DAG node. Exactly the attributes that its type uses are set; the rest are None.

    value:        Const value
    param:        Par name;  scope: Par "regional" | "global"
    state:        Var / Noise / Deriv state-variable index;  neighbour: Var self vs neighbour
    channel:      Agg / EdgeAttr channel name
    observation:  Out observation model
    children:     indices of child nodes (operands), in canonical order
    """

    type: str
    children: tuple[int, ...] = ()
    value: float | None = None
    param: str | None = None
    scope: str | None = None
    state: int | None = None
    neighbour: bool | None = None
    channel: str | None = None
    observation: str | None = None

    def __post_init__(self) -> None:
        if self.type not in NODE_TYPES:
            raise ValueError(f"unknown node type {self.type!r}")
        n = len(self.children)
        if self.type in _LEAVES and n != 0:
            raise ValueError(f"{self.type} is a leaf, found {n} children")
        if self.type in _UNARY and n != 1:
            raise ValueError(f"{self.type} takes one child, found {n}")
        if self.type in _NARY and n < 2:
            raise ValueError(f"{self.type} takes at least two children, found {n}")


@dataclass(frozen=True)
class Dag:
    """
    nodes:              (M,)  topologically ordered: every child index < its parent's index
    deriv_roots:        (V,)  the Deriv node of each state variable
    out_root:                 the Out node
    agg_nodes:          one Agg node per coupling channel, in card order
    regional_par_nodes: (P,)  Par node of each regional parameter, in Theta column order
    global_par_nodes:   (K,)  Par node of each global parameter, in Psi order
    """

    nodes: tuple[Node, ...]
    deriv_roots: tuple[int, ...]
    out_root: int
    agg_nodes: tuple[int, ...]
    regional_par_nodes: tuple[int, ...]
    global_par_nodes: tuple[int, ...]

    def __post_init__(self) -> None:
        for m, node in enumerate(self.nodes):
            if any(c >= m for c in node.children):
                raise ValueError(f"node {m} ({node.type}) has a child that is not earlier")

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def edges(self) -> np.ndarray:
        """(E, 2) int array of (parent, child) pairs, in node then child order."""
        pairs = [(m, c) for m, node in enumerate(self.nodes) for c in node.children]
        return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def node_features(dag: Dag) -> np.ndarray:
    """
    returns: (M, 21) float32 X_dag, laid out as in the module docstring
    """
    X = np.zeros((dag.n_nodes, N_FEATURES), dtype=np.float32)
    base = len(NODE_TYPES)
    for m, node in enumerate(dag.nodes):
        X[m, NODE_TYPES.index(node.type)] = 1.0
        if node.type == "Const":
            X[m, base] = np.sign(node.value)
            if node.value != 0:
                X[m, base + 1] = np.clip(np.log10(abs(node.value)), -LOG10_CLIP, LOG10_CLIP)
        elif node.type == "Par":
            X[m, base + 2] = 1.0 if node.scope == "regional" else -1.0
        elif node.type == "Var":
            X[m, base + 3] = -1.0 if node.neighbour else 1.0
        elif node.type == "Out":
            X[m, base + 4 + OBSERVATION_COLUMNS.index(node.observation)] = 1.0
    return X


def depth_from_roots(dag: Dag) -> np.ndarray:
    """(M,) shortest number of edges from any Deriv root down to each node; -1 if unreachable.

    A diagnostic: architecture.md §2 quotes the Agg depth (2 for Linear and Hopf, deeper for
    models whose coupling sits under a transfer function).
    """
    depth = np.full(dag.n_nodes, -1, dtype=np.int64)
    frontier = list(dag.deriv_roots)
    for r in frontier:
        depth[r] = 0
    while frontier:
        nxt = []
        for m in frontier:
            for c in dag.nodes[m].children:
                if depth[c] < 0:
                    depth[c] = depth[m] + 1
                    nxt.append(c)
        frontier = nxt
    return depth
