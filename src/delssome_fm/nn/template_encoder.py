"""The template encoder (architecture.md §2): a Graphormer over the equation DAG, and the
readouts that tag parameters and summarise the model.

A Graphormer is a transformer over the DAG's nodes in which
    - each node's input is its 21 features plus embeddings of its in- and out-degree, and
    - attention between two nodes gets a learned bias per head for their shortest-path length
      (spec/dag.shortest_paths, edge direction ignored), so the encoder knows where in the
      equations two terms sit relative to each other.

Readouts, all in the backbone width d (the encoder output is projected first):

    kappa   (B, P, d)  the Par node of each regional parameter, in Theta column order
    lam     (B, K, d)  the Par node of each global parameter, in Psi order
    u       (B, d)     mean over the V Deriv roots
    o       (B, d)     the Out node
    m_agg   (B, d)     mean over the Agg nodes; 0 for a model without coupling
    z       (B, d)     mean over all nodes

P, K, V and the number of Agg nodes vary by model, so the index tensors are padded with -1
and every readout carries a mask.

With `node_values` (the parameter-token backbone, architecture.md §3.4, 2026-10-29) the
encoder also reads values: there is no constant/parameter distinction, only global vs
regional. Each Const node is recoded as a global Par node (`constants_as_global_parameters`),
and every global Par node (constants and the model's Psi) adds MLP_val(its encoded value) to
its input. Regional Par nodes carry no value: theirs differ by region and enter through the
parameter tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from delssome_fm.nn.layers import MLP, TransformerLayer, key_padding_bias, masked_mean
from delssome_fm.spec.dag import N_FEATURES, NODE_TYPES

CONST, PAR = NODE_TYPES.index("Const"), NODE_TYPES.index("Par")
SIGN, LOG10, SCOPE = len(NODE_TYPES), len(NODE_TYPES) + 1, len(NODE_TYPES) + 2   # dag.py columns


@dataclass
class GraphBatch:
    """A batch of equation DAGs, padded to the largest M in the batch.

    features: (B, M, 21) float    degree: (B, M, 2) integer, in/out
    distances: (B, M, M) integer, -1 where there is no path (incl. padding)
    node_mask: (B, M) bool, True for real nodes
    regional_par (B, P_MAX), global_par (B, K_MAX), deriv_roots (B, V_MAX), agg (B, C_MAX):
        node indices, -1 padded;  out_root: (B,) node index
    const_value: (B, M) float, each Const node's value encoded as the parameters are
        (train/data.encode_parameters), 0 elsewhere; None when the store has no Const values
        (stores packed before 2026-10-29, multiplier-policy stores). Read only by the
        parameter-token backbone.
    """

    features: torch.Tensor
    degree: torch.Tensor
    distances: torch.Tensor
    node_mask: torch.Tensor
    regional_par: torch.Tensor
    global_par: torch.Tensor
    deriv_roots: torch.Tensor
    agg: torch.Tensor
    out_root: torch.Tensor
    const_value: torch.Tensor | None = None


@dataclass
class Readouts:
    """architecture.md §2 readouts; masks are True where an entry exists."""

    kappa: torch.Tensor
    kappa_mask: torch.Tensor
    lam: torch.Tensor
    lam_mask: torch.Tensor
    u: torch.Tensor
    o: torch.Tensor
    m_agg: torch.Tensor
    z: torch.Tensor


class TemplateEncoder(nn.Module):
    """Graphormer over the equation DAG: GraphBatch -> H_tmpl (B, M, d_t)."""

    def __init__(self, d_t: int, n_layers: int, n_heads: int, ffn_multiplier: int,
                 dropout: float, max_distance: int, max_degree: int, node_values: bool = False):
        """node_values: read the values of global nodes (module docstring)."""
        super().__init__()
        self.max_distance = max_distance
        self.max_degree = max_degree
        self.input = nn.Linear(N_FEATURES, d_t)
        self.in_degree = nn.Embedding(max_degree + 1, d_t)
        self.out_degree = nn.Embedding(max_degree + 1, d_t)
        # one bias per head for each distance 0..max_distance, plus one for "no path"
        self.distance_bias = nn.Embedding(max_distance + 2, n_heads)
        self.layers = nn.ModuleList(TransformerLayer(d_t, n_heads, ffn_multiplier, dropout)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_t)
        self.value = MLP([1, d_t, d_t]) if node_values else None

    def forward(self, g: GraphBatch, values: torch.Tensor | None = None) -> torch.Tensor:
        """values (B, M): with node_values, the encoded value of every global node (Const
        and global Par; `global_values`), ignored elsewhere."""
        degree = g.degree.long().clamp(max=self.max_degree)
        features = g.features
        if self.value is not None:
            if values is None:
                raise ValueError("this encoder reads node values: pass global_values(...)")
            features = constants_as_global_parameters(features)
        x = self.input(features) + self.in_degree(degree[..., 0]) + self.out_degree(degree[..., 1])
        if self.value is not None:
            is_global = (features[..., PAR] == 1) & (features[..., SCOPE] == -1)
            x = x + self.value(values[..., None].to(x.dtype)) * is_global[..., None].to(x.dtype)
        distances = g.distances.long()
        no_path = torch.full_like(distances, self.max_distance + 1)
        bucket = torch.where(distances < 0, no_path, distances.clamp(max=self.max_distance))
        bias = self.distance_bias(bucket).permute(0, 3, 1, 2) + key_padding_bias(g.node_mask)
        for layer in self.layers:
            x = layer(x, bias)
        return self.norm(x)


def constants_as_global_parameters(features: torch.Tensor) -> torch.Tensor:
    """(..., 21) node features with every Const node recoded as a global Par node: type
    one-hot Par, parameter scope -1, and the sign and log10|c| columns cleared (the value
    enters through the encoder's value MLP instead)."""
    out = features.clone()
    const = features[..., CONST] == 1
    out[..., CONST] = 0.0
    out[..., PAR] = torch.where(const, 1.0, out[..., PAR])
    out[..., SCOPE] = torch.where(const, -1.0, out[..., SCOPE])
    out[..., SIGN] = 0.0
    out[..., LOG10] = 0.0
    return out


def global_values(g: GraphBatch, psi: torch.Tensor) -> torch.Tensor:
    """(B, M) the encoded value of every global node: each Const node's (g.const_value) and
    each global Par node's Psi[k] (psi (B, K), as encode_parameters gives it); 0 elsewhere."""
    if g.const_value is None:
        raise ValueError("the batch has no Const values: the store needs node_value.npy "
                         "(scripts/pack_node_values.py) and the sinh parameter policy")
    M = g.const_value.shape[1]
    nodes = torch.arange(M, device=psi.device)
    at = g.global_par[..., None] == nodes             # (B, K, M); -1 padding matches nothing
    return g.const_value + (at.to(psi.dtype) * psi[..., None]).sum(1)


def read_out(h: torch.Tensor, g: GraphBatch) -> Readouts:
    """h: (B, M, d) node embeddings -> the readouts of the module docstring."""
    kappa, kappa_mask = _gather(h, g.regional_par)
    lam, lam_mask = _gather(h, g.global_par)
    roots, roots_mask = _gather(h, g.deriv_roots)
    aggs, agg_mask = _gather(h, g.agg)
    o = h[torch.arange(len(h), device=h.device), g.out_root]
    return Readouts(kappa=kappa, kappa_mask=kappa_mask, lam=lam, lam_mask=lam_mask,
                    u=masked_mean(roots, roots_mask[..., None], 1), o=o,
                    m_agg=masked_mean(aggs, agg_mask[..., None], 1),
                    z=masked_mean(h, g.node_mask[..., None], 1))


def _gather(h: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """h (B, M, d), index (B, n) with -1 padding -> (B, n, d) rows (zero at padding), mask."""
    mask = index >= 0
    rows = torch.gather(h, 1, index.clamp(min=0)[..., None].expand(-1, -1, h.shape[-1]))
    return rows * mask[..., None].to(h.dtype), mask
