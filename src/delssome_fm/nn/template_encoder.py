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
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from delssome_fm.nn.layers import TransformerLayer, key_padding_bias, masked_mean
from delssome_fm.spec.dag import N_FEATURES


@dataclass
class GraphBatch:
    """A batch of equation DAGs, padded to the largest M in the batch.

    features: (B, M, 21) float    degree: (B, M, 2) integer, in/out
    distances: (B, M, M) integer, -1 where there is no path (incl. padding)
    node_mask: (B, M) bool, True for real nodes
    regional_par (B, P_MAX), global_par (B, K_MAX), deriv_roots (B, V_MAX), agg (B, C_MAX):
        node indices, -1 padded;  out_root: (B,) node index
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
                 dropout: float, max_distance: int, max_degree: int):
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

    def forward(self, g: GraphBatch) -> torch.Tensor:
        degree = g.degree.long().clamp(max=self.max_degree)
        x = self.input(g.features) + self.in_degree(degree[..., 0]) + self.out_degree(degree[..., 1])
        distances = g.distances.long()
        no_path = torch.full_like(distances, self.max_distance + 1)
        bucket = torch.where(distances < 0, no_path, distances.clamp(max=self.max_distance))
        bias = self.distance_bias(bucket).permute(0, 3, 1, 2) + key_padding_bias(g.node_mask)
        for layer in self.layers:
            x = layer(x, bias)
        return self.norm(x)


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
