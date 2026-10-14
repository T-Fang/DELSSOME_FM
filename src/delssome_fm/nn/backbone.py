"""The backbone (architecture.md §4): template encoder, region tokens, and one transformer
over the sequence [CLS, 68 region tokens, M template tokens], with a learned embedding per
segment. Stage 1 and stage 2 share it; only the heads differ.

Outputs used downstream are h_CLS (B, d) and the region outputs r (B, N, d). The template
positions of the output are discarded, but the template tokens are in the sequence so that
region tokens can attend to individual equation nodes, not only to pooled readouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from delssome_fm.nn.layers import TransformerLayer, key_padding_bias
from delssome_fm.nn.template_encoder import GraphBatch, Readouts, TemplateEncoder, read_out
from delssome_fm.nn.tokens import RegionTokens

CLS, REGION, TEMPLATE = 0, 1, 2   # segment ids


@dataclass(frozen=True)
class BackboneSize:
    """Widths and depths (configs/stage1.yaml)."""

    n_regions: int
    d_model: int
    d_template: int
    template_layers: int
    backbone_layers: int
    n_heads: int
    ffn_multiplier: int
    dropout: float
    max_distance: int
    max_degree: int


@dataclass
class BackboneOutput:
    h_cls: torch.Tensor      # (B, d)
    regions: torch.Tensor    # (B, N, d)  r_i
    readouts: Readouts


class Backbone(nn.Module):
    def __init__(self, size: BackboneSize):
        super().__init__()
        s = size
        self.template = TemplateEncoder(s.d_template, s.template_layers, s.n_heads,
                                        s.ffn_multiplier, s.dropout, s.max_distance, s.max_degree)
        self.project = nn.Linear(s.d_template, s.d_model)   # W_proj
        self.tokens = RegionTokens(s.d_model, s.n_regions)
        self.cls = nn.Parameter(torch.randn(s.d_model) * 0.02)
        self.segment = nn.Embedding(3, s.d_model)
        self.layers = nn.ModuleList(TransformerLayer(s.d_model, s.n_heads, s.ffn_multiplier,
                                                     s.dropout)
                                    for _ in range(s.backbone_layers))
        self.norm = nn.LayerNorm(s.d_model)

    def forward(self, theta: torch.Tensor, psi: torch.Tensor, sc: torch.Tensor,
                graph: GraphBatch) -> BackboneOutput:
        """theta (B, N, P), psi (B, K), sc (B, N, N): as RegionTokens; graph: the models' DAGs."""
        template = self.project(self.template(graph))                 # (B, M, d)
        readouts = read_out(template, graph)
        regions = self.tokens(theta, psi, sc, readouts)               # (B, N, d)
        B, N, _ = regions.shape
        x = torch.cat([self.cls.expand(B, 1, -1), regions, template], dim=1)
        segments = torch.cat([torch.full((N + 1,), REGION, device=x.device),
                              torch.full((template.shape[1],), TEMPLATE, device=x.device)])
        segments[0] = CLS
        x = x + self.segment(segments)[None]
        valid = torch.cat([torch.ones(B, N + 1, dtype=torch.bool, device=x.device),
                           graph.node_mask], dim=1)
        bias = key_padding_bias(valid)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.norm(x)
        return BackboneOutput(h_cls=x[:, 0], regions=x[:, 1:N + 1], readouts=readouts)
