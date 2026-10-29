"""The backbone (architecture.md §4): template encoder, input tokens, and one transformer
over the sequence [CLS, tokens, M template tokens], with a learned embedding per segment.
Stage 1 and stage 2 share it; only the heads differ. Two token schemes (nn/tokens.py):

    region     68 region tokens (architecture.md §3); r_i is region i's output
    parameter  P x 68 parameter tokens (§3.4, 2026-10-29); the template encoder reads the
               values of global parameters and constants, and r_i is the mean of region i's
               P outputs (absent parameters masked out of attention and of the mean)

Outputs used downstream are h_CLS (B, d) and the region outputs r (B, N, d). The template
positions of the output are discarded, but the template tokens are in the sequence so that
the tokens can attend to individual equation nodes, not only to pooled readouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from delssome_fm.nn.layers import TransformerLayer, key_padding_bias, masked_mean
from delssome_fm.nn.template_encoder import (GraphBatch, Readouts, TemplateEncoder,
                                             global_values, read_out)
from delssome_fm.nn.tokens import ParameterTokens, RegionTokens

CLS, TOKEN, TEMPLATE = 0, 1, 2   # segment ids
TOKEN_SCHEMES = ("region", "parameter")


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
    token_scheme: str


@dataclass
class BackboneOutput:
    h_cls: torch.Tensor      # (B, d)
    regions: torch.Tensor    # (B, N, d)  r_i
    readouts: Readouts


class Backbone(nn.Module):
    def __init__(self, size: BackboneSize):
        super().__init__()
        s = size
        if s.token_scheme not in TOKEN_SCHEMES:
            raise ValueError(f"token_scheme must be one of {TOKEN_SCHEMES}, found {s.token_scheme!r}")
        self.token_scheme = s.token_scheme
        per_parameter = s.token_scheme == "parameter"
        self.template = TemplateEncoder(s.d_template, s.template_layers, s.n_heads,
                                        s.ffn_multiplier, s.dropout, s.max_distance, s.max_degree,
                                        node_values=per_parameter)
        self.project = nn.Linear(s.d_template, s.d_model)   # W_proj
        self.tokens = (ParameterTokens if per_parameter else RegionTokens)(s.d_model, s.n_regions)
        self.cls = nn.Parameter(torch.randn(s.d_model) * 0.02)
        self.segment = nn.Embedding(3, s.d_model)
        self.layers = nn.ModuleList(TransformerLayer(s.d_model, s.n_heads, s.ffn_multiplier,
                                                     s.dropout)
                                    for _ in range(s.backbone_layers))
        self.norm = nn.LayerNorm(s.d_model)

    def forward(self, theta: torch.Tensor, psi: torch.Tensor, sc: torch.Tensor,
                graph: GraphBatch) -> BackboneOutput:
        """theta (B, N, P), psi (B, K), sc (B, N, N): as RegionTokens; graph: the models' DAGs."""
        B, N, P = theta.shape
        if self.token_scheme == "region":
            template = self.project(self.template(graph))             # (B, M, d)
            readouts = read_out(template, graph)
            tokens = self.tokens(theta, psi, sc, readouts)            # (B, N, d)
            token_mask = torch.ones(B, N, dtype=torch.bool, device=theta.device)
        else:
            template = self.project(self.template(graph, global_values(graph, psi)))
            readouts = read_out(template, graph)
            tokens = self.tokens(theta, sc, readouts.kappa).flatten(1, 2)   # (B, N * P, d)
            token_mask = readouts.kappa_mask[:, None, :].expand(B, N, P).flatten(1)
        T = tokens.shape[1]
        x = torch.cat([self.cls.expand(B, 1, -1), tokens, template], dim=1)
        segments = torch.cat([torch.full((T + 1,), TOKEN, device=x.device),
                              torch.full((template.shape[1],), TEMPLATE, device=x.device)])
        segments[0] = CLS
        x = x + self.segment(segments)[None]
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), token_mask,
                           graph.node_mask], dim=1)
        bias = key_padding_bias(valid)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.norm(x)
        out = x[:, 1:T + 1]
        if self.token_scheme == "parameter":                      # region i: mean of its P
            out = masked_mean(out.view(B, N, P, -1), readouts.kappa_mask[:, None, :, None], 2)
        return BackboneOutput(h_cls=x[:, 0], regions=out, readouts=readouts)
