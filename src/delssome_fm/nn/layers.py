"""Small building blocks shared by the template encoder, the backbone and the heads: an MLP,
a pre-LayerNorm transformer layer whose attention takes an additive bias, and a masked mean.

Shapes use the symbols of docs/architecture.md: B batch, L sequence length, d width.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Linear layers with GELU between them: widths[0] -> widths[1] -> ... -> widths[-1]."""

    def __init__(self, widths: list[int]):
        super().__init__()
        if len(widths) < 2:
            raise ValueError(f"an MLP needs at least an input and an output width, found {widths}")
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    """Pre-LayerNorm transformer encoder layer (self-attention, then a feed-forward block).

    The attention logits get an additive `bias` (B, H or 1, L, L): the template encoder uses
    it for its shortest-path bias, and both encoders use -inf in it to hide padded keys.
    """

    def __init__(self, d: int, n_heads: int, ffn_multiplier: int, dropout: float):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"width {d} is not divisible by {n_heads} heads")
        self.n_heads = n_heads
        self.dropout = dropout
        self.norm_attention = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.attention_out = nn.Linear(d, d)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = MLP([d, ffn_multiplier * d, d])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d), bias: (B, H or 1, L, L) -> (B, L, d)"""
        B, L, d = x.shape
        q, k, v = (self.qkv(self.norm_attention(x))
                   .view(B, L, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype),
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.attention_out(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ffn(self.norm_ffn(x)))


def key_padding_bias(valid: torch.Tensor) -> torch.Tensor:
    """valid: (B, L) True for real positions -> (B, 1, 1, L) 0 there, -inf at padding."""
    bias = torch.zeros(valid.shape, dtype=torch.float32, device=valid.device)
    return bias.masked_fill(~valid, float("-inf"))[:, None, None, :]


def masked_mean(x: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean of x over `dim`, counting only entries where `valid` (broadcast to x) is True.
    Where nothing is valid the result is 0 (e.g. m_agg of a model without coupling)."""
    w = valid.to(x.dtype)
    return (x * w).sum(dim) / w.sum(dim).clamp(min=1.0)
