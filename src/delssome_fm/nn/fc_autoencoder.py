"""FC autoencoder (2026-10-25): compresses a draw's FC, the 2278 z-scored arctanh FC values of
the upper triangle (the stage-1 pairwise target), into a latent vector and back.

    encoder   MLP([2278, *hidden, latent_dim])
    decoder   MLP([latent_dim, *reversed(hidden), 2278])

It is trained alone first (train/autoencoder.py). Stage 1's "latent" FC head then predicts
the latent vector and decodes it with the frozen decoder (nn/heads.py), so the network
predicts a low-dimensional summary of FC while the loss is still taken on FC itself.
"""

from __future__ import annotations

import torch
from torch import nn

from delssome_fm.nn.layers import MLP

N_EDGES = 68 * 67 // 2


class FCAutoencoder(nn.Module):
    def __init__(self, latent_dim: int, hidden: tuple[int, ...], n_edges: int = N_EDGES):
        super().__init__()
        self.latent_dim, self.hidden = latent_dim, tuple(hidden)
        self.encoder = MLP([n_edges, *hidden, latent_dim])
        self.decoder = MLP([latent_dim, *reversed(hidden), n_edges])

    def forward(self, fc: torch.Tensor) -> torch.Tensor:
        """fc (B, 2278) z-scored arctanh FC -> its reconstruction (B, 2278)."""
        return self.decoder(self.encoder(fc))
