"""
Lightweight contrastive encoder for daily routine feature vectors, plus the
NT-Xent (InfoNCE) contrastive loss used to train it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutineEncoder(nn.Module):
    """Encoder f_theta: daily feature vector -> embedding z (before projection).
    """

    def __init__(self, input_dim: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    """g_phi: embedding -> projection space used only for the contrastive loss."""

    def __init__(self, embed_dim: int = 32, proj_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def augment(x: torch.Tensor, jitter_std: float = 0.05, dropout_p: float = 0.1) -> torch.Tensor:
    """Produce an augmented view of a batch of daily feature vectors.
    """
    noise = torch.randn_like(x) * jitter_std
    mask = (torch.rand_like(x) > dropout_p).float()
    return (x + noise) * mask


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Standard NT-Xent / InfoNCE contrastive loss (SimCLR-style).
    """
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))  # exclude self-similarity

    # Positive index for row i: i+N (if i<N) else i-N.
    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)

    return F.cross_entropy(sim, targets)
