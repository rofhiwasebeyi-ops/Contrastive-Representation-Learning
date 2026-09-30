"""
Generative replay module: a lightweight conditional VAE that compresses a
resident's routine history into a small learned representation, and can
synthesise stand-in routine days for replay -- instead of storing raw days.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim: int, n_residents: int, latent_dim: int = 16,
                 cond_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.resident_embed = nn.Embedding(n_residents, cond_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor, resident_id: torch.Tensor):
        c = self.resident_embed(resident_id)
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.mu_head(h), self.logvar_head(h), c

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x: torch.Tensor, resident_id: torch.Tensor):
        mu, logvar, c = self.encode(x, resident_id)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, resident_id: int, device=None) -> torch.Tensor:
        """Synthesise n stand-in routine days for the given resident, without
        needing any real historical day as input -- this is the actual
        replay mechanism: sample z ~ N(0, I), decode conditioned on resident."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.mu_head.out_features, device=device)
        rid = torch.full((n,), resident_id, dtype=torch.long, device=device)
        c = self.resident_embed(rid)
        return self.decode(z, c)

    def storage_size_bytes(self) -> int:
        """Parameter count x 4 bytes (float32) -- the actual retained footprint
        of this strategy, used for the storage/privacy comparison.
        This is what's retained instead of raw sensor days."""
        return sum(p.numel() for p in self.parameters()) * 4


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, kl_weight: float = 0.1) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl
