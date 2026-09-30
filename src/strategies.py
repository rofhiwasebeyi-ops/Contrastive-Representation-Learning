"""
The three strategies implemented as a common interface so train.py 
can run all of them identically over the same chronological day stream 
and log directly comparable metrics.

Each strategy implements `observe_day(x_day)`, called once per day in
chronological order, which:
  1. scores the day for deviation using the CURRENT encoder (before updating),
  2. updates the encoder according to the strategy's rule,
  3. updates the routine prototype.

Scoring happens BEFORE updating so a day's anomaly score reflects what the
model believed walking into that day, not what it becomes after training on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from anomaly import RoutinePrototype
from encoder import RoutineEncoder, ProjectionHead, augment, nt_xent_loss
from generative_replay import ConditionalVAE, vae_loss


@dataclass
class DayResult:
    day_index: int
    deviation_score: float
    is_injected_deviation: bool


class BaseStrategy:
    name = "base"

    def __init__(self, input_dim: int, embed_dim: int = 32, hidden_dim: int = 64,
                 lr: float = 1e-3, temperature: float = 0.5, jitter_std: float = 0.05,
                 dropout_p: float = 0.1, prototype_momentum: float = 0.95,
                 device: str = "cpu"):
        self.device = device
        self.encoder = RoutineEncoder(input_dim, embed_dim, hidden_dim).to(device)
        self.proj = ProjectionHead(embed_dim).to(device)
        self.prototype = RoutinePrototype(embed_dim, momentum=prototype_momentum)
        self.temperature = temperature
        self.jitter_std = jitter_std
        self.dropout_p = dropout_p
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.proj.parameters()), lr=lr
        )
        self.results: list[DayResult] = []
        self.deviation_threshold = None  # set after warmup, see train.py

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(x)
        self.encoder.train()
        return z

    def _contrastive_step(self, batch: torch.Tensor):
        """One optimisation step of the contrastive objective on `batch`
        (shape (N, input_dim)), producing two augmented views per row."""
        v1 = augment(batch, jitter_std=self.jitter_std, dropout_p=self.dropout_p)
        v2 = augment(batch, jitter_std=self.jitter_std, dropout_p=self.dropout_p)
        z1 = self.proj(self.encoder(v1))
        z2 = self.proj(self.encoder(v2))
        loss = nt_xent_loss(z1, z2, temperature=self.temperature)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def observe_day(self, x_day: torch.Tensor, day_index: int, is_deviation: bool) -> DayResult:
        raise NotImplementedError

    def storage_bytes(self) -> int:
        raise NotImplementedError


class NaiveFineTuning(BaseStrategy):
    """Lower-bound baseline: trains on each new day only, no history at all."""
    name = "naive_finetuning"

    def observe_day(self, x_day, day_index, is_deviation):
        z = self.embed(x_day)
        score = self.prototype.score(z.cpu().numpy()[0])
        self._contrastive_step(x_day.repeat(8, 1))  # replicate single day to form a batch
        self.prototype.update(z.cpu().numpy()[0], self.deviation_threshold)
        return DayResult(day_index, score, is_deviation)

    def storage_bytes(self) -> int:
        return 0  # retains nothing beyond model weights (not counted as "data")


class RealBufferReplay(BaseStrategy):
    """Retains genuine past days in a buffer and replays them alongside new data."""
    name = "real_buffer_replay"

    def __init__(self, *args, buffer_size: int = 30, replay_batch: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer: list[torch.Tensor] = []
        self.buffer_size = buffer_size
        self.replay_batch = replay_batch

    def observe_day(self, x_day, day_index, is_deviation):
        z = self.embed(x_day)
        score = self.prototype.score(z.cpu().numpy()[0])

        if len(self.buffer) > 0:
            k = min(self.replay_batch, len(self.buffer))
            idx = torch.randperm(len(self.buffer))[:k].tolist()
            batch = torch.cat([self.buffer[i] for i in idx] + [x_day], dim=0)
        else:
            batch = x_day.repeat(8, 1)
        self._contrastive_step(batch)

        self.buffer.append(x_day.detach().clone())
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)  # simple FIFO; importance sampling is a documented extension

        self.prototype.update(z.cpu().numpy()[0], self.deviation_threshold)
        return DayResult(day_index, score, is_deviation)

    def storage_bytes(self) -> int:
        if not self.buffer:
            return 0
        return len(self.buffer) * self.buffer[0].numel() * 4  # float32


class GenerativeReplay(BaseStrategy):
    """Proposed strategy: a per-resident conditional VAE synthesises stand-in
    days for replay, instead of a buffer of real historical days."""
    name = "generative_replay"

    def __init__(self, *args, resident_id: int = 0, n_residents: int = 1,
                 vae_lr: float = 1e-3, replay_batch: int = 8,
                 vae_train_every: int = 1, vae_hidden_dim: int = 64,
                 vae_latent_dim: int = 16, kl_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        input_dim = self.encoder.net[0].in_features
        self.vae = ConditionalVAE(input_dim, n_residents=n_residents,
                                   latent_dim=vae_latent_dim, hidden_dim=vae_hidden_dim
                                   ).to(self.device)
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)
        self.resident_id = resident_id
        self.replay_batch = replay_batch
        self.vae_train_every = vae_train_every
        self.kl_weight = kl_weight
        self.real_days_seen: list[torch.Tensor] = []  # transient, NOT retained long-term

    def _train_vae_step(self, x_day: torch.Tensor):
        rid = torch.full((x_day.shape[0],), self.resident_id, dtype=torch.long, device=self.device)
        recon, mu, logvar = self.vae(x_day, rid)
        loss = vae_loss(recon, x_day, mu, logvar, kl_weight=self.kl_weight)
        self.vae_opt.zero_grad()
        loss.backward()
        self.vae_opt.step()

    def observe_day(self, x_day, day_index, is_deviation):
        z = self.embed(x_day)
        score = self.prototype.score(z.cpu().numpy()[0])

        # Update the generator on today's real day (this is the only place a
        # real day is used -- it is not retained afterwards, only the VAE's
        # updated parameters are kept, which is the entire storage argument).
        self._train_vae_step(x_day)

        if self.vae.mu_head.out_features > 0 and day_index > 5:
            synthetic = self.vae.sample(self.replay_batch, self.resident_id, device=self.device)
            batch = torch.cat([synthetic, x_day], dim=0)
        else:
            batch = x_day.repeat(8, 1)
        self._contrastive_step(batch)

        self.prototype.update(z.cpu().numpy()[0], self.deviation_threshold)
        return DayResult(day_index, score, is_deviation)

    def storage_bytes(self) -> int:
        return self.vae.storage_size_bytes()  # generator parameters only, no raw days


STRATEGY_REGISTRY = {
    "naive_finetuning": NaiveFineTuning,
    "real_buffer_replay": RealBufferReplay,
    "generative_replay": GenerativeReplay,
}
