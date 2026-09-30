"""
RoutinePrototype: online running centroid of a resident's own embeddings.
A day's deviation score = distance from its embedding to the current
prototype. This is the "anomaly" signal (framed as a wellbeing/support
signal, not intrusion detection).
"""
from __future__ import annotations

import numpy as np


class RoutinePrototype:
    """Exponential-moving-average centroid of a resident's embeddings.
    """

    def __init__(self, embed_dim: int, momentum: float = 0.95, warmup: int = 5):
        self.momentum = momentum
        self.warmup = warmup
        self.centroid: np.ndarray | None = None
        self.n_updates = 0
        self._history: list[np.ndarray] = []  # used only during warmup

    def score(self, z: np.ndarray) -> float:
        """Euclidean distance from z to the current prototype. Returns 0.0
        (i.e. 'not yet anomalous') during warmup, when there's no stable
        prototype to compare against yet."""
        if self.centroid is None:
            return 0.0
        return float(np.linalg.norm(z - self.centroid))

    def update(self, z: np.ndarray, deviation_threshold: float | None = None):
        if self.n_updates < self.warmup:
            self._history.append(z)
            self.n_updates += 1
            if self.n_updates == self.warmup:
                self.centroid = np.mean(self._history, axis=0)
            return

        if deviation_threshold is not None:
            if self.score(z) > deviation_threshold:
                return  # don't let a flagged day pull the prototype toward it

        self.centroid = self.momentum * self.centroid + (1 - self.momentum) * z
        self.n_updates += 1
