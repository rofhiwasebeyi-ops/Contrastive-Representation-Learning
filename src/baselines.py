"""
Existing-method baseline for sub-RQ4, evaluated on the SAME
Aruba/Milan daily feature vectors as the main method -- same data, same
train/test structure, different method.

PCAPersonalBaseline: median-of-first-N-days personal routine profile,
deviation = PCA reconstruction error against it.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class PCAPersonalBaseline:
    """Median-of-reference-window personal routine profile + PCA reconstruction
    error, following the personal-baseline-deviation approach used in prior
    smart-home behavioural-anomaly literature."""
    name = "pca_personal_baseline"

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.pca = None
        self.baseline_profile = None

    def fit(self, X_ref: np.ndarray):
        self.baseline_profile = np.median(X_ref, axis=0)
        k = min(self.n_components, X_ref.shape[0], X_ref.shape[1])
        self.pca = PCA(n_components=k)
        self.pca.fit(X_ref - self.baseline_profile)

    def score(self, X: np.ndarray) -> np.ndarray:
        centred = X - self.baseline_profile
        projected = self.pca.transform(centred)
        reconstructed = self.pca.inverse_transform(projected)
        return np.mean((centred - reconstructed) ** 2, axis=1)
