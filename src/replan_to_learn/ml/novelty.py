"""
SIH26054 Replan to Learn: L5 UNKNOWN novelty-detection arm.

Spec (04_ml_rul_mission_probe.md Sec 1.3): "UNKNOWN is a real class trained
on held-out fault types the model never saw -- a novelty-detection arm
(one-class SVM on the healthy residual manifold, thresholded on Mahalanobis
distance)."

Implementation: PCA-reduced feature space (172 raw dims is too
high-dimensional for a stable 15-parameter-scale covariance estimate with
the training-set sizes available here), then BOTH:
  1. a one-class SVM (RBF kernel) fit on the healthy manifold in PCA space,
  2. a Mahalanobis distance against the healthy manifold's PCA mean/covariance,
     thresholded at the chi-squared (df = n_pca_components) 99th percentile.
A sample is flagged novel only if BOTH agree it is outside the healthy
manifold -- requiring agreement between a discriminative one-class boundary
and a distributional distance check is deliberately conservative (spec's
own framing: this exists so the system can honestly say "I have not seen
this," not so it flags everything unfamiliar as UNKNOWN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NoveltyDetector:
    pca_mean: np.ndarray            # (172,)
    pca_components: np.ndarray      # (k, 172)
    healthy_pca_mean: np.ndarray    # (k,)
    healthy_pca_cov_inv: np.ndarray  # (k, k)
    mahalanobis_threshold: float
    ocsvm: object                   # fitted sklearn.svm.OneClassSVM (in PCA space)

    def _to_pca(self, X: np.ndarray) -> np.ndarray:
        return (X - self.pca_mean) @ self.pca_components.T

    def mahalanobis_distances(self, X: np.ndarray) -> np.ndarray:
        Xp = self._to_pca(X) - self.healthy_pca_mean
        return np.sqrt(np.einsum("ij,jk,ik->i", Xp, self.healthy_pca_cov_inv, Xp))

    def is_novel(self, X: np.ndarray) -> np.ndarray:
        Xp = self._to_pca(X)
        ocsvm_outlier = self.ocsvm.predict(Xp) == -1
        maha = self.mahalanobis_distances(X)
        maha_outlier = maha > self.mahalanobis_threshold
        return ocsvm_outlier & maha_outlier

    @classmethod
    def fit(cls, X_healthy: np.ndarray, n_components: int = 20, nu: float = 0.05) -> "NoveltyDetector":
        from scipy import stats
        from sklearn.decomposition import PCA
        from sklearn.svm import OneClassSVM

        pca = PCA(n_components=min(n_components, X_healthy.shape[1], X_healthy.shape[0] - 1))
        Xp = pca.fit_transform(X_healthy)
        k = Xp.shape[1]

        mean = np.mean(Xp, axis=0)
        cov = np.cov(Xp, rowvar=False) + np.eye(k) * 1e-6
        cov_inv = np.linalg.inv(cov)

        threshold = float(np.sqrt(stats.chi2.ppf(0.99, df=k)))

        ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        ocsvm.fit(Xp)

        return cls(
            pca_mean=pca.mean_,
            pca_components=pca.components_,
            healthy_pca_mean=mean,
            healthy_pca_cov_inv=cov_inv,
            mahalanobis_threshold=threshold,
            ocsvm=ocsvm,
        )
