"""
Tests for ml/novelty.py's UNKNOWN novelty-detection arm (spec Sec 1.3:
one-class SVM on the healthy residual manifold, thresholded on Mahalanobis
distance).
"""

from __future__ import annotations

import numpy as np
import pytest

from replan_to_learn.ml.novelty import NoveltyDetector


@pytest.fixture(scope="module")
def healthy_manifold():
    rng = np.random.default_rng(123)
    # A compact healthy cluster in a 172-dim space, correlated across a few
    # "real" directions (mimicking correlated residual channels) plus noise.
    base = rng.normal(0, 1, size=(400, 8))
    mixing = rng.normal(0, 1, size=(8, 172))
    X = base @ mixing + rng.normal(0, 0.05, size=(400, 172))
    return X


class TestNoveltyDetector:
    def test_healthy_manifold_mostly_not_flagged_novel(self, healthy_manifold):
        detector = NoveltyDetector.fit(healthy_manifold, n_components=8, nu=0.05)
        flags = detector.is_novel(healthy_manifold)
        false_positive_rate = float(np.mean(flags))
        # nu=0.05 permits ~5% of the training manifold itself to be
        # boundary cases; allow generous slack since Mahalanobis AND-gate
        # makes this conservative (fewer false positives than nu alone).
        assert false_positive_rate < 0.15

    def test_far_outlier_is_flagged_novel(self, healthy_manifold):
        detector = NoveltyDetector.fit(healthy_manifold, n_components=8, nu=0.05)
        outlier = np.full((1, 172), 50.0)  # far outside the healthy manifold's scale
        flags = detector.is_novel(outlier)
        assert bool(flags[0]) is True

    def test_mahalanobis_distance_increases_away_from_manifold_mean(self, healthy_manifold):
        detector = NoveltyDetector.fit(healthy_manifold, n_components=8, nu=0.05)
        near = np.mean(healthy_manifold, axis=0, keepdims=True)
        far = near + 30.0
        d_near = detector.mahalanobis_distances(near)[0]
        d_far = detector.mahalanobis_distances(far)[0]
        assert d_far > d_near
