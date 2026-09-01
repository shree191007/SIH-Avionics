"""
Tests for ml/features.py's streaming feature builder.
Verifies real, checked behavior: output shape, bounded ring-buffer memory,
window semantics, and that features actually respond to injected signal
(not just "the code runs").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from replan_to_learn.contracts.residuals import NUM_RESIDUAL_CHANNELS, ResidualFrame
from replan_to_learn.gate.datatypes import AttributionFrame as GateAttributionFrame
from replan_to_learn.gate.datatypes import ReasonCode, Verdict
from replan_to_learn.ml.features import (
    N_REGIME_CONTEXT_FEATURES,
    StreamingFeatureBuilder,
    TOTAL_FEATURES,
    WINDOW_S,
)


def _rf(t: float, z, regime: int = 4, status: int = 0) -> ResidualFrame:
    return ResidualFrame(
        t=t,
        z=tuple(float(v) for v in z),
        regime=regime,
        regime_stable_s=10.0,
        spatial=(0.1, 0.05, 0.2, 1),
        sigma=(1.0,) * NUM_RESIDUAL_CHANNELS,
        x_hat=(0.0,) * NUM_RESIDUAL_CHANNELS,
        status=status,
        flight_id="test",
    )


def _gate_af(theta_hat, crlb, verdict=Verdict.NAMED, named=0, ambiguous_set=()):
    n = len(theta_hat)
    return GateAttributionFrame(
        t=0.0,
        verdict=verdict,
        theta_hat=np.asarray(theta_hat, dtype=np.float64),
        crlb=np.asarray(crlb, dtype=np.float64),
        named=named,
        ambiguous_set=tuple(ambiguous_set),
        cos_matrix=np.zeros((n, n)),
        borrowed_regimes=(),
        r2=None,
        alpha_hat=None,
        fleet_epoch=None,
        reason=ReasonCode.NONE,
        model_version="1.0.0",
        regime_grid_version="REGIME_GRID_V1",
    )


class TestFeatureVectorShape:
    def test_empty_buffer_returns_zero_vector_of_correct_length(self):
        fb = StreamingFeatureBuilder()
        vec = fb.compute()
        assert vec.shape == (172,)
        assert TOTAL_FEATURES == 172
        assert np.all(vec == 0.0)

    def test_single_ingest_produces_finite_172_vector(self):
        fb = StreamingFeatureBuilder()
        fb.ingest(_rf(0.0, [1.0] * 9))
        vec = fb.compute()
        assert vec.shape == (172,)
        assert np.all(np.isfinite(vec))

    def test_ring_buffer_is_bounded_at_largest_window(self):
        fb = StreamingFeatureBuilder()
        for i in range(5000):
            fb.ingest(_rf(float(i), [0.0] * 9))
        assert len(fb._buf) <= WINDOW_S[-1]


class TestChannelStatisticsRespondToSignal:
    def test_mean_feature_reflects_injected_constant_offset(self):
        """channel_stats block 0 (60s window) layout: [mean, slope, iqr, max|z|]
        per channel, channel 0 = z_egt_1 at offset 0."""
        fb = StreamingFeatureBuilder()
        for i in range(70):
            fb.ingest(_rf(float(i), [3.0] + [0.0] * 8))
        vec = fb.compute()
        mean_egt1_60s = vec[0]
        assert mean_egt1_60s == pytest.approx(3.0, abs=1e-3)

    def test_healthy_flat_residuals_give_near_zero_features(self):
        fb = StreamingFeatureBuilder()
        for i in range(70):
            fb.ingest(_rf(float(i), [0.0] * 9))
        vec = fb.compute()
        # channel-stat block should be all zero for a perfectly flat healthy signal
        assert np.allclose(vec[:108], 0.0, atol=1e-6)

    def test_positive_trend_produces_positive_slope_feature(self):
        fb = StreamingFeatureBuilder()
        for i in range(70):
            fb.ingest(_rf(float(i), [0.05 * i] + [0.0] * 8))
        vec = fb.compute()
        slope_egt1_60s = vec[1]  # [mean, slope, iqr, max] -> index 1
        assert slope_egt1_60s > 0.0


class TestRegimeContext:
    def test_regime_fractions_sum_to_one(self):
        fb = StreamingFeatureBuilder()
        rng_regimes = [0, 0, 1, 1, 2, 3, 4, 5, 0, 1]
        for i, r in enumerate(rng_regimes * 5):
            fb.ingest(_rf(float(i), [0.0] * 9, regime=r))
        vec = fb.compute()
        regime_block = vec[108 + 15 + 6 + 4: 108 + 15 + 6 + 4 + N_REGIME_CONTEXT_FEATURES]
        fractions = regime_block[:13]
        assert fractions.sum() == pytest.approx(1.0, abs=1e-4)

    def test_transition_count_increases_with_more_switching(self):
        fb_stable = StreamingFeatureBuilder()
        for i in range(60):
            fb_stable.ingest(_rf(float(i), [0.0] * 9, regime=2))
        vec_stable = fb_stable.compute()

        fb_switching = StreamingFeatureBuilder()
        for i in range(60):
            fb_switching.ingest(_rf(float(i), [0.0] * 9, regime=i % 3))
        vec_switching = fb_switching.compute()

        idx_transitions = 108 + 15 + 6 + 4 + 13
        assert vec_switching[idx_transitions] > vec_stable[idx_transitions]


class TestEstimatorContext:
    def test_theta_hat_flows_into_estimator_context_block(self):
        fb = StreamingFeatureBuilder()
        theta = np.zeros(15)
        theta[3] = 0.85  # injector 1 fault
        af = _gate_af(theta, np.ones(15) * 5.0, verdict=Verdict.NAMED, named=3)
        for i in range(10):
            fb.ingest(_rf(float(i), [0.0] * 9), gate_attribution=af)
        vec = fb.compute()
        estimator_block = vec[-25:]
        theta_hat_block = estimator_block[:15]
        assert theta_hat_block[3] == pytest.approx(0.85, abs=1e-3)

    def test_no_gate_attribution_gives_zero_filled_estimator_context(self):
        fb = StreamingFeatureBuilder()
        for i in range(10):
            fb.ingest(_rf(float(i), [0.0] * 9))  # no gate_attribution
        vec = fb.compute()
        estimator_block = vec[-25:]
        assert np.all(estimator_block == 0.0)
