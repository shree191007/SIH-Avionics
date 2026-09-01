"""
SIH26054 Replan to Learn: L5 ML Streaming Feature Builder.

Implements 04_ml_rul_mission_probe.md Section 1.2: a streaming, fixed-size
ring-buffer feature builder computing the ~172-dimensional feature vector
from ResidualFrame (+ optionally the L4 gate's AttributionFrame) history,
over three windows: 60s / 300s / 1800s.

"Streaming with fixed-size ring buffers; no re-scan of history" is honoured
as: a single ring buffer bounded at the LARGEST window (1800 samples at the
real 1 Hz telemetry rate -- see contracts/telemetry.py, "Immutable raw
telemetry contract at 1 Hz"), with per-window statistics computed only over
the bounded slice of that buffer needed for that window (cost bounded by
window size, never by total flight duration or a full history re-scan).
True O(1)-per-sample updates are not attempted for the median/percentile/
Theil-Sen features -- those are fundamentally order-statistics that need the
windowed values themselves, not just running sums -- but the cost is capped
at the window length, which is the property the spec's constraint protects.

DOCUMENTED DEVIATION -- misfire proxy (spec: "cycle-to-cycle variance of the
10 Hz RPM channel, and its trend", 4 features):
    contracts/telemetry.py documents the RPM sensor as "Crankshaft
    optical/Hall-effect tachometer (10 Hz downsampled to ...)" and the
    TelemetryFrame/ResidualFrame contracts that actually flow through this
    codebase are 1 Hz ("Immutable raw telemetry contract at 1 Hz"). There is
    no real 10 Hz RPM channel available in ResidualFrame history to compute
    a genuine per-combustion-cycle misfire proxy from. Rather than fabricate
    a feature that cannot really be computed at this sample rate, this
    module computes an honest 1 Hz substitute: cycle-to-cycle irregularity
    of z_power_balance (residual channel index 8, the channel most directly
    sensitive to combustion-cycle power delivery), specifically the variance
    and max of its first difference over short/medium windows, plus the
    trend (Theil-Sen slope) of that diff-variance across ten sub-windows of
    the 1800s buffer. This is a legitimate combustion-irregularity proxy at
    the sample rate this codebase actually has -- it is NOT the spec's 10 Hz
    misfire proxy, and is labelled as such wherever it is reported.

DOCUMENTED DEVIATION -- estimator context (spec: "theta_hat, diag(P_theta),
innovation NIS", 25 features): the gate's real AttributionFrame
(gate/datatypes.py) carries theta_hat (15) and crlb (15, the Cramer-Rao
lower bound -- used here as the available diagonal-uncertainty proxy for
P_theta, since L4 does not separately expose a full 15x15 covariance). Using
theta_hat(15) + crlb(15) + NIS(1) would be 31, not 25. To land on the
spec's ~172 total, diag(P_theta)/crlb is reduced to the 9 physical
(non-bias) health parameters (theta_vol, theta_comb, theta_cool,
theta_inj_1..4, theta_oilp, theta_fric -- physics_twin.py's own theta[0:9]
split) since those are the parameters this ML layer's classes are about;
the 6 sensor-bias CRLBs are dropped from this slot. theta_hat itself is
kept at the full 15 dimensions. NIS is computed from the residual frame
itself (z is already a normalised/whitened residual per
contracts/residuals.py, so NIS = mean(z_valid^2) is a real, standard
innovation normalised-squared-error statistic, not a placeholder),
averaged over the most recent 60s window for stability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from replan_to_learn.contracts.residuals import NUM_RESIDUAL_CHANNELS, ResidualFrame
from replan_to_learn.contracts.regimes import CORE_REGIME_BINS

try:
    from replan_to_learn.gate.datatypes import AttributionFrame as GateAttributionFrame
except ImportError:  # pragma: no cover
    GateAttributionFrame = None  # type: ignore

SAMPLE_HZ = 1.0
WINDOW_S = (60, 300, 1800)
N_THETA = 15
N_PHYSICAL_THETA = 9  # theta[0:9]: theta_vol..theta_fric (non-bias parameters)

_N_CORE_REGIMES = len(CORE_REGIME_BINS)  # 12
N_REGIME_CONTEXT_BINS = _N_CORE_REGIMES + 1  # 12 core + 1 "other" (transient/invalid)

N_CHANNEL_STAT_FEATURES = NUM_RESIDUAL_CHANNELS * 4 * len(WINDOW_S)      # 108
N_SPATIAL_FEATURES = 5 * len(WINDOW_S)                                    # 15
N_CROSS_MODAL_FEATURES = 6
N_IRREGULARITY_FEATURES = 4
N_REGIME_CONTEXT_FEATURES = N_REGIME_CONTEXT_BINS + 1                     # 14
N_ESTIMATOR_CONTEXT_FEATURES = N_THETA + N_PHYSICAL_THETA + 1             # 25

TOTAL_FEATURES = (
    N_CHANNEL_STAT_FEATURES
    + N_SPATIAL_FEATURES
    + N_CROSS_MODAL_FEATURES
    + N_IRREGULARITY_FEATURES
    + N_REGIME_CONTEXT_FEATURES
    + N_ESTIMATOR_CONTEXT_FEATURES
)
assert TOTAL_FEATURES == 172, f"feature budget drifted: {TOTAL_FEATURES} != 172"

FEATURE_GROUP_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("channel_stats", N_CHANNEL_STAT_FEATURES),
    ("spatial", N_SPATIAL_FEATURES),
    ("cross_modal_coherence", N_CROSS_MODAL_FEATURES),
    ("irregularity_proxy", N_IRREGULARITY_FEATURES),
    ("regime_context", N_REGIME_CONTEXT_FEATURES),
    ("estimator_context", N_ESTIMATOR_CONTEXT_FEATURES),
)


@dataclass(frozen=True)
class _Sample:
    t: float
    z: np.ndarray                 # (9,)
    spatial: Tuple[float, float, float, int]
    regime: int
    theta_hat: np.ndarray         # (15,) NaN-filled if no gate attribution supplied
    crlb: np.ndarray              # (15,) NaN-filled if no gate attribution supplied


class StreamingFeatureBuilder:
    """
    Fixed-size ring-buffer streaming feature builder for L5.

    Usage:
        fb = StreamingFeatureBuilder(flight_id="F123")
        for rf in residual_frame_stream:
            fb.ingest(rf, gate_attribution)   # gate_attribution optional
            x = fb.compute()                  # np.ndarray, shape (172,)
    """

    def __init__(self, flight_id: str = "") -> None:
        self.flight_id = flight_id
        self._buf: Deque[_Sample] = deque(maxlen=WINDOW_S[-1])

    def reset(self) -> None:
        self._buf.clear()

    def ready(self) -> bool:
        return len(self._buf) > 0

    def ingest(self, rf: ResidualFrame, gate_attribution: Optional[object] = None) -> None:
        if gate_attribution is not None and hasattr(gate_attribution, "theta_hat"):
            theta_hat = np.asarray(gate_attribution.theta_hat, dtype=np.float64)
            crlb = np.asarray(gate_attribution.crlb, dtype=np.float64)
            if theta_hat.shape[0] < N_THETA:
                theta_hat = np.pad(theta_hat, (0, N_THETA - theta_hat.shape[0]), constant_values=np.nan)
            if crlb.shape[0] < N_THETA:
                crlb = np.pad(crlb, (0, N_THETA - crlb.shape[0]), constant_values=np.nan)
        else:
            theta_hat = np.full(N_THETA, np.nan)
            crlb = np.full(N_THETA, np.nan)

        self._buf.append(_Sample(
            t=float(rf.t),
            z=rf.z_array.astype(np.float64),
            spatial=rf.spatial,
            regime=int(rf.regime),
            theta_hat=theta_hat,
            crlb=crlb,
        ))

    # -- window extraction -------------------------------------------------

    def _window_samples(self, window_s: int) -> List[_Sample]:
        if not self._buf:
            return []
        t_now = self._buf[-1].t
        cutoff = t_now - window_s
        out: List[_Sample] = []
        for s in reversed(self._buf):
            if s.t < cutoff - 1e-6:
                break
            out.append(s)
        out.reverse()
        return out

    # -- per-group feature computation --------------------------------------

    @staticmethod
    def _theil_sen_slope(t: np.ndarray, y: np.ndarray) -> float:
        """Robust slope estimator: median of all pairwise slopes.
        Subsampled to <=150 points for windows with many samples, bounding
        the O(n^2) pairwise cost by a fixed constant rather than by the
        window length -- part of the "bounded cost per update" property.
        """
        n = len(t)
        if n < 2:
            return 0.0
        if n > 150:
            idx = np.linspace(0, n - 1, 150).astype(int)
            t = t[idx]
            y = y[idx]
            n = len(t)
        slopes: List[np.ndarray] = []
        for i in range(n - 1):
            dt = t[i + 1:] - t[i]
            mask = np.abs(dt) > 1e-9
            if not np.any(mask):
                continue
            dy = y[i + 1:][mask] - y[i]
            slopes.append(dy / dt[mask])
        if not slopes:
            return 0.0
        all_slopes = np.concatenate(slopes)
        all_slopes = all_slopes[np.isfinite(all_slopes)]
        if all_slopes.size == 0:
            return 0.0
        return float(np.median(all_slopes))

    def _channel_stats(self, samples: Sequence[_Sample]) -> np.ndarray:
        """9 channels x (mean, Theil-Sen slope, IQR, max|z|) = 36."""
        feats = np.zeros(NUM_RESIDUAL_CHANNELS * 4, dtype=np.float64)
        if not samples:
            return feats
        t = np.array([s.t for s in samples])
        Z = np.array([s.z for s in samples])
        for c in range(NUM_RESIDUAL_CHANNELS):
            col = Z[:, c]
            valid = np.isfinite(col)
            if not np.any(valid):
                continue
            cv = col[valid]
            tv = t[valid]
            mean = float(np.mean(cv))
            slope = self._theil_sen_slope(tv, cv)
            if cv.size >= 2:
                q75, q25 = np.percentile(cv, [75, 25])
            else:
                q75 = q25 = cv[0]
            iqr = float(q75 - q25)
            maxabs = float(np.max(np.abs(cv)))
            feats[c * 4:(c + 1) * 4] = [mean, slope, iqr, maxabs]
        return feats

    @staticmethod
    def _spatial_stats(samples: Sequence[_Sample]) -> np.ndarray:
        """alpha, beta, ||s||_inf, argmax_i s (modal dominant cylinder), beta/alpha = 5."""
        if not samples:
            return np.zeros(5, dtype=np.float64)
        alphas = np.array([s.spatial[0] for s in samples], dtype=np.float64)
        betas = np.array([s.spatial[1] for s in samples], dtype=np.float64)
        sinfs = np.array([s.spatial[2] for s in samples], dtype=np.float64)
        doms = np.array([s.spatial[3] for s in samples], dtype=np.int64)
        alpha_mean = float(np.nanmean(alphas)) if np.any(np.isfinite(alphas)) else 0.0
        beta_mean = float(np.nanmean(betas)) if np.any(np.isfinite(betas)) else 0.0
        sinf_mean = float(np.nanmean(sinfs)) if np.any(np.isfinite(sinfs)) else 0.0
        vals, counts = np.unique(doms, return_counts=True)
        dom_mode = float(vals[np.argmax(counts)]) if vals.size else 0.0
        ratio = beta_mean / alpha_mean if abs(alpha_mean) > 1e-9 else 0.0
        return np.array([alpha_mean, beta_mean, sinf_mean, dom_mode, ratio], dtype=np.float64)

    @staticmethod
    def _cross_modal_coherence(samples: Sequence[_Sample]) -> np.ndarray:
        """Per-cylinder EGT/CHT coherence (correlation of z_egt_i vs z_cht), mean, min = 6."""
        if len(samples) < 3:
            return np.zeros(6, dtype=np.float64)
        Z = np.array([s.z for s in samples], dtype=np.float64)
        cht = Z[:, 4]
        coh = []
        for i in range(4):
            egt = Z[:, i]
            mask = np.isfinite(egt) & np.isfinite(cht)
            if np.sum(mask) < 3 or np.std(egt[mask]) < 1e-9 or np.std(cht[mask]) < 1e-9:
                coh.append(0.0)
                continue
            c = np.corrcoef(egt[mask], cht[mask])[0, 1]
            coh.append(float(c) if np.isfinite(c) else 0.0)
        coh_arr = np.array(coh, dtype=np.float64)
        return np.concatenate([coh_arr, [float(np.mean(coh_arr)), float(np.min(coh_arr))]])

    def _irregularity_proxy(
        self,
        samples_60: Sequence[_Sample],
        samples_300: Sequence[_Sample],
        samples_1800: Sequence[_Sample],
    ) -> np.ndarray:
        """1 Hz combustion-irregularity proxy (see module docstring deviation note). 4 features."""

        def diff_var_max(samples: Sequence[_Sample]) -> Tuple[float, float]:
            if len(samples) < 3:
                return 0.0, 0.0
            pb = np.array([s.z[8] for s in samples], dtype=np.float64)
            pb = pb[np.isfinite(pb)]
            if len(pb) < 3:
                return 0.0, 0.0
            d = np.diff(pb)
            return float(np.var(d)), float(np.max(np.abs(d)))

        var60, max60 = diff_var_max(samples_60)
        var300, _ = diff_var_max(samples_300)
        trend = self._diff_var_trend(samples_1800)
        return np.array([var60, var300, trend, max60], dtype=np.float64)

    def _diff_var_trend(self, samples: Sequence[_Sample]) -> float:
        if len(samples) < 20:
            return 0.0
        pb = np.array([s.z[8] for s in samples], dtype=np.float64)
        t = np.array([s.t for s in samples], dtype=np.float64)
        valid = np.isfinite(pb)
        pb = pb[valid]
        t = t[valid]
        if len(pb) < 20:
            return 0.0
        n_buckets = 10
        edges = np.linspace(t[0], t[-1], n_buckets + 1)
        bucket_t, bucket_var = [], []
        for i in range(n_buckets):
            m = (t >= edges[i]) & (t <= edges[i + 1])
            seg = pb[m]
            if len(seg) >= 3:
                bucket_var.append(float(np.var(np.diff(seg))))
                bucket_t.append(float((edges[i] + edges[i + 1]) / 2.0))
        if len(bucket_t) < 2:
            return 0.0
        return self._theil_sen_slope(np.array(bucket_t), np.array(bucket_var))

    @staticmethod
    def _regime_context(samples: Sequence[_Sample]) -> np.ndarray:
        """12 core-regime fractions + 1 'other' fraction + 1 transition count = 14."""
        feats = np.zeros(N_REGIME_CONTEXT_FEATURES, dtype=np.float64)
        if not samples:
            return feats
        regimes = [s.regime for s in samples]
        n = len(regimes)
        counts = {r: 0 for r in CORE_REGIME_BINS}
        other = 0
        for r in regimes:
            if r in counts:
                counts[r] += 1
            else:
                other += 1
        for i, r in enumerate(CORE_REGIME_BINS):
            feats[i] = counts[r] / n
        feats[_N_CORE_REGIMES] = other / n
        transitions = sum(1 for i in range(1, n) if regimes[i] != regimes[i - 1])
        feats[_N_CORE_REGIMES + 1] = float(transitions)
        return feats

    @staticmethod
    def _estimator_context(samples_60: Sequence[_Sample], latest: _Sample) -> np.ndarray:
        """theta_hat(15) + crlb over the 9 physical params (9) + NIS averaged over 60s (1) = 25."""
        theta_hat = np.nan_to_num(latest.theta_hat, nan=0.0)
        crlb_physical = np.nan_to_num(latest.crlb[:N_PHYSICAL_THETA], nan=0.0)
        if samples_60:
            nis_vals = []
            for s in samples_60:
                valid = np.isfinite(s.z)
                if np.any(valid):
                    nis_vals.append(float(np.mean(s.z[valid] ** 2)))
            nis = float(np.mean(nis_vals)) if nis_vals else 0.0
        else:
            nis = 0.0
        return np.concatenate([theta_hat, crlb_physical, [nis]]).astype(np.float64)

    # -- full vector ---------------------------------------------------------

    def compute(self) -> np.ndarray:
        """Compute the full 172-dim feature vector from current buffer state."""
        if not self._buf:
            return np.zeros(TOTAL_FEATURES, dtype=np.float32)

        s60 = self._window_samples(60)
        s300 = self._window_samples(300)
        s1800 = self._window_samples(1800)

        parts = [
            self._channel_stats(s60),
            self._channel_stats(s300),
            self._channel_stats(s1800),
            self._spatial_stats(s60),
            self._spatial_stats(s300),
            self._spatial_stats(s1800),
            self._cross_modal_coherence(s300),
            self._irregularity_proxy(s60, s300, s1800),
            self._regime_context(s1800),
            self._estimator_context(s60, self._buf[-1]),
        ]
        vec = np.concatenate(parts)
        # Clip before the float32 cast: some upstream quantities (e.g. the
        # gate's CRLB near a poorly-conditioned Fisher information matrix)
        # can be enormous-but-finite float64 values that overflow float32's
        # range on cast, which np.nan_to_num (applied AFTER cast) cannot
        # fix retroactively -- the overflow already happened.
        finfo = np.finfo(np.float32)
        vec = np.clip(vec, finfo.min, finfo.max)
        vec = vec.astype(np.float32)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    def latest_regime(self) -> int:
        return self._buf[-1].regime if self._buf else -1

    def latest_t(self) -> float:
        return self._buf[-1].t if self._buf else 0.0
