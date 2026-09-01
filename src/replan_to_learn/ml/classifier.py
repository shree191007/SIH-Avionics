"""
SIH26054 Replan to Learn: L5 ML Fault Classifier.

Gradient-boosted decision trees (LightGBM), depth <= 6, <= 300 trees, per
04_ml_rul_mission_probe.md Sec 1.3. Replaces the previous hand-rolled
softmax placeholder.

G3.1 (structural, not statistical): `attributions()` is the ONLY prediction
entry point that returns a class-labelled posterior, and it ALWAYS routes
the raw model output through `ml.gate_bridge.mask_posterior_by_gate()`
against a real `gate.datatypes.AttributionFrame`. There is no parameter on
this class through which a caller can supply its own separable_set/
ambiguous_set to override that mask -- see ml/gate_bridge.py's docstring
and tests/ml/test_g31_structural_masking.py for the defeat-attempt test.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from replan_to_learn.gate.datatypes import AttributionFrame as GateAttributionFrame
from replan_to_learn.ml.datatypes import (
    CLASS_ORDER,
    NUM_CLASSES,
    AttributionFrame,
    FaultClassProbs,
    MLCalibrationReport,
    MLModelArtifact,
)
from replan_to_learn.ml.features import TOTAL_FEATURES
from replan_to_learn.ml.gate_bridge import mask_posterior_by_gate

_FEATURE_NAMES = tuple(f"f{i}" for i in range(TOTAL_FEATURES))


class MLClassifier:
    """LightGBM multiclass GBT fault classifier with isotonic calibration,
    novelty detection, and structurally-enforced G3.1 masking."""

    def __init__(self, artifact: Optional[MLModelArtifact] = None) -> None:
        self.artifact = artifact
        self._booster: Optional[Any] = None
        self._calibrators: Dict[str, Any] = {}
        self._novelty: Optional[Any] = None
        if artifact is not None:
            self._load_artifact(artifact)

    def _load_artifact(self, artifact: MLModelArtifact) -> None:
        import lightgbm as lgb

        self._booster = lgb.Booster(model_str=artifact.booster_model_str)
        self._calibrators = {
            cls: pickle.loads(blob) for cls, blob in artifact.calibrators.items()
        }
        self._novelty = pickle.loads(artifact.novelty_detector) if artifact.novelty_detector else None
        self.artifact = artifact

    @classmethod
    def from_file(cls, path: str) -> "MLClassifier":
        return cls(MLModelArtifact.load(path))

    # -- raw inference ---------------------------------------------------

    def predict_proba_raw(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Uncalibrated (but isotonic-rescaled) posterior, NOT gate-masked.
        Internal use only -- external callers must use attributions()."""
        if self._booster is None:
            raise RuntimeError("Classifier not fitted/loaded. Load an artifact or call fit() first.")
        x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        raw = self._booster.predict(x)[0]  # (num_class,), already softmax-normalized
        raw = {CLASS_ORDER[i]: float(raw[i]) for i in range(NUM_CLASSES)}

        if self._calibrators:
            calibrated = {}
            for c in CLASS_ORDER:
                cal = self._calibrators.get(c)
                if cal is not None:
                    calibrated[c] = float(np.clip(cal.predict([raw[c]])[0], 1e-9, 1.0))
                else:
                    calibrated[c] = raw[c]
            total = sum(calibrated.values())
            if total > 1e-12:
                calibrated = {c: v / total for c, v in calibrated.items()}
            return calibrated
        return raw

    def is_novel(self, feature_vector: np.ndarray) -> bool:
        if self._novelty is None:
            return False
        from replan_to_learn.ml.novelty import NoveltyDetector

        detector: NoveltyDetector = self._novelty
        return detector.is_novel(np.asarray(feature_vector, dtype=np.float64).reshape(1, -1))[0]

    def top_shap_features(self, feature_vector: np.ndarray, top_n: int = 5) -> Tuple[Tuple[str, float], ...]:
        """TreeSHAP-equivalent per-prediction attribution via LightGBM's
        native pred_contrib=True (exact, no separate `shap` dependency --
        shap is not installed in this environment; LightGBM's own
        pred_contrib IS the same SHAP-value computation for tree ensembles)."""
        if self._booster is None:
            return ()
        x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        contrib = self._booster.predict(x, pred_contrib=True)  # (1, num_class*(n_features+1))
        contrib = np.asarray(contrib)[0]
        n_features = TOTAL_FEATURES
        per_class = contrib.reshape(NUM_CLASSES, n_features + 1)  # last col = expected value/bias
        # Sum |contribution| across classes to rank features by overall
        # influence on this specific prediction (a standard multiclass
        # SHAP-attribution summary), dropping the per-class bias term.
        magnitude = np.sum(np.abs(per_class[:, :n_features]), axis=0)
        order = np.argsort(-magnitude)[:top_n]
        return tuple((_FEATURE_NAMES[i], float(magnitude[i])) for i in order)

    # -- gate-masked prediction (the only path with a class label) -------

    def attributions(
        self,
        feature_vector: np.ndarray,
        gate_frame: GateAttributionFrame,
        regime: int,
        flight_id: str = "",
        contributing_channels: Tuple[int, ...] = (),
        compute_shap: bool = True,
    ) -> AttributionFrame:
        """
        The only prediction entry point that returns a class-labelled
        result. Always masks against the real gate verdict (G3.1) --
        see ml/gate_bridge.py.
        """
        raw = self.predict_proba_raw(feature_vector)
        masked, is_ambiguous, separable_set, ambiguous_set = mask_posterior_by_gate(raw, gate_frame)

        novel = self.is_novel(feature_vector)
        probs = FaultClassProbs(probs=masked)
        top = probs.top_class()
        confidence = float(masked[top])

        shap_top5 = self.top_shap_features(feature_vector) if compute_shap else ()

        gate_verdict_name = gate_frame.verdict.name if hasattr(gate_frame.verdict, "name") else str(gate_frame.verdict)

        return AttributionFrame(
            fault_probs=probs,
            separable_set=separable_set,
            ambiguous_set=ambiguous_set,
            gate_verdict=gate_verdict_name,
            confidence=confidence,
            contributing_channels=contributing_channels,
            regime=regime,
            is_novel=novel,
            tree_shap_top5=shap_top5,
            flight_id=flight_id,
        )

    # -- calibration metrics (G3.2) ---------------------------------------

    @staticmethod
    def compute_ece(
        confidences: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 10,
    ) -> MLCalibrationReport:
        """Standard top-1 ECE: bin by predicted top-class confidence,
        compare mean confidence to empirical accuracy in each bin."""
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        max_bin_ece = 0.0
        reliability = []
        n = len(confidences)
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            if i == 0:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences > lo) & (confidences <= hi)
            if np.sum(mask) == 0:
                continue
            bin_conf = float(np.mean(confidences[mask]))
            bin_acc = float(np.mean(correct[mask]))
            bin_size = int(np.sum(mask))
            bin_ece = abs(bin_acc - bin_conf)
            ece += bin_ece * bin_size / n
            max_bin_ece = max(max_bin_ece, bin_ece)
            reliability.append((bin_conf, bin_acc, bin_size))

        return MLCalibrationReport(
            ece=float(ece),
            max_ece_per_bin=float(max_bin_ece),
            reliability_data=reliability,
            threshold=0.05,
            passed=float(ece) <= 0.05,
        )

    @staticmethod
    def compute_brier(probs_matrix: np.ndarray, y_true_idx: np.ndarray) -> float:
        """Multiclass Brier score: mean squared error between predicted
        probability vectors and one-hot true labels."""
        n, k = probs_matrix.shape
        onehot = np.zeros((n, k), dtype=np.float64)
        onehot[np.arange(n), y_true_idx] = 1.0
        return float(np.mean(np.sum((probs_matrix - onehot) ** 2, axis=1)))
