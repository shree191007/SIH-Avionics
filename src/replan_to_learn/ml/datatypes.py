"""
SIH26054 Replan to Learn: L5 ML Layer Datatypes.

Class list corrected to 04_ml_rul_mission_probe.md Section 1.3's real set:
HEALTHY, INJECTOR_1..4, COOLING, COMBUSTION, OIL_SYSTEM, SENSOR_EGT_1..4,
UNKNOWN (13 classes) -- replacing the previous placeholder's 6 lumped
classes.

theta-index -> ML-class mapping (THETA_TO_CLASS below): the physics twin's
15-parameter theta vector (physics_twin.py N_THETA docstring:
theta_vol,theta_comb,theta_cool,theta_inj[1..4],theta_oilp,theta_fric,
b_egt[1..4],b_cht,b_poil) does not have a 1:1 correspondence with the
spec's 13 ML classes -- there is no class for theta_vol, theta_fric, b_cht
or b_poil individually. This mapping is a documented design decision, not
part of the spec text itself:
    theta_vol   (0)      -> COMBUSTION   (volumetric-efficiency loss has no
                                           dedicated class; nearest physical
                                           symptom group is the power-balance/
                                           combustion cluster it is entangled
                                           with per gate/identifiability.py's
                                           own naming-accuracy test suite)
    theta_comb  (1)      -> COMBUSTION
    theta_cool  (2)      -> COOLING
    theta_inj_i (3..6)   -> INJECTOR_i
    theta_oilp  (7)      -> OIL_SYSTEM
    theta_fric  (8)      -> OIL_SYSTEM   (friction/FMEP loss has no dedicated
                                           class; nearest mechanical-wear
                                           symptom group is oil system)
    b_egt_i     (9..12)  -> SENSOR_EGT_i
    b_cht       (13)     -> COOLING      (CHT sensor bias has no dedicated
                                           SENSOR_CHT class in the spec's
                                           list; nearest symptom group is
                                           cooling, which it is entangled
                                           with per the gate's own tests)
    b_poil      (14)     -> OIL_SYSTEM   (same reasoning as b_cht -> COOLING)
UNKNOWN is not reachable via this theta mapping -- it is populated
separately by the novelty-detection arm (ml/novelty.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# Class list (spec Sec 1.3)
# ============================================================================

CLASS_ORDER: Tuple[str, ...] = (
    "HEALTHY",
    "INJECTOR_1",
    "INJECTOR_2",
    "INJECTOR_3",
    "INJECTOR_4",
    "COOLING",
    "COMBUSTION",
    "OIL_SYSTEM",
    "SENSOR_EGT_1",
    "SENSOR_EGT_2",
    "SENSOR_EGT_3",
    "SENSOR_EGT_4",
    "UNKNOWN",
)
NUM_CLASSES = len(CLASS_ORDER)
assert NUM_CLASSES == 13

# theta index -> ML class name (see module docstring for the documented mapping).
THETA_TO_CLASS: Dict[int, str] = {
    0: "COMBUSTION",   # theta_vol
    1: "COMBUSTION",   # theta_comb
    2: "COOLING",      # theta_cool
    3: "INJECTOR_1",
    4: "INJECTOR_2",
    5: "INJECTOR_3",
    6: "INJECTOR_4",
    7: "OIL_SYSTEM",   # theta_oilp
    8: "OIL_SYSTEM",   # theta_fric
    9: "SENSOR_EGT_1",
    10: "SENSOR_EGT_2",
    11: "SENSOR_EGT_3",
    12: "SENSOR_EGT_4",
    13: "COOLING",     # b_cht
    14: "OIL_SYSTEM",  # b_poil
}

THETA_NAMES: Tuple[str, ...] = (
    "theta_vol", "theta_comb", "theta_cool",
    "theta_inj_1", "theta_inj_2", "theta_inj_3", "theta_inj_4",
    "theta_oilp", "theta_fric",
    "b_egt_1", "b_egt_2", "b_egt_3", "b_egt_4",
    "b_cht", "b_poil",
)


def class_for_theta_index(idx: int) -> Optional[str]:
    return THETA_TO_CLASS.get(idx)


# ============================================================================
# Feature vector container (see ml/features.py for the streaming builder)
# ============================================================================

@dataclass(frozen=True)
class ResidualFeatures172:
    """
    172-dim streaming feature vector for the L5 GBT classifier, plus the
    metadata needed for reporting (flight_id, t, regime).
    Produced by ml.features.StreamingFeatureBuilder.compute().
    """
    vector: np.ndarray        # float32[172]
    t: float
    regime: int
    flight_id: str = ""

    def __post_init__(self) -> None:
        if self.vector.shape[0] != 172:
            raise ValueError(f"ResidualFeatures172.vector must have 172 elements, got {self.vector.shape[0]}")


# ============================================================================
# Prediction output
# ============================================================================

@dataclass(frozen=True)
class FaultClassProbs:
    """Per-class fault probabilities from the L5 classifier (13 classes)."""
    probs: Dict[str, float]

    def __post_init__(self) -> None:
        missing = [c for c in CLASS_ORDER if c not in self.probs]
        if missing:
            raise ValueError(f"FaultClassProbs missing classes: {missing}")

    def __getattr__(self, name: str) -> float:
        # Backward-compatible attribute access, e.g. probs.HEALTHY
        if name in CLASS_ORDER:
            return self.probs[name]
        raise AttributeError(name)

    def top_class(self) -> str:
        return max(self.probs.items(), key=lambda kv: kv[1])[0]

    def to_dict(self) -> Dict[str, float]:
        return dict(self.probs)

    @classmethod
    def from_array(cls, arr: Sequence[float], class_order: Sequence[str] = CLASS_ORDER) -> "FaultClassProbs":
        return cls(probs={c: float(p) for c, p in zip(class_order, arr)})

    @classmethod
    def uniform(cls) -> "FaultClassProbs":
        p = 1.0 / NUM_CLASSES
        return cls(probs={c: p for c in CLASS_ORDER})


@dataclass(frozen=True)
class AttributionFrame:
    """
    L5 output: fused physics-gate + ML attribution.

    G3.1 structural constraint: `fault_probs` here is ALWAYS the
    gate-masked posterior (see ml/gate_bridge.py::mask_posterior_by_gate),
    never the raw model output. `separable_set`/`ambiguous_set` are always
    derived from the real gate AttributionFrame the classifier was called
    with -- there is no constructor path in ml/classifier.py that accepts
    caller-supplied override sets.
    """
    fault_probs: FaultClassProbs
    separable_set: Tuple[str, ...]
    ambiguous_set: Tuple[str, ...]
    gate_verdict: str            # NAMED, AMBIGUOUS, BORROWED, INVALID
    confidence: float
    contributing_channels: Tuple[int, ...]
    regime: int
    is_novel: bool = False
    tree_shap_top5: Tuple[Tuple[str, float], ...] = ()
    flight_id: str = ""


# ============================================================================
# Model artifact
# ============================================================================

@dataclass
class MLModelArtifact:
    """
    Persisted GBT model artifact. `booster_model_str` is LightGBM's native
    text serialization (Booster.model_to_string()); dependency-free to
    reload (LightGBM's own loader), matching spec Sec 1.3's "exported to a
    dependency-free C inference stub for edge" intent at the model-format
    level (the C-stub export itself is out of scope for this pass -- see
    training report).
    """
    booster_model_str: str
    class_order: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    calibrators: Dict[str, bytes]      # per-class isotonic regressor, pickled
    version: str = "1.0.0"
    novelty_detector: Optional[bytes] = None   # pickled OneClassSVM + Mahalanobis threshold
    training_provenance: Optional[Dict[str, object]] = None

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MLModelArtifact":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain an MLModelArtifact")
        return obj


@dataclass(frozen=True)
class MLCalibrationReport:
    """Expected Calibration Error (ECE) and reliability report for L5 (G3.2)."""
    ece: float
    max_ece_per_bin: float
    reliability_data: List[Tuple[float, float, int]]
    brier_score: float = 0.0
    threshold: float = 0.05
    passed: bool = False

    def __post_init__(self) -> None:
        if not self.passed:
            object.__setattr__(self, "passed", self.ece <= self.threshold)
