"""
SIH26054 Replan to Learn: Stage 5 evaluation harness datatypes.
Implements 05_gcs_integration_validation_deployment.md Sec 5 (six-arm
comparison) and 04_ml_rul_mission_probe.md Sec 5 (the original arm/metric
definitions this stage retains).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TestCase:
    """
    One evaluation scenario: a real telemetry sequence plus ground truth.
    true_fault_index is None for a genuinely healthy case; an int (0-8,
    the multiplicative theta index) for an injected fault;
    ambiguous_pair for a case constructed to be genuinely unresolvable
    from local single-regime data alone (ground truth is "AMBIGUOUS is
    the only honest verdict", not any single index).
    """
    case_id: str
    frames: List[Any]  # List[TelemetryFrame]
    true_fault_index: Optional[int]
    is_genuinely_ambiguous: bool = False
    source: str = ""  # e.g. "NGAFID:23517", "synthetic:theta_inj1=0.85"
    injected_theta: Optional[np.ndarray] = None
    """
    For synthetic cases only: the full 15-dim theta vector to force onto
    gate.theta_hat before gate.update() -- matches the established fault-
    injection pattern in tests/gate/test_gate_naming_accuracy.py's
    _run_single_fault (setting gate.theta_hat directly, since these
    synthetic cases' raw telemetry itself is not physically perturbed).
    None for real-flight cases, which must let the gate's own UKF produce
    theta_hat from genuine sensor data.
    """


@dataclass
class ArmVerdict:
    """One arm's output for one test case: what it named (if anything), and how confident it claimed to be."""
    case_id: str
    arm: str
    named_index: Optional[int]
    confidence: float  # in [0,1]; arms without real calibration report 1.0 for any non-refusal claim
    refused: bool
    latency_ms: float


@dataclass
class ArmMetrics:
    arm: str
    n_cases: int
    over_confident_misattribution_rate: float  # DECISIVE metric (Sec 5.2)
    refusal_rate: float
    resolution_rate: float  # fraction of refusals subsequently resolved (probe/borrow) -- arms without those stages report 0.0
    false_positive_rate_per_100h: Optional[float]
    precision_macro: Optional[float]
    recall_macro: Optional[float]
    mean_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm, "n_cases": self.n_cases,
            "over_confident_misattribution_rate": self.over_confident_misattribution_rate,
            "refusal_rate": self.refusal_rate, "resolution_rate": self.resolution_rate,
            "false_positive_rate_per_100h": self.false_positive_rate_per_100h,
            "precision_macro": self.precision_macro, "recall_macro": self.recall_macro,
            "mean_latency_ms": self.mean_latency_ms, "p95_latency_ms": self.p95_latency_ms,
        }


@dataclass
class EvalReport:
    arm_metrics: Dict[str, ArmMetrics]
    n_test_cases: int
    decisive_result: str  # human-readable: does F beat C on over-confident misattribution?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_test_cases": self.n_test_cases,
            "decisive_result": self.decisive_result,
            "arms": {k: v.to_dict() for k, v in self.arm_metrics.items()},
        }
