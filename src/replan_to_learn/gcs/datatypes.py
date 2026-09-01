"""
SIH26054 Replan to Learn: GCS Datatypes.
Operator panel data structures for engine health, diagnosis, prognostics, mission, and fleet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EngineHealthSummary:
    health_percent: int
    status: str
    primary_issue: Optional[str]
    egt: Tuple[float, float, float, float]
    cht: float
    p_oil: float
    t_oil: float
    n_rpm: float
    mdot_f: float


@dataclass(frozen=True)
class DiagnosisPanel:
    verdict: str
    named: Optional[int]
    ambiguous_set: Tuple[int, ...]
    confidence: float
    separable_set: Tuple[str, ...]
    cos_matrix: Optional[np.ndarray]
    contributing_channels: Tuple[int, ...]
    attribution_basis: str
    tree_shap_top5: List[Tuple[str, float]]


@dataclass(frozen=True)
class PrognosticsPanel:
    rul_p05: float
    rul_p50: float
    rul_p95: float
    p_fail_before_mission: float
    mission_end_time_s: float
    degradation_trajectory: Optional[np.ndarray]


@dataclass(frozen=True)
class MissionPanel:
    risk: str
    risk_probability: float
    recommendation: str
    alternatives: List[str]
    mission_endurance_s: float


@dataclass(frozen=True)
class FleetPanel:
    fleet_epoch: Optional[int]
    contributor_count: int
    borrowed_regimes: Tuple[int, ...]
    r2: Optional[float]
    transfer_status: str


@dataclass(frozen=True)
class GCSPanel:
    engine_health: EngineHealthSummary
    diagnosis: DiagnosisPanel
    prognostics: PrognosticsPanel
    mission: MissionPanel
    fleet: FleetPanel
