"""
SIH26054 Replan to Learn: L7 Planner Datatypes.
Manoeuvre specifications, simulation results, probe scores, and cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ManoeuvreType(Enum):
    STEP_CLIMB = "step_climb"
    POWER_DERATE = "power_derate"
    MIXTURE_ENRICHMENT = "mixture_enrichment"
    AIRSPEED_STEP = "airspeed_step"
    DESCENT_SEGMENT = "descent_segment"


@dataclass(frozen=True)
class CandidateManoeuvre:
    """
    Parameterised candidate manoeuvre for probe selection.
    """
    manoeuvre_type: ManoeuvreType
    parameters: Dict[str, float]
    estimated_duration_s: float
    estimated_cost: float
    target_regimes: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manoeuvre_type": self.manoeuvre_type.value,
            "parameters": dict(self.parameters),
            "estimated_duration_s": float(self.estimated_duration_s),
            "estimated_cost": float(self.estimated_cost),
            "target_regimes": list(self.target_regimes),
        }


@dataclass(frozen=True)
class FeasibilityResult:
    """
    Feasibility check result for a candidate manoeuvre.
    """
    is_feasible: bool
    violations: Tuple[str, ...]
    fuel_reserve_after_s: float
    return_margin_s: float


@dataclass(frozen=True)
class CostModel:
    """
    Multi-component cost model for manoeuvres.
    """
    w_life: float = 1.0
    w_time: float = 1.0
    w_fuel: float = 1.0
    w_exposure: float = 0.0
    c0: float = 0.01

    def compute_cost(self, delta_life_h: float, delta_time_h: float, delta_fuel_kg: float, delta_exposure: float) -> float:
        return (
            self.w_life * delta_life_h
            + self.w_time * delta_time_h
            + self.w_fuel * delta_fuel_kg
            + self.w_exposure * delta_exposure
            + self.c0
        )


@dataclass(frozen=True)
class ProbeScore:
    """
    Information-gain score for a candidate manoeuvre.
    """
    manoeuvre: CandidateManoeuvre
    separability_gain: float
    conditioning_gain: float
    cost: float
    composite_score: float
    passes_value_test: bool
    feasibility: FeasibilityResult


@dataclass(frozen=True)
class ProbeSelectionResult:
    """
    Result of probe selection.
    """
    selected_probe: Optional[ProbeScore]
    all_scores: List[ProbeScore]
    ambiguous_set: Tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class MissionSimulationResult:
    """
    Result of simulating a mission on the current engine state.
    """
    risk: str
    risk_probability: float
    recommended_action: str
    alternatives: List[Dict[str, Any]]
    margin_warnings: Tuple[str, ...]
