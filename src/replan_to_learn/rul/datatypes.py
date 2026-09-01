"""
SIH26054 Replan to Learn: L6 RUL Datatypes.
Degradation state, particle weights, failure thresholds, and RUL report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FailureThreshold:
    """
    Physics-derived failure threshold for a health parameter.
    Computed from PhysicsTwin, not assumed from data.

    direction: "below" (failure = theta DECREASING through threshold_value
    -- true for every efficiency-loss parameter: theta_vol, theta_comb,
    theta_cool, theta_inj1-4, theta_oilp, all fault toward 0) or "above"
    (failure = theta INCREASING through threshold_value -- true only for
    theta_fric, where the fault direction is excess friction, > nominal).
    Was missing entirely until this field was added -- the particle
    filter's first-passage-time search (rul/particle_filter.py's
    estimate_rul()) had no way to know which direction meant failure and
    was hardcoded to "below", which for theta_fric (threshold ~1.53,
    ABOVE the nominal 1.0 starting point) meant a perfectly healthy
    theta_fric particle satisfied `theta_j > threshold_value` as FALSE
    immediately, reporting first-passage-time=0 (i.e. "already failed")
    for a healthy engine. Real bug, found and fixed alongside the
    max_steps performance fix in the same function.
    """
    theta_index: int
    theta_name: str
    threshold_value: float
    limit_description: str
    operating_point: Dict[str, float]
    direction: str = "below"


@dataclass(frozen=True)
class RULReport:
    """
    Remaining Useful Life estimate from particle filter.
    """
    p05: float
    p50: float
    p95: float
    p_fail_before_mission_end: float
    mission_end_time_s: float
    theta_index: int
    theta_name: str
    current_theta: float
    threshold: float
    particles_converged: bool

    def to_dict(self) -> Dict[str, float]:
        return {
            "p05": float(self.p05),
            "p50": float(self.p50),
            "p95": float(self.p95),
            "p_fail_before_mission_end": float(self.p_fail_before_mission_end),
            "mission_end_time_s": float(self.mission_end_time_s),
            "theta_index": int(self.theta_index),
            "current_theta": float(self.current_theta),
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True)
class RULConfig:
    """
    Configuration for RUL particle filter.
    """
    n_particles: int = 500
    theta_drift_std: float = 1e-4
    rho_mean_reversion: float = 0.95
    rho_drift_std: float = 1e-5
    roughening_eps: float = 0.05
    ess_threshold_frac: float = 0.5
    freeze_on_ambiguous: bool = True
