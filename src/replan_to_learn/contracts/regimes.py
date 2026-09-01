"""
SIH26054 Replan to Learn: Foundation Data Contracts & Schemas.
Regime Contract: REGIME_GRID_V1, Quasi-Steady Stability & Encounter Tracker.

Defines the permanent REGIME_GRID_V1 (12 core fingerprint bins + 3 operational bins),
enforces the 20-second quasi-steady stability detection rules:
- |dN/dt| < 30 rpm/s
- |dp_im/dt| < 2.0 kPa/s (2000 Pa/s)
- |dT_hd/dt| < 0.15 K/s
and maintains per-aircraft regime exposure counters n_i(r).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple
import json
import math
import numpy as np


# ============================================================================
# 1. Versioning and Grid Constants
# ============================================================================

REGIME_GRID_VERSION = "REGIME_GRID_V1"
MCP_POWER_KW = 100.0            # Rotax 915 iS Maximum Continuous Power (kW)
ALTITUDE_THRESHOLD_M = 3000.0   # Altitude extension threshold (m MSL)
ROC_CRUISE_THRESHOLD_MS = 1.5   # Vertical speed threshold for climb/descent (m/s)
IDLE_RPM_THRESHOLD = 1600.0     # Engine idle threshold (rpm)

# Quasi-steady derivative thresholds
MAX_DERIV_RPM_PER_S = 30.0      # |dN/dt| < 30.0 rpm/s
MAX_DERIV_MAP_PA_PER_S = 2000.0 # |dp_im/dt| < 2.0 kPa/s (2000 Pa/s)
MAX_DERIV_CHT_K_PER_S = 0.15    # |dT_hd/dt| < 0.15 K/s
MIN_STABLE_DURATION_S = 20.0    # 20s continuous sustained stability window


class RegimeBin(IntEnum):
    """REGIME_GRID_V1 12-Core + 3 Operational Bins."""
    # Climb Bins (ROC > +1.5 m/s, hp <= 3000m)
    CLIMB_LOW_POWER = 0
    CLIMB_MID_POWER = 1
    CLIMB_HIGH_POWER = 2
    # Cruise Bins (|ROC| <= 1.5 m/s, hp <= 3000m)
    CRUISE_LOW_POWER = 3
    CRUISE_MID_POWER = 4
    CRUISE_HIGH_POWER = 5
    # Descent Bins (ROC < -1.5 m/s, hp <= 3000m)
    DESCENT_LOW_POWER = 6
    DESCENT_MID_POWER = 7
    DESCENT_HIGH_POWER = 8
    # Altitude Extension Bins (hp > 3000m)
    ALTITUDE_LOW_POWER = 9
    ALTITUDE_MID_POWER = 10
    ALTITUDE_HIGH_POWER = 11
    # Operational / Non-Fingerprint Bins
    IDLE = 12
    TRANSIENT = 13
    INVALID = 14


BIN_CLIMB_LOW_POWER = RegimeBin.CLIMB_LOW_POWER.value
BIN_CLIMB_MID_POWER = RegimeBin.CLIMB_MID_POWER.value
BIN_CLIMB_HIGH_POWER = RegimeBin.CLIMB_HIGH_POWER.value
BIN_CRUISE_LOW_POWER = RegimeBin.CRUISE_LOW_POWER.value
BIN_CRUISE_MID_POWER = RegimeBin.CRUISE_MID_POWER.value
BIN_CRUISE_HIGH_POWER = RegimeBin.CRUISE_HIGH_POWER.value
BIN_DESCENT_LOW_POWER = RegimeBin.DESCENT_LOW_POWER.value
BIN_DESCENT_MID_POWER = RegimeBin.DESCENT_MID_POWER.value
BIN_DESCENT_HIGH_POWER = RegimeBin.DESCENT_HIGH_POWER.value
BIN_ALTITUDE_LOW_POWER = RegimeBin.ALTITUDE_LOW_POWER.value
BIN_ALTITUDE_MID_POWER = RegimeBin.ALTITUDE_MID_POWER.value
BIN_ALTITUDE_HIGH_POWER = RegimeBin.ALTITUDE_HIGH_POWER.value
BIN_IDLE = RegimeBin.IDLE.value
BIN_TRANSIENT = RegimeBin.TRANSIENT.value
BIN_INVALID = RegimeBin.INVALID.value

CORE_REGIME_BINS = list(range(12))
NUM_CORE_BINS = 12
NUM_TOTAL_BINS = 15

REGIME_NAMES: Dict[int, str] = {
    0: "CLIMB_LOW_POWER",
    1: "CLIMB_MID_POWER",
    2: "CLIMB_HIGH_POWER",
    3: "CRUISE_LOW_POWER",
    4: "CRUISE_MID_POWER",
    5: "CRUISE_HIGH_POWER",
    6: "DESCENT_LOW_POWER",
    7: "DESCENT_MID_POWER",
    8: "DESCENT_HIGH_POWER",
    9: "ALTITUDE_LOW_POWER",
    10: "ALTITUDE_MID_POWER",
    11: "ALTITUDE_HIGH_POWER",
    12: "IDLE",
    13: "TRANSIENT",
    14: "INVALID",
}


# ============================================================================
# 2. Regime Grid Specification Definition
# ============================================================================

@dataclass(frozen=True)
class RegimeDefinition:
    """Immutable specification for a single regime bin."""
    bin_id: int
    name: str
    is_core_fingerprint: bool
    flight_phase: str           # 'CLIMB', 'CRUISE', 'DESCENT', 'ALTITUDE', 'IDLE', 'TRANSIENT', 'INVALID'
    power_band: str             # 'LOW', 'MID', 'HIGH', 'NONE'
    min_power_frac: float
    max_power_frac: float
    min_roc_ms: float
    max_roc_ms: float
    min_altitude_m: float
    max_altitude_m: float
    description: str


REGIME_GRID_DEFINITIONS: Dict[int, RegimeDefinition] = {
    0: RegimeDefinition(0, "CLIMB_LOW_POWER", True, "CLIMB", "LOW", 0.0, 0.45, 1.5, 50.0, -1000.0, 3000.0, "Climb below 3000m at < 45% MCP"),
    1: RegimeDefinition(1, "CLIMB_MID_POWER", True, "CLIMB", "MID", 0.45, 0.75, 1.5, 50.0, -1000.0, 3000.0, "Climb below 3000m at 45-75% MCP"),
    2: RegimeDefinition(2, "CLIMB_HIGH_POWER", True, "CLIMB", "HIGH", 0.75, 1.50, 1.5, 50.0, -1000.0, 3000.0, "Climb below 3000m at > 75% MCP"),
    3: RegimeDefinition(3, "CRUISE_LOW_POWER", True, "CRUISE", "LOW", 0.0, 0.45, -1.5, 1.5, -1000.0, 3000.0, "Level cruise below 3000m at < 45% MCP"),
    4: RegimeDefinition(4, "CRUISE_MID_POWER", True, "CRUISE", "MID", 0.45, 0.75, -1.5, 1.5, -1000.0, 3000.0, "Level cruise below 3000m at 45-75% MCP"),
    5: RegimeDefinition(5, "CRUISE_HIGH_POWER", True, "CRUISE", "HIGH", 0.75, 1.50, -1.5, 1.5, -1000.0, 3000.0, "Level cruise below 3000m at > 75% MCP"),
    6: RegimeDefinition(6, "DESCENT_LOW_POWER", True, "DESCENT", "LOW", 0.0, 0.45, -50.0, -1.5, -1000.0, 3000.0, "Descent below 3000m at < 45% MCP"),
    7: RegimeDefinition(7, "DESCENT_MID_POWER", True, "DESCENT", "MID", 0.45, 0.75, -50.0, -1.5, -1000.0, 3000.0, "Descent below 3000m at 45-75% MCP"),
    8: RegimeDefinition(8, "DESCENT_HIGH_POWER", True, "DESCENT", "HIGH", 0.75, 1.50, -50.0, -1.5, -1000.0, 3000.0, "Descent below 3000m at > 75% MCP"),
    9: RegimeDefinition(9, "ALTITUDE_LOW_POWER", True, "ALTITUDE", "LOW", 0.0, 0.45, -50.0, 50.0, 3000.0, 15000.0, "High altitude (>3000m) at < 45% MCP"),
    10: RegimeDefinition(10, "ALTITUDE_MID_POWER", True, "ALTITUDE", "MID", 0.45, 0.75, -50.0, 50.0, 3000.0, 15000.0, "High altitude (>3000m) at 45-75% MCP"),
    11: RegimeDefinition(11, "ALTITUDE_HIGH_POWER", True, "ALTITUDE", "HIGH", 0.75, 1.50, -50.0, 50.0, 3000.0, 15000.0, "High altitude (>3000m) at > 75% MCP"),
    12: RegimeDefinition(12, "IDLE", False, "IDLE", "NONE", 0.0, 0.30, -50.0, 50.0, -1000.0, 15000.0, "Engine idle speed (<1600 rpm)"),
    13: RegimeDefinition(13, "TRANSIENT", False, "TRANSIENT", "NONE", 0.0, 1.50, -50.0, 50.0, -1000.0, 15000.0, "Quasi-steady stability conditions violated"),
    14: RegimeDefinition(14, "INVALID", False, "INVALID", "NONE", 0.0, 1.50, -50.0, 50.0, -1000.0, 15000.0, "Invalid or missing telemetry sensor stream"),
}


# ============================================================================
# 3. Quasi-Steady Stability & Regime Classifier
# ============================================================================

class RegimeClassifier:
    """
    Online Quasi-Steady Stability Detector and REGIME_GRID_V1 Classifier.

    Tracks derivatives of N (RPM), MAP (p_im), and CHT (T_hd) over time.
    Requires continuous satisfaction of stability conditions for >= 20.0 seconds
    before assigning a core regime bin (0..11).
    """

    def __init__(
        self,
        min_stable_duration_s: float = MIN_STABLE_DURATION_S,
        mcp_kw: float = MCP_POWER_KW,
    ) -> None:
        self.min_stable_duration_s = min_stable_duration_s
        self.mcp_kw = mcp_kw
        self.reset()

    def reset(self) -> None:
        """Reset internal classifier history."""
        self._prev_t: Optional[float] = None
        self._prev_rpm: Optional[float] = None
        self._prev_map_pa: Optional[float] = None
        self._prev_cht_k: Optional[float] = None
        self._prev_alt_m: Optional[float] = None
        self._stable_duration_s: float = 0.0
        self._last_candidate_bin: Optional[int] = None

    @property
    def current_stable_duration(self) -> float:
        """Current accumulated continuous stable duration in seconds."""
        return self._stable_duration_s

    def check_stability_derivatives(
        self,
        dt: float,
        d_rpm: float,
        d_map_pa: float,
        d_cht_k: float,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if derivatives satisfy the quasi-steady criteria:
        |dN/dt| < 30 rpm/s, |dp_im/dt| < 2000 Pa/s, |dT_hd/dt| < 0.15 K/s.
        """
        if dt <= 1e-6:
            return False, {"d_rpm_dt": 0.0, "d_map_dt": 0.0, "d_cht_dt": 0.0}

        d_rpm_dt = abs(d_rpm / dt)
        d_map_dt = abs(d_map_pa / dt)
        d_cht_dt = abs(d_cht_k / dt)

        is_stable = (
            d_rpm_dt < MAX_DERIV_RPM_PER_S and
            d_map_dt < MAX_DERIV_MAP_PA_PER_S and
            d_cht_dt < MAX_DERIV_CHT_K_PER_S
        )

        return is_stable, {
            "d_rpm_dt": d_rpm_dt,
            "d_map_dt": d_map_dt,
            "d_cht_dt": d_cht_dt,
        }

    def determine_candidate_bin(
        self,
        rpm: float,
        p_brake_kw: float,
        roc_ms: float,
        altitude_m: float,
    ) -> int:
        """
        Map instantaneous operating state (RPM, Power, ROC, Altitude) to a regime bin ID.
        """
        if rpm < IDLE_RPM_THRESHOLD:
            return BIN_IDLE

        power_frac = max(0.0, p_brake_kw / self.mcp_kw)

        # High Altitude extension check (> 3000m MSL)
        if altitude_m > ALTITUDE_THRESHOLD_M:
            if power_frac < 0.45:
                return BIN_ALTITUDE_LOW_POWER
            elif power_frac <= 0.75:
                return BIN_ALTITUDE_MID_POWER
            else:
                return BIN_ALTITUDE_HIGH_POWER

        # Flight Phase Determination based on Rate of Climb (ROC)
        if roc_ms > ROC_CRUISE_THRESHOLD_MS:
            # Climb Phase
            if power_frac < 0.45:
                return BIN_CLIMB_LOW_POWER
            elif power_frac <= 0.75:
                return BIN_CLIMB_MID_POWER
            else:
                return BIN_CLIMB_HIGH_POWER
        elif roc_ms < -ROC_CRUISE_THRESHOLD_MS:
            # Descent Phase
            if power_frac < 0.45:
                return BIN_DESCENT_LOW_POWER
            elif power_frac <= 0.75:
                return BIN_DESCENT_MID_POWER
            else:
                return BIN_DESCENT_HIGH_POWER
        else:
            # Level Cruise Phase
            if power_frac < 0.45:
                return BIN_CRUISE_LOW_POWER
            elif power_frac <= 0.75:
                return BIN_CRUISE_MID_POWER
            else:
                return BIN_CRUISE_HIGH_POWER

    def step(
        self,
        t: float,
        rpm: float,
        map_pa: float,
        cht_k: float,
        altitude_m: float = 0.0,
        p_brake_kw: float = 0.0,
        is_valid_telemetry: bool = True,
        **kwargs: object,
    ) -> Tuple[int, float]:
        """
        Process a 1 Hz telemetry step and emit (regime_bin, stable_duration_s).
        Supports aliases: alt_m for altitude_m, p_brake for p_brake_kw.
        """
        if "alt_m" in kwargs:
            altitude_m = float(kwargs["alt_m"])  # type: ignore[arg-type]
        if "p_brake" in kwargs:
            p_brake_kw = float(kwargs["p_brake"])  # type: ignore[arg-type]

        if not is_valid_telemetry or math.isnan(rpm) or math.isnan(map_pa) or math.isnan(cht_k):
            self._stable_duration_s = 0.0
            self._prev_t = t
            self._prev_rpm = rpm
            self._prev_map_pa = map_pa
            self._prev_cht_k = cht_k
            self._prev_alt_m = altitude_m
            return BIN_INVALID, 0.0

        # Handle initial step or time jump
        if self._prev_t is None:
            self._prev_t = t
            self._prev_rpm = rpm
            self._prev_map_pa = map_pa
            self._prev_cht_k = cht_k
            self._prev_alt_m = altitude_m
            self._stable_duration_s = 0.0
            return BIN_TRANSIENT, 0.0

        dt = t - self._prev_t
        if dt <= 0.0 or dt > 5.0:  # Time step discontinuity / gap
            self.reset()
            self._prev_t = t
            self._prev_rpm = rpm
            self._prev_map_pa = map_pa
            self._prev_cht_k = cht_k
            self._prev_alt_m = altitude_m
            return BIN_TRANSIENT, 0.0

        d_rpm = rpm - self._prev_rpm  # type: ignore[operator]
        d_map_pa = map_pa - self._prev_map_pa  # type: ignore[operator]
        d_cht_k = cht_k - self._prev_cht_k  # type: ignore[operator]
        d_alt_m = altitude_m - self._prev_alt_m  # type: ignore[operator]
        roc_ms = d_alt_m / dt

        is_stable, _ = self.check_stability_derivatives(dt, d_rpm, d_map_pa, d_cht_k)
        candidate_bin = self.determine_candidate_bin(rpm, p_brake_kw, roc_ms, altitude_m)

        # Update history
        self._prev_t = t
        self._prev_rpm = rpm
        self._prev_map_pa = map_pa
        self._prev_cht_k = cht_k
        self._prev_alt_m = altitude_m

        if is_stable:
            if self._last_candidate_bin == candidate_bin:
                self._stable_duration_s += dt
            else:
                self._stable_duration_s = dt
                self._last_candidate_bin = candidate_bin

            # Check if stability window is satisfied
            if self._stable_duration_s >= self.min_stable_duration_s:
                return candidate_bin, self._stable_duration_s
            else:
                return BIN_TRANSIENT, self._stable_duration_s
        else:
            self._stable_duration_s = 0.0
            self._last_candidate_bin = None
            return BIN_TRANSIENT, 0.0


# ============================================================================
# 4. Aircraft Encounter Tracker n_i(r)
# ============================================================================

@dataclass
class AircraftExposureProfile:
    """Encounter history and quasi-steady flight exposure for a single aircraft."""
    aircraft_id: str
    exposure_seconds: Dict[int, float] = field(default_factory=lambda: {r: 0.0 for r in range(NUM_CORE_BINS)})
    last_updated_t: float = 0.0

    def add_exposure(self, regime_bin: int, duration_s: float, t: float = 0.0) -> None:
        """Accumulate quasi-steady seconds for a core regime bin."""
        if regime_bin in self.exposure_seconds:
            self.exposure_seconds[regime_bin] += max(0.0, float(duration_s))
            self.last_updated_t = max(self.last_updated_t, float(t))

    def get_exposure(self, regime_bin: int) -> float:
        """Get total quasi-steady seconds accumulated for a regime bin."""
        return self.exposure_seconds.get(regime_bin, 0.0)

    def compute_borrow_weight(self, regime_bin: int, target_seconds: float = 120.0) -> float:
        """
        Compute fleet borrow confidence weight v_i(theta_j, r) in [0.0, 1.0].
        If exposure >= target_seconds (120s), weight is 1.0 (fully autonomous/local).
        If exposure is low, weight approaches 0.0 (requires borrowing from fleet).
        """
        n_seconds = self.get_exposure(regime_bin)
        return float(min(1.0, max(0.0, n_seconds / max(target_seconds, 1e-6))))

    def to_dict(self) -> Dict[str, object]:
        return {
            "aircraft_id": self.aircraft_id,
            "regime_grid_version": REGIME_GRID_VERSION,
            "exposure_seconds": {str(k): float(v) for k, v in self.exposure_seconds.items()},
            "last_updated_t": float(self.last_updated_t),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> AircraftExposureProfile:
        exp_dict: Dict[int, float] = {
            int(k): float(v) for k, v in data.get("exposure_seconds", {}).items()  # type: ignore[union-attr]
        }
        for r in range(NUM_CORE_BINS):
            if r not in exp_dict:
                exp_dict[r] = 0.0
        return cls(
            aircraft_id=str(data.get("aircraft_id", "")),
            exposure_seconds=exp_dict,
            last_updated_t=float(data.get("last_updated_t", 0.0)),
        )


class EncounterTracker:
    """
    Fleet-wide encounter tracking manager.
    Tracks n_i(r) for all registered aircraft.
    """

    def __init__(self) -> None:
        self.profiles: Dict[str, AircraftExposureProfile] = {}

    def get_or_create_profile(self, aircraft_id: str) -> AircraftExposureProfile:
        """Retrieve or create exposure profile for an aircraft."""
        if aircraft_id not in self.profiles:
            self.profiles[aircraft_id] = AircraftExposureProfile(aircraft_id=aircraft_id)
        return self.profiles[aircraft_id]

    def record_step(
        self,
        aircraft_id: str,
        regime_bin: int,
        dt: float,
        stable_duration_s: float,
        t: float = 0.0,
    ) -> None:
        """
        Record a step. Only accumulates exposure if regime is a core bin and stable >= 20s.
        """
        if regime_bin in CORE_REGIME_BINS and stable_duration_s >= MIN_STABLE_DURATION_S:
            profile = self.get_or_create_profile(aircraft_id)
            profile.add_exposure(regime_bin, dt, t=t)

    def export_json(self) -> str:
        """Export all profiles to JSON."""
        data = {
            "regime_grid_version": REGIME_GRID_VERSION,
            "aircraft_count": len(self.profiles),
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()}
        }
        return json.dumps(data, indent=2)

    def load_json(self, json_str: str) -> None:
        """Load profiles from JSON."""
        data = json.loads(json_str)
        if data.get("regime_grid_version") != REGIME_GRID_VERSION:
            raise ValueError(
                f"Regime grid version mismatch: expected {REGIME_GRID_VERSION}, "
                f"got {data.get('regime_grid_version')}"
            )
        self.profiles.clear()
        for aid, p_data in data.get("profiles", {}).items():
            self.profiles[aid] = AircraftExposureProfile.from_dict(p_data)
