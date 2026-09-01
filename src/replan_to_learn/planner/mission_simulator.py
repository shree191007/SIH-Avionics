"""
SIH26054 Replan to Learn: L7 Mission Simulator.
Simulates planned sorties using PhysicsTwin.predict with current theta estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from replan_to_learn.planner.datatypes import (
    CandidateManoeuvre,
    CostModel,
    FeasibilityResult,
    ManoeuvreType,
    MissionSimulationResult,
)

# ---------------------------------------------------------------------------
# ISA (International Standard Atmosphere) troposphere model. Standard ICAO
# constants (not fitted / not engine-specific) -- used to replace the
# previous fixed map_pa=101325.0 / t_amb=288.15 / p_amb=101325.0 that were
# applied to EVERY frame regardless of the altitude actually being
# simulated (a real physical inconsistency: climbing to 5000m did nothing
# to ambient pressure/temperature). Mirrors the lapse-rate convention
# already established in rul/failure_threshold_solver.py's
# _isa_plus20_temp, but WITHOUT the +20C hot-day margin -- that margin is
# specific to L6's worst-case failure-threshold search, not appropriate
# for L7's nominal-day mission simulation.
# ---------------------------------------------------------------------------
ISA_SEA_LEVEL_T_K = 288.15
ISA_SEA_LEVEL_P_PA = 101325.0
ISA_LAPSE_RATE_K_PER_M = 0.0065
# g0*M/(R*L) for dry air in the standard troposphere (ICAO standard
# atmosphere constant, not a fitted or assumed value).
ISA_PRESSURE_EXPONENT = 5.25588


def isa_temperature_k(altitude_m: float) -> float:
    """Standard-day ambient temperature (K) at altitude, ICAO troposphere model."""
    alt = max(0.0, float(altitude_m))
    return ISA_SEA_LEVEL_T_K - ISA_LAPSE_RATE_K_PER_M * alt


def isa_pressure_pa(altitude_m: float) -> float:
    """Standard-day ambient pressure (Pa) at altitude, ICAO troposphere model."""
    alt = max(0.0, float(altitude_m))
    ratio = max(1e-6, 1.0 - ISA_LAPSE_RATE_K_PER_M * alt / ISA_SEA_LEVEL_T_K)
    return ISA_SEA_LEVEL_P_PA * (ratio ** ISA_PRESSURE_EXPONENT)


# --- Manifold pressure / induction temperature estimate from RPM. ---------
# Real [ROTAX] anchors: idle=1800rpm (engine_speed_limits_rpm.idle),
# max_continuous=5500rpm at 1450mbar (manifold_pressure_targets.at_5500_rpm_mbar).
# The idle-MAP value (40000 Pa) is [ASSUMED] -- not published in the
# operator's manual -- but is kept numerically consistent with the same
# idle-MAP convention already used in rul/failure_threshold_solver.py's
# idle_frames (map_pa=40000.0), rather than inventing a new number.
ROTAX_IDLE_RPM = 1800.0  # [ROTAX]
ROTAX_MAX_CONTINUOUS_RPM = 5500.0  # [ROTAX]
ROTAX_MAX_CONTINUOUS_MAP_PA = 145000.0  # [ROTAX] 1450 mbar
ASSUMED_IDLE_MAP_PA = 40000.0  # [ASSUMED], matches rul/failure_threshold_solver.py idle_frames


def estimate_map_pa(n_rpm: float) -> float:
    """
    Linear MAP-vs-RPM interpolation between the real idle and max-continuous
    Rotax anchors. This is a coarse stand-in for the proprietary Rotax
    performance model (not available to this project -- see
    model/rotax_915is_reference_data.json's usage_notes), but it is real
    anchor-point-grounded rather than a flat constant independent of engine
    speed.
    """
    span = ROTAX_MAX_CONTINUOUS_RPM - ROTAX_IDLE_RPM
    frac = 0.0 if span <= 0 else (float(n_rpm) - ROTAX_IDLE_RPM) / span
    frac = min(1.0, max(0.0, frac))
    return ASSUMED_IDLE_MAP_PA + frac * (ROTAX_MAX_CONTINUOUS_MAP_PA - ASSUMED_IDLE_MAP_PA)


def estimate_t_im_k(t_amb_k: float, n_rpm: float) -> float:
    """
    Induction (post-intercooler) temperature estimate: ambient plus a
    compression/charge-heating offset that grows with power/RPM.
    [ASSUMED] linear range 15K (idle) - 25K (max continuous), chosen to
    match the order of magnitude of the +15/+20/+25 offsets already used
    for climb/cruise/idle stress points in rul/failure_threshold_solver.py's
    _build_stress_points, rather than an unrelated round number.
    """
    span = ROTAX_MAX_CONTINUOUS_RPM - ROTAX_IDLE_RPM
    frac = 0.0 if span <= 0 else (float(n_rpm) - ROTAX_IDLE_RPM) / span
    frac = min(1.0, max(0.0, frac))
    return t_amb_k + 15.0 + 10.0 * frac


@dataclass(frozen=True)
class MissionState:
    altitude_m: float
    ias_kt: float
    rpm: float
    fuel_kg: float
    elapsed_s: float
    regime: int


def frame_from_state(state: MissionState) -> Any:
    """
    Build a physically-consistent TelemetryFrame for a MissionState:
    ambient pressure/temperature from the ISA model at state.altitude_m,
    manifold pressure/induction temperature estimated from state.rpm --
    instead of the previous fixed map_pa=101325.0/t_amb=288.15/p_amb=101325.0
    applied identically regardless of altitude.
    """
    from replan_to_learn.contracts.telemetry import TelemetryFrame

    t_amb = isa_temperature_k(state.altitude_m)
    p_amb = isa_pressure_pa(state.altitude_m)
    map_pa = estimate_map_pa(state.rpm)
    t_im = estimate_t_im_k(t_amb, state.rpm)

    return TelemetryFrame(
        t=state.elapsed_s,
        egt=(950.0, 955.0, 960.0, 965.0),
        cht=360.0,
        p_oil=3.5e5,
        t_oil=350.0,
        n_rpm=state.rpm,
        mdot_f=0.05,
        map_pa=map_pa,
        tps=0.5,
        t_im=t_im,
        p_amb=p_amb,
        t_amb=t_amb,
        v_tas=state.ias_kt * 0.51444,
        h_p=state.altitude_m,
        valid_mask=0xFFFF,
        flight_id="",
        aircraft_id="",
        engine_id="",
    )


class MissionSimulator:
    """
    Physics-based mission simulator.
    Uses PhysicsTwin.predict to simulate sorties with current theta estimate.
    """

    def __init__(
        self,
        physics_twin: Any,
        base_fuel_kg: float = 50.0,
        reserve_fuel_kg: float = 10.0,
        return_margin_min: float = 15.0,
    ) -> None:
        self.physics_twin = physics_twin
        self.base_fuel_kg = base_fuel_kg
        self.reserve_fuel_kg = reserve_fuel_kg
        self.return_margin_min = return_margin_min

    def simulate(
        self,
        theta: np.ndarray,
        mission_profile: Sequence[MissionState],
        current_fuel_kg: float,
    ) -> MissionSimulationResult:
        """
        Simulate mission with given theta and profile.
        """
        margin_warnings = []
        alternatives = []

        try:
            frames = [frame_from_state(state) for state in mission_profile]

            if frames:
                y_hat = self.physics_twin.predict(frames, theta)
                if y_hat.ndim == 1:
                    y_hat = y_hat.reshape(1, -1)
                final_power = float(y_hat[-1, 8])
                final_cht = float(y_hat[-1, 4])
            else:
                final_power = 0.0
                final_cht = 360.0

            if final_cht > 420.0:
                margin_warnings.append("CHT exceeds safe limit during mission")
            if final_power > 110.0:
                margin_warnings.append("Power demand exceeds MCP")

        except Exception:
            margin_warnings.append("Simulation encountered numerical issues")

        if margin_warnings:
            risk = "HIGH" if len(margin_warnings) > 1 else "MEDIUM"
            recommended = "REPLAN: derate or reduce altitude"
            alternatives.append({"action": "derate", "rpm_delta": -150})
            alternatives.append({"action": "altitude_change", "delta_h_m": -500})
        else:
            risk = "LOW"
            recommended = "SAFE: proceed as planned"

        return MissionSimulationResult(
            risk=risk,
            risk_probability=0.05 if risk == "LOW" else 0.4,
            recommended_action=recommended,
            alternatives=alternatives,
            margin_warnings=tuple(margin_warnings),
        )

    def simulate_manoeuvre(
        self,
        theta: np.ndarray,
        base_profile: Sequence[MissionState],
        manoeuvre: CandidateManoeuvre,
        current_fuel_kg: float,
    ) -> MissionSimulationResult:
        """
        Simulate mission with a candidate manoeuvre inserted. Fuel burn
        along the manoeuvre is now integrated from the twin's own
        predicted mdot_f at each perturbed state, instead of the previous
        fabricated flat per-step deltas (0.5/0.3/0.4/0.2/0.1 kg/step with
        no physical basis).
        """
        profile = list(base_profile)
        insert_idx = min(len(profile) // 2, len(profile) - 1)
        duration = manoeuvre.estimated_duration_s
        params = manoeuvre.parameters
        n_steps = max(1, int(duration))
        dt_s = duration / n_steps if n_steps else 1.0

        if manoeuvre.manoeuvre_type == ManoeuvreType.STEP_CLIMB:
            delta_h = params.get("delta_h_m", 1000.0)
            for i in range(n_steps):
                if insert_idx + i < len(profile):
                    s = profile[insert_idx + i]
                    profile[insert_idx + i] = replace(
                        s,
                        altitude_m=s.altitude_m + delta_h * (i + 1) / n_steps,
                        rpm=s.rpm + 100,
                        regime=0,
                    )
        elif manoeuvre.manoeuvre_type == ManoeuvreType.POWER_DERATE:
            delta_n = params.get("delta_n_rpm", -150.0)
            for i in range(n_steps):
                if insert_idx + i < len(profile):
                    s = profile[insert_idx + i]
                    profile[insert_idx + i] = replace(s, rpm=s.rpm + delta_n, regime=3)
        elif manoeuvre.manoeuvre_type == ManoeuvreType.MIXTURE_ENRICHMENT:
            # The physics twin has no exogenous mixture/lambda input channel
            # (commanded lambda comes from an internal N/MAP lookup table,
            # not a settable TelemetryFrame field) -- so this manoeuvre
            # cannot change the twin's predicted trajectory beyond dwelling
            # at the current regime. Documented limitation, not silently
            # faked: it just holds regime for the dwell.
            for i in range(n_steps):
                if insert_idx + i < len(profile):
                    s = profile[insert_idx + i]
                    profile[insert_idx + i] = replace(s, regime=3)
        elif manoeuvre.manoeuvre_type == ManoeuvreType.AIRSPEED_STEP:
            delta_ias = params.get("delta_ias_kt", 10.0)
            for i in range(n_steps):
                if insert_idx + i < len(profile):
                    s = profile[insert_idx + i]
                    profile[insert_idx + i] = replace(s, ias_kt=s.ias_kt + delta_ias, regime=0)
        elif manoeuvre.manoeuvre_type == ManoeuvreType.DESCENT_SEGMENT:
            for i in range(n_steps):
                if insert_idx + i < len(profile):
                    s = profile[insert_idx + i]
                    profile[insert_idx + i] = replace(
                        s,
                        altitude_m=max(0.0, s.altitude_m - 200.0 * (i + 1) / n_steps),
                        rpm=s.rpm - 200,
                        regime=6,
                    )

        affected_idx = [insert_idx + i for i in range(n_steps) if insert_idx + i < len(profile)]
        if affected_idx:
            frames = [frame_from_state(profile[idx]) for idx in affected_idx]
            try:
                y_hat = self.physics_twin.predict(frames, theta)
                if y_hat.ndim == 1:
                    y_hat = y_hat.reshape(1, -1)
                mdot_f = np.asarray(y_hat[:, 7], dtype=np.float64)
            except Exception:
                # [ASSUMED] fallback nominal cruise fuel flow (kg/s) if the
                # twin cannot be evaluated (e.g. degenerate mock in tests) --
                # only used when prediction genuinely fails, not the normal path.
                mdot_f = np.full(len(affected_idx), 0.02, dtype=np.float64)

            fuel = current_fuel_kg
            for k, idx in enumerate(affected_idx):
                fuel = max(0.0, fuel - float(mdot_f[k]) * dt_s)
                profile[idx] = replace(profile[idx], fuel_kg=fuel)

        return self.simulate(theta, profile, current_fuel_kg)
