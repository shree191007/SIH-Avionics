"""
SIH26054 Replan to Learn: L6 physics-derived failure threshold solver.
Implements 04_ml_rul_mission_probe.md Sec 2.4:

    theta_j^fail = min{ theta_j : any operating limit is violated somewhere
                         in the required mission envelope, on an ISA+20 day }

Prior to this module, RUL failure thresholds were hardcoded guesses in
stage3.py's _init_rul_filters() (0.80, 0.70, 0.75, 1.30 -- round numbers
with no derivation), directly contradicting this section's explicit
requirement that the threshold be COMPUTED from the twin, not picked. This
solver replaces that with a real binary search against real Rotax 915iS
operator's-manual limits (model/rotax_915is_reference_data.json).

Limits evaluated, per spec:
  - CHT/coolant limit at max continuous power in climb
  - knock/misfire margin at the leanest commanded cruise mixture
    (approximated here via the real EGT limit, since this model has no
    knock model -- EGT is the directly-measurable proxy the real engine's
    own limits document uses)
  - minimum oil pressure at hot idle
  - minimum power for required rate of climb at maximum take-off weight
    (the required-climb-power VALUE is airframe-specific and not in the
    engine-only Rotax reference data available to this project -- marked
    [ASSUMED] below, unlike the other three limits which are real
    published Rotax numbers)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from replan_to_learn.contracts.telemetry import TelemetryFrame
from replan_to_learn.rul.datatypes import FailureThreshold

ISA_SEA_LEVEL_K = 288.15
ISA_LAPSE_RATE_K_PER_M = 0.0065
ISA_PLUS_20_K = 20.0


def _isa_plus20_temp(altitude_m: float) -> float:
    """ISA temperature at altitude, plus the +20C hot-day margin the spec's threshold definition requires."""
    return (ISA_SEA_LEVEL_K - ISA_LAPSE_RATE_K_PER_M * altitude_m) + ISA_PLUS_20_K


def _make_frame(t: float, n_rpm: float, map_pa: float, t_im: float, p_amb: float, t_amb: float, v_tas: float, h_p: float) -> TelemetryFrame:
    return TelemetryFrame(
        t=t, egt=(900.0, 900.0, 900.0, 900.0), cht=350.0, p_oil=3.0e5, t_oil=350.0,
        n_rpm=n_rpm, mdot_f=0.05, map_pa=map_pa, tps=0.9, t_im=t_im,
        p_amb=p_amb, t_amb=t_amb, v_tas=v_tas, h_p=h_p,
        valid_mask=0xFFFF, flight_id="", aircraft_id="", engine_id="",
    )


class StressOperatingPoint:
    """A real-limit stress scenario: a repeated-frame sequence long enough to equilibrate, plus the channel/limit it's evaluated against."""

    def __init__(self, name: str, frames: list, channel_index: int, limit_value: float, direction: str, description: str) -> None:
        self.name = name
        self.frames = frames
        self.channel_index = channel_index  # index into physics_twin.predict()'s y_hat: 0-3=egt,4=cht,5=p_oil,6=t_oil,7=mdot_f,8=p_brake_kw
        self.limit_value = limit_value
        self.direction = direction  # "max" (fail if predicted > limit) or "min" (fail if predicted < limit)
        self.description = description

    def violates(self, predicted: float) -> bool:
        if self.direction == "max":
            return predicted > self.limit_value
        return predicted < self.limit_value


def _build_stress_points(rotax: Dict[str, Any], n_equilibration_steps: int = 200, dt: float = 0.1) -> Dict[str, StressOperatingPoint]:
    """
    Builds the four spec-listed stress scenarios from real Rotax 915iS
    operator's-manual data. n_equilibration_steps=200 (20s) matches the
    real quasi-steady stabilization window used elsewhere in this
    codebase. This solver's whole purpose is finding where the TRUE
    steady-state prediction crosses a real limit, so under-equilibrating
    isn't a minor accuracy loss -- it changes the answer's magnitude
    directly. Confirmed empirically: an earlier 60-step (6s) version put
    theta_cool's failure threshold at 0.0139 (98.6% cooling loss before
    CHT reaches 120C), because CHT's sensitivity to theta_cool is still
    growing well past 6s (this session's gate work found it plateaus only
    around 20s -- see gate/identifiability.py's
    _REGIME_EQUILIBRATION_STEPS docstring) -- the 6s snapshot badly
    understated true steady-state CHT for every theta_cool value tested,
    systematically requiring far more degradation to trip the limit than
    reality would.
    """
    engine_limits_rpm = rotax["engine_speed_limits_rpm"]
    oil_limits = rotax["oil_limits"]
    coolant_limits = rotax["coolant_limits"]
    egt_limits = rotax["egt_limits"]

    # --- CHT/coolant limit at max continuous power in climb ---
    # Real anchor: max_continuous = 5500rpm / 1450mbar (power_anchors_ROTAX).
    # Climb = low airspeed (less ram-air cooling than cruise) at a moderate
    # altitude, hot day (ISA+20).
    climb_alt_m = 1500.0
    climb_t_amb = _isa_plus20_temp(climb_alt_m)
    cht_frames = [
        _make_frame(i * dt, n_rpm=5500.0, map_pa=145000.0, t_im=climb_t_amb + 15.0,
                    p_amb=84700.0, t_amb=climb_t_amb, v_tas=35.0, h_p=climb_alt_m)
        for i in range(n_equilibration_steps)
    ]
    cht_limit_k = coolant_limits["temperature_c"]["normal_max"] + 273.15

    # --- EGT limit at the leanest commanded cruise mixture (proxy for
    # knock/misfire margin -- no knock model in this twin) ---
    cruise_alt_m = 3000.0
    cruise_t_amb = _isa_plus20_temp(cruise_alt_m)
    egt_frames = [
        _make_frame(i * dt, n_rpm=5000.0, map_pa=120000.0, t_im=cruise_t_amb + 20.0,
                    p_amb=70000.0, t_amb=cruise_t_amb, v_tas=60.0, h_p=cruise_alt_m)
        for i in range(n_equilibration_steps)
    ]
    egt_limit_k = egt_limits["max_c"] + 273.15

    # --- Minimum oil pressure at hot idle ---
    idle_rpm = float(engine_limits_rpm["idle"])
    idle_frames = [
        _make_frame(i * dt, n_rpm=idle_rpm, map_pa=40000.0, t_im=climb_t_amb + 25.0,
                    p_amb=101325.0, t_amb=climb_t_amb, v_tas=0.0, h_p=0.0)
        for i in range(n_equilibration_steps)
    ]
    # Below 3500rpm the real minimum is 0.8 bar (oil_limits.pressure_bar.min_below_3500rpm).
    oil_p_limit_pa = oil_limits["pressure_bar"]["min_below_3500rpm"] * 1e5

    # --- Minimum power for required rate of climb at MTOW ---
    # [ASSUMED]: the required climb-power VALUE is airframe-specific
    # (depends on MTOW, wing loading, climb-rate requirement) and is not
    # published in the engine-only Rotax reference data available to this
    # project. Approximated as a documented fraction of the real
    # max-continuous power anchor (99kW at 5500rpm) rather than an
    # unrelated round number -- replace with a real airframe performance
    # requirement when one is available.
    power_frames = [
        _make_frame(i * dt, n_rpm=5500.0, map_pa=145000.0, t_im=climb_t_amb + 15.0,
                    p_amb=84700.0, t_amb=climb_t_amb, v_tas=35.0, h_p=climb_alt_m)
        for i in range(n_equilibration_steps)
    ]
    required_climb_power_frac = 0.60  # [ASSUMED]
    required_climb_power_kw = rotax["power_anchors_ROTAX"]["max_continuous"]["power_kw"] * required_climb_power_frac

    return {
        "cht": StressOperatingPoint(
            "CHT/coolant limit at max continuous power in climb", cht_frames, channel_index=4,
            limit_value=cht_limit_k, direction="max",
            description=f"CHT > {coolant_limits['temperature_c']['normal_max']}C (real Rotax coolant limit) at 5500rpm climb, ISA+20",
        ),
        "egt": StressOperatingPoint(
            "EGT (misfire/knock proxy) at leanest commanded cruise mixture", egt_frames, channel_index=0,
            limit_value=egt_limit_k, direction="max",
            description=f"EGT > {egt_limits['max_c']}C (real Rotax EGT limit) at lean cruise, ISA+20",
        ),
        "oil_pressure": StressOperatingPoint(
            "Minimum oil pressure at hot idle", idle_frames, channel_index=5,
            limit_value=oil_p_limit_pa, direction="min",
            description=f"p_oil < {oil_limits['pressure_bar']['min_below_3500rpm']}bar (real Rotax minimum below 3500rpm) at hot idle",
        ),
        "power": StressOperatingPoint(
            "Minimum power for required rate of climb at MTOW", power_frames, channel_index=8,
            limit_value=required_climb_power_kw, direction="min",
            description=(
                f"P_brake < {required_climb_power_kw:.1f}kW ([ASSUMED] {required_climb_power_frac:.0%} "
                f"of the real {rotax['power_anchors_ROTAX']['max_continuous']['power_kw']}kW max-continuous "
                "anchor -- real MTOW climb-power requirement not available) at 5500rpm climb, ISA+20"
            ),
        ),
    }


# Which stress scenario applies to each of the 9 multiplicative health
# parameters, based on which channel each ALGEBRAICALLY affects in
# physics_twin.py (established this session tracing the actual formulas:
# theta_vol/theta_fric affect P_brake but are algebraically absent from
# EGT; theta_comb affects EGT/P_brake but not CHT/mdot_f; theta_cool only
# affects CHT; theta_oilp only affects p_oil; injectors affect per-cylinder
# EGT). Direction ("below" nominal is a fault vs "above" nominal is a
# fault) matches physical meaning: less volumetric/combustion/cooling/oil-
# pump efficiency is a fault (theta -> 0 direction), more friction is a
# fault (theta -> +inf direction).
_PARAM_STRESS_POINT = {
    0: ("power", "below"),        # theta_vol
    1: ("egt", "below"),          # theta_comb
    2: ("cht", "below"),          # theta_cool
    3: ("egt", "below"),          # theta_inj1 (per-cylinder EGT, cylinder 0)
    4: ("egt", "below"),          # theta_inj2
    5: ("egt", "below"),          # theta_inj3
    6: ("egt", "below"),          # theta_inj4
    7: ("oil_pressure", "below"), # theta_oilp
    8: ("power", "above"),        # theta_fric
}

_PARAM_INJECTOR_CYLINDER = {3: 0, 4: 1, 5: 2, 6: 3}


def solve_failure_threshold(
    twin: Any,
    theta_index: int,
    theta_name: str,
    rotax_reference_path: Optional[Path] = None,
    search_tol: float = 0.005,
    max_iter: int = 30,
) -> FailureThreshold:
    """
    Binary-searches theta[theta_index] (holding all other parameters at
    nominal) for the value at which the twin's steady-state prediction, at
    the real-limit stress operating point relevant to that parameter,
    exactly crosses the real physical limit.
    """
    if theta_index not in _PARAM_STRESS_POINT:
        raise ValueError(f"No failure-threshold stress point defined for theta_index={theta_index}")

    if rotax_reference_path is None:
        rotax_reference_path = Path(twin.model_dir) / "rotax_915is_reference_data.json"
    rotax = json.loads(Path(rotax_reference_path).read_text())
    stress_points = _build_stress_points(rotax)

    point_key, direction = _PARAM_STRESS_POINT[theta_index]
    point = stress_points[point_key]
    channel_index = point.channel_index
    if theta_index in _PARAM_INJECTOR_CYLINDER:
        channel_index = _PARAM_INJECTOR_CYLINDER[theta_index]

    n_theta = twin.N_THETA if hasattr(twin, "N_THETA") else 15
    nominal = np.zeros(n_theta, dtype=np.float64)
    nominal[0:9] = 1.0

    def predicted_at(theta_val: float) -> float:
        theta = nominal.copy()
        theta[theta_index] = theta_val
        y = twin.predict(point.frames, theta)
        return float(y[-1, channel_index])

    if direction == "below":
        lo, hi = 0.10, 1.0
        # Ensure the search bracket actually contains a sign change: at
        # theta=1.0 (nominal) the limit must NOT be violated, and at the
        # low end it must be.
        if point.violates(predicted_at(hi)):
            raise RuntimeError(
                f"theta_index={theta_index}: real limit is already violated at NOMINAL "
                f"({point.description}) -- this stress operating point or limit needs revisiting, "
                "not a solvable failure threshold."
            )
        if not point.violates(predicted_at(lo)):
            lo = 0.01
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if point.violates(predicted_at(mid)):
                lo = mid
            else:
                hi = mid
            if hi - lo < search_tol:
                break
        threshold_value = hi
    else:
        lo, hi = 1.0, 3.0
        if point.violates(predicted_at(lo)):
            raise RuntimeError(
                f"theta_index={theta_index}: real limit is already violated at NOMINAL "
                f"({point.description}) -- this stress operating point or limit needs revisiting, "
                "not a solvable failure threshold."
            )
        if not point.violates(predicted_at(hi)):
            hi = 6.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if point.violates(predicted_at(mid)):
                hi = mid
            else:
                lo = mid
            if hi - lo < search_tol:
                break
        threshold_value = lo

    return FailureThreshold(
        theta_index=theta_index,
        theta_name=theta_name,
        threshold_value=float(threshold_value),
        limit_description=point.description,
        operating_point={
            "stress_scenario": point.name,
            "n_rpm": point.frames[-1].n_rpm,
            "map_pa": point.frames[-1].map_pa,
            "t_amb": point.frames[-1].t_amb,
            "v_tas": point.frames[-1].v_tas,
            "h_p": point.frames[-1].h_p,
        },
        direction=direction,
    )


def load_or_solve_failure_threshold(
    twin: Any,
    theta_index: int,
    theta_name: str,
    cache_path: Optional[Path] = None,
) -> FailureThreshold:
    """
    Loads a precomputed threshold from model/rul_failure_thresholds.json
    (see scripts/solve_rul_failure_thresholds.py) if present, falling back
    to a live solve otherwise. The 200-step-equilibrated binary search
    costs ~70s for all 9 parameters -- real but too slow to redo on every
    Stage3Orchestrator construction for a fixed property of the
    calibrated engine model, hence the cache.
    """
    if cache_path is None:
        model_dir = Path(getattr(twin, "model_dir", "model"))
        cache_path = model_dir / "rul_failure_thresholds.json"

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        entry = cached.get("thresholds", {}).get(str(theta_index))
        if entry is not None:
            return FailureThreshold(
                theta_index=theta_index,
                theta_name=entry.get("theta_name", theta_name),
                threshold_value=float(entry["threshold_value"]),
                limit_description=entry["limit_description"],
                operating_point=entry["operating_point"],
                direction=entry.get("direction", "below"),
            )

    return solve_failure_threshold(twin, theta_index, theta_name)
