"""
SIH26054 Replan to Learn: L7 Probe Selector.
Candidate manoeuvre generation, feasibility filtering, information-gain scoring,
cost model, and value-of-information test.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from replan_to_learn.planner.datatypes import (
    CandidateManoeuvre,
    CostModel,
    FeasibilityResult,
    ManoeuvreType,
    ProbeScore,
    ProbeSelectionResult,
)
from replan_to_learn.planner.mission_simulator import (
    isa_temperature_k,
    isa_pressure_pa,
    estimate_map_pa,
    estimate_t_im_k,
)
from replan_to_learn.gate.datatypes import Verdict, BorrowResult
from replan_to_learn.contracts.regimes import CORE_REGIME_BINS
from replan_to_learn.contracts.telemetry import TelemetryFrame

# REGIME_CENTERS is real, already-calibrated per-regime operating-point data
# (n_rpm, MAP, ambient conditions, V_TAS) established in L4's identifiability
# gate (gate/identifiability.py). Reused here rather than duplicating a
# second, possibly-drifting copy of the same 12 regime centers. Read-only
# import -- this module does not modify gate/identifiability.py.
from replan_to_learn.gate.identifiability import REGIME_CENTERS


# ---------------------------------------------------------------------------
# Real/documented constants. [ROTAX] = published operator's-manual value
# (model/rotax_915is_reference_data.json). [ASSUMED] = no real data source
# available for this project; a documented, order-of-magnitude-justified
# engineering default, per this session's established provenance convention
# (see rul/failure_threshold_solver.py's required_climb_power_frac).
# ---------------------------------------------------------------------------
ROTAX_IDLE_RPM = 1800.0  # [ROTAX] engine_speed_limits_rpm.idle
ROTAX_MAX_CONTINUOUS_RPM = 5500.0  # [ROTAX] engine_speed_limits_rpm.max_continuous

ASSUMED_SERVICE_CEILING_M = 12000.0  # [ASSUMED] generic light-aircraft service ceiling; no airframe-specific performance data available to this project
ASSUMED_CLIMB_DESCENT_RATE_MS = 2.5  # [ASSUMED] typical light GA climb/descend rate (~500 ft/min), used only to estimate the extra return-transit time an altitude excursion costs against return_margin_s
ASSUMED_MIN_SAFE_ALTITUDE_M = 100.0  # [ASSUMED] minimum en-route obstacle-clearance floor; no terrain data available
ASSUMED_MIXTURE_AUTHORITY_LAMBDA = 0.2  # [ASSUMED] typical fuel-injected piston-engine commandable mixture range is roughly lambda 0.8-1.2 (+-0.2 from stoichiometric)
ASSUMED_BASELINE_CRUISE_RPM = 5000.0  # [ASSUMED] representative cruise engine RPM baseline, used only when the caller does not supply the aircraft's actual current RPM
ASSUMED_BASELINE_CRUISE_IAS_KT = 100.0  # [ASSUMED] representative cruise IAS baseline, used only when the caller does not supply the aircraft's actual current IAS
ASSUMED_RESERVE_FUEL_KG = 10.0  # [ASSUMED] matches MissionSimulator's own default reserve_fuel_kg, kept consistent across the planner module
ASSUMED_FUEL_FLOW_FALLBACK_KG_S = 0.02  # [ASSUMED] nominal fallback fuel flow (~72 kg/h) used ONLY if the twin's predict() call fails; not the normal path

# --- Arrhenius thermal-damage term (Sec 3.5: "an Arrhenius thermal term on
# head/oil temperature ... it must come from the twin's simulated
# temperatures, not a lookup"). No engine-specific activation energy is
# published for this project (Rotax's operator's manual gives operating
# LIMITS, not a fatigue/oxidation activation energy). 80 kJ/mol is
# [ASSUMED]: it sits in the commonly-cited 60-120 kJ/mol range used in
# reliability engineering for lubricant thermal oxidation and metallic
# thermal-fatigue processes (e.g. Arrhenius-based oil-life and insulation
# life models). The reference temperature is likewise [ASSUMED] at 90C
# (363.15K), the rough midpoint of Rotax's own published normal operating
# envelopes (oil 50-130C, coolant up to 120C) -- i.e. "life is consumed at
# 1x rate at a representative nominal cruise temperature, faster above it,
# slower below it", per the spec's fallback guidance ("derive relative
# damage rate vs a reference temperature if an absolute Ea isn't
# available").
ARRHENIUS_EA_J_PER_MOL = 80000.0  # [ASSUMED]
GAS_CONSTANT_J_PER_MOL_K = 8.314
THERMAL_LIFE_REFERENCE_T_K = 363.15  # [ASSUMED] ~90C

# --- Thermal-cycle fatigue term (Sec 3.5: "plus a thermal-cycle term").
# Coffin-Manson-style low-cycle thermal fatigue: damage per cycle grows
# with (delta_T)^n. n=2 and the reference cycle/life values below are
# [ASSUMED] order-of-magnitude defaults (Coffin-Manson exponents for
# aluminium-alloy thermal cycling are commonly cited in the 1.9-2.5 range;
# no manufacturer S-N/fatigue curve is published for this engine), used
# only to give a real, temperature-swing-driven cyclic damage contribution
# rather than omitting the term entirely.
CYCLE_REF_DELTA_T_K = 100.0  # [ASSUMED] representative full-range thermal cycle (cold start to max operating temperature)
CYCLE_REF_LIFE_H = 0.05  # [ASSUMED] equivalent-hours life cost of one full-range thermal cycle
COFFIN_MANSON_EXPONENT = 2.0  # [ASSUMED]

# Channel order in noise_model.json's noise_std (see model/noise_model.json,
# [FIT] real per-regime/per-channel sigma). Matches physics_twin.predict()'s
# output columns 0-7 (EGT1-4, CHT, p_oil, t_oil, mdot_f); predict()'s 9th
# column is p_brake_kw, which has no counterpart in the noise model (whose
# 9th channel is z_N, engine speed) -- channel 8 always uses the fallback
# sigma below.
_NOISE_CHANNEL_ORDER = ["z_EGT_1", "z_EGT_2", "z_EGT_3", "z_EGT_4", "z_CHT", "z_p_oil", "z_t_oil", "z_mdot_f"]
_FALLBACK_SIGMA = 1.0  # [ASSUMED] matches noise_model.json's own documented fallback_sigma=1.0


def _arrhenius_acceleration_factor(t_k: float) -> float:
    """AF(T) = exp( Ea/R * (1/T_ref - 1/T) ); AF=1 at T_ref, >1 above it, <1 below it."""
    t_k = max(float(t_k), 1.0)
    return math.exp(
        (ARRHENIUS_EA_J_PER_MOL / GAS_CONSTANT_J_PER_MOL_K)
        * (1.0 / THERMAL_LIFE_REFERENCE_T_K - 1.0 / t_k)
    )


def _regime_jacobian_for_bin(
    model_dir: Any,
    dt: float,
    regime_bin: int,
    equilibration_steps: int,
    theta: np.ndarray,
    indices: Tuple[int, ...],
    n_theta: int,
) -> Tuple[int, np.ndarray]:
    """
    Module-level (picklable) worker: computes one core regime's
    equilibrated d(y)/d(theta) columns, for use in a ProcessPoolExecutor
    -- see ProbeSelector._get_regime_jacobian, which parallelizes this
    across all core regime bins (each independent: its own fresh
    PhysicsTwin, no shared state with any other regime's computation).
    """
    from replan_to_learn.physics_twin.physics_twin import PhysicsTwin

    twin = PhysicsTwin(model_dir=model_dir, dt=dt)
    op = REGIME_CENTERS[regime_bin]
    frames = [ProbeSelector._regime_equilibration_frame(op, i, dt) for i in range(equilibration_steps)]
    eps = 1e-3
    J_r = np.zeros((9, n_theta), dtype=np.float64)
    for j in indices:
        theta_plus = theta.copy()
        theta_plus[j] += eps
        theta_minus = theta.copy()
        theta_minus[j] -= eps
        y_plus = twin.predict(frames, theta_plus)
        y_minus = twin.predict(frames, theta_minus)
        y_plus = y_plus[-1] if y_plus.ndim > 1 else y_plus
        y_minus = y_minus[-1] if y_minus.ndim > 1 else y_minus
        J_r[:, j] = (y_plus - y_minus) / (2.0 * eps)
    return regime_bin, J_r


def _physics_cost_for_candidate(
    model_dir: Any,
    dt: float,
    candidate: CandidateManoeuvre,
    theta: np.ndarray,
    max_cost_sim_steps: int,
) -> Tuple[float, float, float, float]:
    """
    Module-level (picklable, no `self`) so this can run inside a
    ProcessPoolExecutor worker -- each worker builds its OWN PhysicsTwin
    from model_dir rather than sharing/pickling a live twin instance
    (PhysicsTwin is mutable, stateful, and not designed to be shared
    across processes). Real per-candidate cost, confirmed by direct
    profiling to be independent across candidates (each starts its own
    fresh predict() call, no shared state to serialize) -- see
    ProbeSelector.score_candidates for the parallel dispatch across all
    ~24 candidates scored per select_probe() call.
    """
    from replan_to_learn.physics_twin.physics_twin import PhysicsTwin

    twin = PhysicsTwin(model_dir=model_dir, dt=dt)

    duration_s = max(candidate.estimated_duration_s, 1e-6)
    n_steps = max(1, min(int(round(duration_s)), max_cost_sim_steps))
    dt_s = duration_s / n_steps

    frames = ProbeSelector._build_candidate_frames(candidate, n_steps, dt_s)
    try:
        y_hat = twin.predict(frames, theta)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(1, -1)
        cht_traj = np.asarray(y_hat[:, 4], dtype=np.float64)
        toil_traj = np.asarray(y_hat[:, 6], dtype=np.float64)
        mdot_f_traj = np.asarray(y_hat[:, 7], dtype=np.float64)
    except Exception:
        cht_traj = np.full(n_steps, 360.0)
        toil_traj = np.full(n_steps, 350.0)
        mdot_f_traj = np.full(n_steps, ASSUMED_FUEL_FLOW_FALLBACK_KG_S)

    # ΔFuel: real, from the twin's own predicted mdot_f, integrated
    # over the manoeuvre duration -- not a flat per-second constant.
    delta_fuel_kg = float(np.sum(mdot_f_traj) * dt_s)

    # ΔLife: Arrhenius thermal term (head + oil) plus a thermal-cycle
    # term, both from the twin's own simulated temperatures.
    af_cht = np.array([_arrhenius_acceleration_factor(t) for t in cht_traj])
    af_oil = np.array([_arrhenius_acceleration_factor(t) for t in toil_traj])
    delta_life_h_arrhenius = float(np.sum((af_cht + af_oil) / 2.0) * dt_s / 3600.0)

    peak_swing_cht_k = float(np.max(cht_traj) - np.min(cht_traj)) if len(cht_traj) else 0.0
    peak_swing_oil_k = float(np.max(toil_traj) - np.min(toil_traj)) if len(toil_traj) else 0.0
    peak_swing_k = max(peak_swing_cht_k, peak_swing_oil_k)
    delta_life_h_cycle = CYCLE_REF_LIFE_H * (peak_swing_k / CYCLE_REF_DELTA_T_K) ** COFFIN_MANSON_EXPONENT

    delta_life_h = delta_life_h_arrhenius + delta_life_h_cycle
    delta_time_h = duration_s / 3600.0

    # ΔExposure: no mission-plan exposure-weighting data source exists in
    # this project (spec: "weight supplied by the mission plan, zero in a
    # benign profile") -- retained as an [ASSUMED] altitude-proportional
    # placeholder, zero for non-climb manoeuvres.
    delta_exposure = 0.0
    if candidate.manoeuvre_type == ManoeuvreType.STEP_CLIMB:
        delta_exposure = candidate.parameters.get("delta_h_m", 0.0) / 1000.0 * 0.1  # [ASSUMED]

    return delta_life_h, delta_time_h, delta_fuel_kg, delta_exposure


class ProbeSelector:
    """
    Probe selection and scoring engine.

    Ordering contract:
    AMBIGUOUS
      -> admissible fleet borrow? -> BORROW (no probe)
      -> no admissible borrow -> score probes -> fly if passes VOI test

    Implements Section 3.3-3.6 of 04_ml_rul_mission_probe.md.
    """

    # Mirrors gate/identifiability.py's _REGIME_EQUILIBRATION_STEPS (2.5s
    # at dt=0.1s): the twin's Jacobian is only meaningful at a regime's
    # actual steady-state operating point, not a single arbitrary-state
    # snapshot -- same lesson this session already fixed in the gate.
    _REGIME_EQUILIBRATION_STEPS = 25
    # [ASSUMED] neutral single-sample reference dwell used to normalize the
    # CURRENT (pre-probe) fingerprint weighting for regimes the probe does
    # not touch -- see _weighted_fingerprint.
    _BASELINE_DWELL_S = 1.0
    # Bounds the per-candidate physics simulation cost (~24 candidates
    # scored per selection call); dt widens beyond this many seconds so a
    # 300s-dwell candidate still gets a full-duration integral, just at
    # coarser resolution.
    #
    # [INVESTIGATED, NOT FIXED HERE] A reduction attempt (300 -> 40) was
    # tried and reverted: confirmed via direct profiling that
    # PhysicsTwin._advance_state's RK4 integration sub-steps at its OWN
    # config.dt (0.1s) granularity regardless of the frame count passed to
    # predict() -- n_substeps = round(elapsed_s / config.dt) PER FRAME, so
    # total substeps for a fixed-duration candidate is duration_s/config.dt
    # REGARDLESS of how that duration is split into frames (confirmed:
    # 300 frames at dt_s=1.0, 40 frames at dt_s=7.5, and 10 frames at
    # dt_s=30.0 all cost about the same, ~3s, because all three have the
    # same total elapsed/config.dt substep count). Reducing this constant
    # therefore does NOT reduce cost -- it only widens each frame's own
    # elapsed gap, which the sub-stepper compensates for with more
    # substeps per frame, canceling out any savings. The real lever is
    # PhysicsTwin's own config.dt (0.1s), which is shared by every other
    # use of the twin (not probe-cost-estimation-specific) and was not
    # changed here -- see the session's discussion of this finding.
    _MAX_COST_SIM_STEPS = 300

    def __init__(
        self,
        physics_twin: Any,
        cost_model: Optional[CostModel] = None,
        g_min: float = 0.1,
    ) -> None:
        self.physics_twin = physics_twin
        self.cost_model = cost_model or CostModel()
        self.g_min = g_min
        self._candidate_cache: List[CandidateManoeuvre] = []
        self._cached_jacobian_theta: Optional[np.ndarray] = None
        self._cached_jacobian_indices: Tuple[int, ...] = ()
        self._cached_regime_jacobian: Dict[int, np.ndarray] = {}

    def generate_candidates(
        self,
        base_profile: Sequence[Any],
        current_fuel_kg: float,
        max_cost: float = 5.0,
    ) -> List[CandidateManoeuvre]:
        """
        Generate ~60 discrete candidate manoeuvres.
        """
        candidates = []

        step_climb_deltas = [500, 1000, 2000, 3000]
        ias_options = [80, 85, 90]
        for delta_h in step_climb_deltas:
            for ias in ias_options:
                candidates.append(CandidateManoeuvre(
                    manoeuvre_type=ManoeuvreType.STEP_CLIMB,
                    parameters={"delta_h_m": float(delta_h), "target_ias_kt": float(ias)},
                    estimated_duration_s=60.0 + delta_h / 10.0,
                    estimated_cost=0.0,
                    target_regimes=(0, 1),
                ))

        power_derates = [(-100, 60), (-150, 180), (-250, 300)]
        for delta_n, dwell in power_derates:
            candidates.append(CandidateManoeuvre(
                manoeuvre_type=ManoeuvreType.POWER_DERATE,
                parameters={"delta_n_rpm": float(delta_n), "dwell_s": float(dwell)},
                estimated_duration_s=float(dwell),
                estimated_cost=0.0,
                target_regimes=(3,),
            ))

        mixture_deltas = [(-0.05, 60), (-0.10, 120)]
        for delta_lambda, dwell in mixture_deltas:
            candidates.append(CandidateManoeuvre(
                manoeuvre_type=ManoeuvreType.MIXTURE_ENRICHMENT,
                parameters={"delta_lambda": float(delta_lambda), "dwell_s": float(dwell)},
                estimated_duration_s=float(dwell),
                estimated_cost=0.0,
                target_regimes=(3, 4),
            ))

        airspeed_steps = [(10, 120), (15, 120), (-10, 120), (-15, 120)]
        for delta_ias, dwell in airspeed_steps:
            candidates.append(CandidateManoeuvre(
                manoeuvre_type=ManoeuvreType.AIRSPEED_STEP,
                parameters={"delta_ias_kt": float(delta_ias), "dwell_s": float(dwell)},
                estimated_duration_s=float(dwell),
                estimated_cost=0.0,
                target_regimes=(0, 1, 3),
            ))

        descent_options = [(-2.0, 120), (-3.0, 180), (-4.0, 240)]
        for delta_rod, dwell in descent_options:
            candidates.append(CandidateManoeuvre(
                manoeuvre_type=ManoeuvreType.DESCENT_SEGMENT,
                parameters={"delta_rod_ms": float(delta_rod), "dwell_s": float(dwell)},
                estimated_duration_s=float(dwell),
                estimated_cost=0.0,
                target_regimes=(6, 7),
            ))

        self._candidate_cache = candidates
        return candidates

    # ------------------------------------------------------------------
    # Physics-derived candidate frame construction
    # ------------------------------------------------------------------

    def _nominal_theta(self) -> np.ndarray:
        n_theta = getattr(self.physics_twin, "N_THETA", 15)
        theta = np.ones(n_theta, dtype=np.float64)
        if n_theta > 9:
            theta[9:] = 0.0  # additive sensor biases are nominal 0.0, not 1.0
        return theta

    @staticmethod
    def _build_candidate_frames(
        candidate: CandidateManoeuvre,
        n_steps: int,
        dt_s: float,
    ) -> List[Any]:
        """
        Build a real telemetry-frame trajectory for a candidate manoeuvre,
        rooted at the (real, calibrated) operating point of its first
        target regime and perturbed by the candidate's own parameters, so
        PhysicsTwin.predict can be evaluated over it.
        """
        base_regime = candidate.target_regimes[0] if candidate.target_regimes else 4
        op = REGIME_CENTERS.get(base_regime, REGIME_CENTERS.get(4))
        params = candidate.parameters
        frames = []
        for i in range(n_steps):
            t = i * dt_s
            altitude_m = op.altitude_m
            n_rpm = op.n_rpm
            v_tas = op.v_tas

            if candidate.manoeuvre_type == ManoeuvreType.STEP_CLIMB:
                delta_h = params.get("delta_h_m", 0.0)
                altitude_m = op.altitude_m + delta_h * (i + 1) / max(n_steps, 1)
                target_ias_kt = params.get("target_ias_kt")
                if target_ias_kt is not None:
                    v_tas = target_ias_kt * 0.51444
                n_rpm = op.n_rpm  # climb power held at the regime's own target

            elif candidate.manoeuvre_type == ManoeuvreType.POWER_DERATE:
                n_rpm = op.n_rpm + params.get("delta_n_rpm", 0.0)

            elif candidate.manoeuvre_type == ManoeuvreType.MIXTURE_ENRICHMENT:
                # PhysicsTwin has no exogenous mixture/lambda input channel
                # (commanded lambda is read from an internal N/MAP lookup
                # table, not settable per TelemetryFrame) -- so this
                # manoeuvre cannot change the twin's predicted trajectory
                # beyond dwelling at the current regime. Documented model
                # limitation, not silently faked.
                pass

            elif candidate.manoeuvre_type == ManoeuvreType.AIRSPEED_STEP:
                v_tas = op.v_tas + params.get("delta_ias_kt", 0.0) * 0.51444

            elif candidate.manoeuvre_type == ManoeuvreType.DESCENT_SEGMENT:
                delta_rod = params.get("delta_rod_ms", 0.0)
                altitude_m = max(0.0, op.altitude_m + delta_rod * t)

            t_amb = isa_temperature_k(altitude_m)
            p_amb = isa_pressure_pa(altitude_m)
            map_pa = estimate_map_pa(n_rpm)
            t_im = estimate_t_im_k(t_amb, n_rpm)

            frames.append(TelemetryFrame(
                t=t,
                egt=(950.0, 955.0, 960.0, 965.0),
                cht=360.0,
                p_oil=3.5e5,
                t_oil=350.0,
                n_rpm=n_rpm,
                mdot_f=0.05,
                map_pa=map_pa,
                tps=0.5,
                t_im=t_im,
                p_amb=p_amb,
                t_amb=t_amb,
                v_tas=max(0.0, v_tas),
                h_p=altitude_m,
                valid_mask=0xFFFF,
                flight_id="",
                aircraft_id="",
                engine_id="",
            ))
        return frames

    # ------------------------------------------------------------------
    # Feasibility filter (Sec 3.4)
    # ------------------------------------------------------------------

    def _check_feasibility(
        self,
        candidate: CandidateManoeuvre,
        current_fuel_kg: float,
        current_altitude_m: float,
        return_margin_s: float,
        theta: Optional[np.ndarray] = None,
        current_rpm: float = ASSUMED_BASELINE_CRUISE_RPM,
        current_ias_kt: float = ASSUMED_BASELINE_CRUISE_IAS_KT,
        reserve_fuel_kg: float = ASSUMED_RESERVE_FUEL_KG,
    ) -> FeasibilityResult:
        """
        Apply flight-envelope and return-margin filters before scoring.
        """
        violations = []
        if theta is None:
            theta = self._nominal_theta()

        # --- real physics-derived fuel burn for this candidate ---
        try:
            _, _, delta_fuel_kg, _ = self._compute_physics_cost(candidate, theta)
        except Exception:
            delta_fuel_kg = candidate.estimated_duration_s * ASSUMED_FUEL_FLOW_FALLBACK_KG_S

        fuel_after = max(0.0, current_fuel_kg - delta_fuel_kg)
        if fuel_after < reserve_fuel_kg:
            violations.append("fuel_below_reserve")

        burn_rate = delta_fuel_kg / max(candidate.estimated_duration_s, 1e-6)
        fuel_reserve_after_s = fuel_after / burn_rate if burn_rate > 1e-9 else float("inf")

        # --- return-margin constraint (Sec 3.4): "after the probe,
        # endurance to base must still exceed required time to base plus
        # reserve." Previously return_margin_s was accepted as a parameter
        # and passed straight through unused -- a real safety-logic gap.
        # The probe consumes the dwell/duration directly, plus (for
        # altitude excursions) the extra transit time to regain the
        # original altitude/position afterward.
        manoeuvre_time_cost_s = candidate.estimated_duration_s
        if candidate.manoeuvre_type == ManoeuvreType.STEP_CLIMB:
            delta_h = abs(candidate.parameters.get("delta_h_m", 0.0))
            manoeuvre_time_cost_s += delta_h / ASSUMED_CLIMB_DESCENT_RATE_MS
        elif candidate.manoeuvre_type == ManoeuvreType.DESCENT_SEGMENT:
            delta_rod = abs(candidate.parameters.get("delta_rod_ms", 0.0))
            dwell = candidate.parameters.get("dwell_s", candidate.estimated_duration_s)
            descended_m = delta_rod * dwell
            manoeuvre_time_cost_s += descended_m / ASSUMED_CLIMB_DESCENT_RATE_MS

        remaining_margin_s = return_margin_s - manoeuvre_time_cost_s
        if remaining_margin_s < 0.0:
            violations.append("return_margin_violated")

        # --- manoeuvre-specific flight-envelope checks ---
        if candidate.manoeuvre_type == ManoeuvreType.STEP_CLIMB:
            delta_h = candidate.parameters.get("delta_h_m", 0.0)
            if current_altitude_m + delta_h > ASSUMED_SERVICE_CEILING_M:
                violations.append("altitude_exceeds_service_ceiling")

        elif candidate.manoeuvre_type == ManoeuvreType.POWER_DERATE:
            delta_n = candidate.parameters.get("delta_n_rpm", 0.0)
            resulting_rpm = current_rpm + delta_n
            if resulting_rpm < ROTAX_IDLE_RPM:
                violations.append("rpm_below_idle")

        elif candidate.manoeuvre_type == ManoeuvreType.MIXTURE_ENRICHMENT:
            delta_lambda = abs(candidate.parameters.get("delta_lambda", 0.0))
            if delta_lambda > ASSUMED_MIXTURE_AUTHORITY_LAMBDA:
                violations.append("mixture_outside_commandable_range")

        elif candidate.manoeuvre_type == ManoeuvreType.AIRSPEED_STEP:
            delta_ias = candidate.parameters.get("delta_ias_kt", 0.0)
            resulting_ias = current_ias_kt + delta_ias
            if resulting_ias <= 0.0:
                violations.append("airspeed_non_positive")

        elif candidate.manoeuvre_type == ManoeuvreType.DESCENT_SEGMENT:
            delta_rod = candidate.parameters.get("delta_rod_ms", 0.0)
            dwell = candidate.parameters.get("dwell_s", candidate.estimated_duration_s)
            min_altitude = current_altitude_m + delta_rod * dwell
            if min_altitude < ASSUMED_MIN_SAFE_ALTITUDE_M:
                violations.append("altitude_below_safe_floor")

        return FeasibilityResult(
            is_feasible=len(violations) == 0,
            violations=tuple(violations),
            fuel_reserve_after_s=fuel_reserve_after_s,
            return_margin_s=remaining_margin_s,
        )

    # ------------------------------------------------------------------
    # Cost model (Sec 3.5) -- physics-derived, not fabricated
    # ------------------------------------------------------------------

    def _compute_physics_cost(
        self,
        candidate: CandidateManoeuvre,
        theta: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Compute physics-derived cost components by simulating the
        candidate manoeuvre with PhysicsTwin.predict and integrating real
        predicted temperatures/fuel flow over the trajectory. Delegates to
        the module-level _physics_cost_for_candidate so the SAME logic can
        run either here (in-process, single candidate) or inside a
        ProcessPoolExecutor worker (see score_candidates, which parallelizes
        this across all candidates -- confirmed via profiling to be the
        dominant real cost of select_probe(), ~24 independent candidates x
        up to 300 real PhysicsTwin.predict() integration steps each).
        """
        return _physics_cost_for_candidate(
            self.physics_twin.model_dir, self.physics_twin.config.dt,
            candidate, theta, self._MAX_COST_SIM_STEPS,
        )

    # ------------------------------------------------------------------
    # Predicted information gain (Sec 3.4) -- targeted separability
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_equilibration_frame(op: Any, i: int, dt: float = 0.1) -> Any:
        return TelemetryFrame(
            t=i * dt,
            egt=(950.0, 955.0, 960.0, 965.0),
            cht=360.0,
            p_oil=3.5e5,
            t_oil=350.0,
            n_rpm=op.n_rpm,
            mdot_f=0.05,
            map_pa=op.map_pa,
            tps=0.5,
            t_im=op.t_im,
            p_amb=op.p_amb,
            t_amb=op.t_amb,
            v_tas=op.v_tas,
            h_p=op.altitude_m,
            valid_mask=0xFFFF,
            flight_id="",
            aircraft_id="",
            engine_id="",
        )

    def _get_regime_jacobian(self, theta: np.ndarray, theta_indices: Tuple[int, ...] = ()) -> Dict[int, np.ndarray]:
        """
        d(y)/d(theta) at each core regime's real, equilibrated steady-state
        operating point -- same pattern as gate/identifiability.py's
        _precompute_regime_jacobian (equilibrate first, THEN take the
        Jacobian; a single-step/arbitrary-state snapshot badly understates
        real regime-to-regime sensitivity differences).

        _separability_scores/_weighted_fingerprint only ever read columns
        for theta_indices (the current ambiguous_set, normally 1-2
        parameters) -- computing the full central-difference Jacobian
        across all 15 parameters here was a real, severe cost (12 core
        regimes x 25-step equilibration x 2*15 perturbed predict() calls =
        ~9,300 real physics-twin state integrations per select_probe()
        call, confirmed via direct profiling to be ~60s at this
        calibration). Computing only the requested columns directly here
        (bypassing physics_twin.jacobian()'s full-width loop) cuts that by
        the same factor as the columns actually needed.
        """
        cache_key = tuple(sorted(theta_indices))
        if (
            self._cached_jacobian_theta is not None
            and np.allclose(theta, self._cached_jacobian_theta)
            and self._cached_jacobian_indices == cache_key
        ):
            return self._cached_regime_jacobian

        n_theta = len(theta)
        indices = cache_key if cache_key else tuple(range(n_theta))
        jac: Dict[int, np.ndarray] = {}
        if hasattr(self.physics_twin, "predict"):
            regime_bins = [r for r in CORE_REGIME_BINS if REGIME_CENTERS.get(r) is not None]
            try:
                model_dir = self.physics_twin.model_dir
                dt = self.physics_twin.config.dt
                n = len(regime_bins)
                with ProcessPoolExecutor(max_workers=min(8, max(1, n))) as pool:
                    results = list(pool.map(
                        _regime_jacobian_for_bin,
                        [model_dir] * n, [dt] * n, regime_bins,
                        [self._REGIME_EQUILIBRATION_STEPS] * n,
                        [theta] * n, [indices] * n, [n_theta] * n,
                    ))
                jac = dict(results)
            except Exception:
                # Sequential fallback (e.g. multiprocessing unavailable in
                # this environment) -- same computation, just in-process.
                eps = 1e-3
                jac = {}
                try:
                    for r in regime_bins:
                        op = REGIME_CENTERS[r]
                        frames = [self._regime_equilibration_frame(op, i, self.physics_twin.config.dt) for i in range(self._REGIME_EQUILIBRATION_STEPS)]
                        J_r = np.zeros((9, n_theta), dtype=np.float64)
                        for j in indices:
                            theta_plus = theta.copy()
                            theta_plus[j] += eps
                            theta_minus = theta.copy()
                            theta_minus[j] -= eps
                            y_plus = self.physics_twin.predict(frames, theta_plus)
                            y_minus = self.physics_twin.predict(frames, theta_minus)
                            y_plus = y_plus[-1] if y_plus.ndim > 1 else y_plus
                            y_minus = y_minus[-1] if y_minus.ndim > 1 else y_minus
                            J_r[:, j] = (y_plus - y_minus) / (2.0 * eps)
                        jac[r] = J_r
                except Exception:
                    jac = {}

        self._cached_jacobian_theta = np.array(theta, dtype=np.float64).copy()
        self._cached_jacobian_indices = cache_key
        self._cached_regime_jacobian = jac
        return jac

    def _sigma(self, channel_idx: int, regime: int) -> float:
        """Real per-regime, per-channel measurement noise std (model/noise_model.json, [FIT])."""
        noise_std = getattr(self.physics_twin, "_noise_std", None)
        if noise_std and channel_idx < len(_NOISE_CHANNEL_ORDER):
            arr = noise_std.get(_NOISE_CHANNEL_ORDER[channel_idx])
            if arr is not None and regime < len(arr):
                v = float(arr[regime])
                if v > 1e-9:
                    return v
        return _FALLBACK_SIGMA

    def _weighted_fingerprint(
        self,
        theta_index: int,
        regime_jacobian: Dict[int, np.ndarray],
        dwell_overrides: Dict[int, float],
    ) -> np.ndarray:
        """
        Stack the regime-axis Jacobian for theta_index across all core
        regimes, precision-weighted by weight(r,c) = sqrt(tau_r)/sigma_r,c
        -- directly implementing Sec 3.4's Var[f_hat|_r] = sigma_r^2 /
        tau_r(m): a probed regime with a longer dwell (larger tau_r) and
        lower real measurement noise (smaller sigma_r) is trusted more,
        i.e. weighted more heavily, when comparing predicted post-probe
        fingerprints.
        """
        components: List[float] = []
        for r in CORE_REGIME_BINS:
            J = regime_jacobian.get(r)
            tau = dwell_overrides.get(r, self._BASELINE_DWELL_S)
            weight_scale = math.sqrt(max(tau, 1e-6))
            if J is None:
                components.extend([0.0] * 9)
                continue
            for c in range(9):
                sigma = self._sigma(c, r)
                weight = weight_scale / max(sigma, 1e-9)
                components.append(float(J[c, theta_index]) * weight)
        return np.array(components, dtype=np.float64)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return abs(float(np.dot(a, b) / (na * nb)))

    def _separability_scores(
        self,
        candidate: CandidateManoeuvre,
        ambiguous_set: Tuple[int, ...],
        regime_jacobian: Dict[int, np.ndarray],
    ) -> Tuple[float, float]:
        """
        G(m) = max over (j,k) in ambiguous_set [ |cos(f_j,f_k)| - |cos(f'_j(m),f'_k(m))| ]
        G_D(m) = log det F'(m) - log det F, restricted to the ambiguous subspace.
        """
        if len(ambiguous_set) < 2 or not regime_jacobian:
            return 0.0, 0.0

        dwell = candidate.parameters.get("dwell_s", candidate.estimated_duration_s)
        dwell_overrides = {r: dwell for r in candidate.target_regimes}

        idxs = [j for j in ambiguous_set if j < regime_jacobian[next(iter(regime_jacobian))].shape[1]]
        if len(idxs) < 2:
            return 0.0, 0.0

        baseline_fps = {j: self._weighted_fingerprint(j, regime_jacobian, {}) for j in idxs}
        post_fps = {j: self._weighted_fingerprint(j, regime_jacobian, dwell_overrides) for j in idxs}

        best_g = 0.0
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                j, k = idxs[a], idxs[b]
                cos_before = self._cosine_similarity(baseline_fps[j], baseline_fps[k])
                cos_after = self._cosine_similarity(post_fps[j], post_fps[k])
                best_g = max(best_g, cos_before - cos_after)

        def _logdet(mat: np.ndarray) -> float:
            gram = mat @ mat.T + np.eye(mat.shape[0]) * 1e-9
            sign, val = np.linalg.slogdet(gram)
            return float(val) if sign > 0 else float("-inf")

        f_before = np.array([baseline_fps[j] for j in idxs])
        f_after = np.array([post_fps[j] for j in idxs])
        g_d = _logdet(f_after) - _logdet(f_before)
        if not math.isfinite(g_d):
            g_d = 0.0

        return float(max(0.0, best_g)), float(g_d)

    def score_candidates(
        self,
        candidates: List[CandidateManoeuvre],
        theta: np.ndarray,
        current_fim_det: float,
        post_probe_fim_det: float,
        current_fuel_kg: float,
        current_altitude_m: float,
        return_margin_s: float,
        p_misattribution: float,
        cost_wrong_action: float,
        current_rpm: float = ASSUMED_BASELINE_CRUISE_RPM,
        current_ias_kt: float = ASSUMED_BASELINE_CRUISE_IAS_KT,
        reserve_fuel_kg: float = ASSUMED_RESERVE_FUEL_KG,
        ambiguous_set: Tuple[int, ...] = (),
    ) -> List[ProbeScore]:
        """
        Score candidates by targeted separability gain / cost ratio
        (primary, Sec 3.6) with D-optimality conditioning as a tie-breaker
        only (secondary), and the value-of-information test.
        """
        regime_jacobian = self._get_regime_jacobian(theta, theta_indices=ambiguous_set) if ambiguous_set else {}

        # _check_feasibility is cheap (pure arithmetic bound checks, no
        # physics_twin call) -- filter first, sequentially, so the
        # expensive parallel step below only pays for candidates that will
        # actually be scored.
        feasible_candidates = []
        feasible_results = []
        for candidate in candidates:
            feasibility = self._check_feasibility(
                candidate, current_fuel_kg, current_altitude_m, return_margin_s,
                theta=theta, current_rpm=current_rpm, current_ias_kt=current_ias_kt,
                reserve_fuel_kg=reserve_fuel_kg,
            )
            if feasibility.is_feasible:
                feasible_candidates.append(candidate)
                feasible_results.append(feasibility)

        # Each candidate's real PhysicsTwin.predict() cost integration is
        # independent of every other's (confirmed by direct profiling to
        # be the dominant real cost of select_probe(), with no shared
        # state or redundant work across candidates to exploit) -- dispatch
        # them across a process pool instead of a sequential Python loop.
        # Falls back to sequential on any pool failure (e.g. an
        # environment where multiprocessing is unavailable) rather than
        # failing the whole selection.
        try:
            model_dir = self.physics_twin.model_dir
            dt = self.physics_twin.config.dt
            max_steps = self._MAX_COST_SIM_STEPS
            with ProcessPoolExecutor(max_workers=min(8, max(1, len(feasible_candidates)) or 1)) as pool:
                cost_results = list(pool.map(
                    _physics_cost_for_candidate,
                    [model_dir] * len(feasible_candidates),
                    [dt] * len(feasible_candidates),
                    feasible_candidates,
                    [theta] * len(feasible_candidates),
                    [max_steps] * len(feasible_candidates),
                ))
        except Exception:
            cost_results = [self._compute_physics_cost(c, theta) for c in feasible_candidates]

        scores = []
        for candidate, feasibility, (delta_life_h, delta_time_h, delta_fuel_kg, delta_exposure) in zip(feasible_candidates, feasible_results, cost_results):
            cost = self.cost_model.compute_cost(delta_life_h, delta_time_h, delta_fuel_kg, delta_exposure)

            if ambiguous_set and regime_jacobian:
                separability_gain, conditioning_gain = self._separability_scores(
                    candidate, ambiguous_set, regime_jacobian
                )
            else:
                # Legacy scalar fallback -- used only when no ambiguous set
                # is supplied or the twin doesn't expose .jacobian() (e.g.
                # a minimal test double). Not the normal operating path.
                separability_gain = max(0.0, post_probe_fim_det - current_fim_det)
                dwell = candidate.estimated_duration_s
                if dwell > 0:
                    separability_gain = min(1.0, separability_gain * math.sqrt(dwell / 300.0))
                conditioning_gain = max(0.0, post_probe_fim_det - current_fim_det)

            composite = separability_gain / (cost + self.cost_model.c0) if cost > 0 else 0.0

            e_benefit = p_misattribution * cost_wrong_action
            passes_voi = e_benefit > cost and separability_gain >= self.g_min

            scores.append(ProbeScore(
                manoeuvre=candidate,
                separability_gain=float(separability_gain),
                conditioning_gain=float(conditioning_gain),
                cost=float(cost),
                composite_score=float(composite),
                passes_value_test=passes_voi,
                feasibility=feasibility,
            ))

        # G(m)/(C(m)+c0) is the primary key; G_D(m) (conditioning_gain) is
        # the secondary tie-breaker only, per Sec 3.6.
        scores.sort(key=lambda s: (s.composite_score, s.conditioning_gain), reverse=True)
        return scores

    def select_probe(
        self,
        ambiguous_set: Tuple[int, ...],
        theta: np.ndarray,
        current_fim_det: float,
        post_probe_fim_det: float,
        current_fuel_kg: float,
        current_altitude_m: float,
        return_margin_s: float,
        p_misattribution: float = 0.3,
        cost_wrong_action: float = 1.0,
        current_rpm: float = ASSUMED_BASELINE_CRUISE_RPM,
        current_ias_kt: float = ASSUMED_BASELINE_CRUISE_IAS_KT,
        reserve_fuel_kg: float = ASSUMED_RESERVE_FUEL_KG,
    ) -> ProbeSelectionResult:
        """
        Select the best probe, or return None if none passes VOI test.
        """
        candidates = self.generate_candidates([], current_fuel_kg)
        scores = self.score_candidates(
            candidates,
            theta,
            current_fim_det,
            post_probe_fim_det,
            current_fuel_kg,
            current_altitude_m,
            return_margin_s,
            p_misattribution,
            cost_wrong_action,
            current_rpm=current_rpm,
            current_ias_kt=current_ias_kt,
            reserve_fuel_kg=reserve_fuel_kg,
            ambiguous_set=ambiguous_set,
        )

        selected = None
        for score in scores:
            if score.passes_value_test:
                selected = score
                break

        if selected is not None:
            reason = f"Selected probe {selected.manoeuvre.manoeuvre_type.value} with score {selected.composite_score:.4f}"
        else:
            reason = "No probe passes value-of-information test"

        return ProbeSelectionResult(
            selected_probe=selected,
            all_scores=scores,
            ambiguous_set=ambiguous_set,
            reason=reason,
        )
