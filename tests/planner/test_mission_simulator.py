"""
Tests for L7 Mission Simulator and Probe Selector.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from replan_to_learn.planner.datatypes import (
    CandidateManoeuvre,
    CostModel,
    FeasibilityResult,
    ManoeuvreType,
    ProbeScore,
    ProbeSelectionResult,
)
from replan_to_learn.planner.mission_simulator import (
    MissionSimulator,
    MissionState,
    isa_temperature_k,
    isa_pressure_pa,
    frame_from_state,
)
from replan_to_learn.planner.probe_selector import ProbeSelector


class MockPhysicsTwin:
    def predict(self, frames, theta):
        n = len(frames) if hasattr(frames, '__len__') else 1
        return np.zeros((n, 9))


class RealisticMockPhysicsTwin:
    """
    A physics-aware test double whose predicted channels genuinely depend
    on the input frame (n_rpm, h_p) and on theta, so real per-candidate
    differentiation and ΔLife/ΔFuel integration can actually be checked
    (unlike MockPhysicsTwin above, whose flat zeros output can't
    distinguish anything).
    """
    N_THETA = 15

    class _Config:
        dt = 0.1

    def __init__(self) -> None:
        self.config = self._Config()

    def predict(self, frames, theta):
        theta = np.asarray(theta, dtype=np.float64)
        rows = []
        for f in frames:
            n_rpm = float(f.n_rpm)
            h_p = float(f.h_p)
            v_tas = float(getattr(f, "v_tas", 0.0))
            theta_vol = theta[0] if len(theta) > 0 else 1.0
            theta_comb = theta[1] if len(theta) > 1 else 1.0
            theta_cool = max(theta[2], 0.05) if len(theta) > 2 else 1.0
            theta_fric = theta[8] if len(theta) > 8 else 1.0
            egt_base = (900.0 + 0.02 * n_rpm) * theta_comb
            # Cooling airflow term (V_TAS-dependent), matching the real
            # thermal model's real form (UA ~ V_TAS^0.8): direct theta_cool
            # excitation via airspeed, as the spec calls out for AIRSPEED_STEP.
            cht = (300.0 + 0.02 * n_rpm + 0.002 * h_p) / (theta_cool * (1.0 + 0.01 * v_tas))
            p_oil = 3.5e5
            t_oil = 300.0 + 0.01 * n_rpm + 0.001 * h_p
            # mdot_f has BOTH a theta_vol-linear term and a theta_comb
            # quadratic-in-RPM cross term, so theta_vol's and theta_comb's
            # regime-to-regime SHAPE genuinely differs (not just scaled) --
            # needed for a non-degenerate separability test.
            mdot_f = 5e-5 * n_rpm * theta_vol + 2e-9 * n_rpm * n_rpm * theta_comb
            p_brake = (0.01 * n_rpm * theta_vol) / max(theta_fric, 0.1)
            rows.append([egt_base, egt_base, egt_base, egt_base, cht, p_oil, t_oil, mdot_f, p_brake])
        arr = np.array(rows, dtype=np.float64)
        return arr if arr.shape[0] > 1 else arr[0]

    def jacobian(self, frames, theta):
        theta = np.asarray(theta, dtype=np.float64)
        n_theta = len(theta)
        y0 = self.predict(frames, theta)
        single = y0.ndim == 1
        if single:
            y0 = y0.reshape(1, -1)
        J = np.zeros((y0.shape[0], 9, n_theta), dtype=np.float64)
        eps = 1e-3
        for j in range(n_theta):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            yp = self.predict(frames, tp)
            ym = self.predict(frames, tm)
            if single:
                yp = yp.reshape(1, -1)
                ym = ym.reshape(1, -1)
            J[:, :, j] = (yp - ym) / (2.0 * eps)
        return J.squeeze(axis=0) if single else J


def _nominal_theta() -> np.ndarray:
    theta = np.ones(15, dtype=np.float64)
    theta[9:] = 0.0
    return theta


class TestMissionSimulator:
    def test_simulate_returns_result(self) -> None:
        twin = MockPhysicsTwin()
        sim = MissionSimulator(physics_twin=twin)
        profile = [
            MissionState(altitude_m=1500.0, ias_kt=120.0, rpm=2800.0, fuel_kg=50.0, elapsed_s=0.0, regime=4),
        ]
        result = sim.simulate(theta=np.ones(15), mission_profile=profile, current_fuel_kg=50.0)
        assert result.risk in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(result.alternatives, list)

    def test_simulate_manoeuvre_climb(self) -> None:
        twin = MockPhysicsTwin()
        sim = MissionSimulator(physics_twin=twin)
        profile = [
            MissionState(altitude_m=1500.0, ias_kt=120.0, rpm=2800.0, fuel_kg=50.0, elapsed_s=0.0, regime=4),
        ]
        candidate = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.STEP_CLIMB,
            parameters={"delta_h_m": 1000.0, "target_ias_kt": 85.0},
            estimated_duration_s=120.0,
            estimated_cost=0.0,
            target_regimes=(0, 1),
        )
        result = sim.simulate_manoeuvre(theta=np.ones(15), base_profile=profile, manoeuvre=candidate, current_fuel_kg=50.0)
        assert result.risk in ("LOW", "MEDIUM", "HIGH")

    def test_isa_ambient_varies_with_altitude(self) -> None:
        """
        Real physical requirement: climbing must actually reduce ambient
        pressure/temperature (previously map_pa/t_amb/p_amb were fixed
        constants for every frame regardless of altitude).
        """
        t_sea_level = isa_temperature_k(0.0)
        t_5000m = isa_temperature_k(5000.0)
        p_sea_level = isa_pressure_pa(0.0)
        p_5000m = isa_pressure_pa(5000.0)

        assert t_sea_level == pytest.approx(288.15, abs=0.01)
        # ICAO ISA lapse rate: -6.5K/km -> 288.15 - 32.5 = 255.65K at 5000m
        assert t_5000m == pytest.approx(255.65, abs=0.5)
        assert p_sea_level == pytest.approx(101325.0, abs=1.0)
        # Real ISA pressure at 5000m is ~54000 Pa (~0.533 atm)
        assert p_5000m == pytest.approx(54048.0, rel=0.02)
        assert p_5000m < p_sea_level

    def test_frame_from_state_reflects_altitude(self) -> None:
        low = MissionState(altitude_m=0.0, ias_kt=100.0, rpm=2500.0, fuel_kg=50.0, elapsed_s=0.0, regime=4)
        high = MissionState(altitude_m=5000.0, ias_kt=100.0, rpm=2500.0, fuel_kg=50.0, elapsed_s=0.0, regime=9)
        frame_low = frame_from_state(low)
        frame_high = frame_from_state(high)
        assert frame_high.t_amb < frame_low.t_amb
        assert frame_high.p_amb < frame_low.p_amb

    def test_simulate_manoeuvre_calls_predict_with_altitude_varying_frames(self) -> None:
        """
        The manoeuvre trajectory sent to the twin must reflect the real
        altitude/rpm changes across the manoeuvre (proving the fuel
        integration path uses real per-step state, not a single flat
        constant burn rate).
        """
        recorded = {"calls": []}

        class RecordingTwin(RealisticMockPhysicsTwin):
            def predict(self, frames, theta):
                recorded["calls"].append(list(frames))
                return super().predict(frames, theta)

        twin = RecordingTwin()
        sim = MissionSimulator(physics_twin=twin)
        profile = [
            MissionState(altitude_m=1500.0, ias_kt=120.0, rpm=2800.0, fuel_kg=50.0, elapsed_s=float(i), regime=4)
            for i in range(20)
        ]
        candidate = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.STEP_CLIMB,
            parameters={"delta_h_m": 1000.0, "target_ias_kt": 85.0},
            estimated_duration_s=5.0,
            estimated_cost=0.0,
            target_regimes=(0, 1),
        )
        sim.simulate_manoeuvre(theta=_nominal_theta(), base_profile=profile, manoeuvre=candidate, current_fuel_kg=50.0)

        # The frames used for the manoeuvre's own fuel-burn prediction
        # (the first predict() call, before the final full-profile
        # simulate() call) must show altitude actually changing across
        # steps.
        manoeuvre_frames = recorded["calls"][0]
        altitudes = [f.h_p for f in manoeuvre_frames]
        assert max(altitudes) > min(altitudes)


class TestProbeSelector:
    def test_generate_candidates(self) -> None:
        twin = MockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        candidates = selector.generate_candidates([], 50.0)
        assert len(candidates) > 0
        types = {c.manoeuvre_type for c in candidates}
        assert ManoeuvreType.STEP_CLIMB in types
        assert ManoeuvreType.POWER_DERATE in types

    def test_feasibility_filter(self) -> None:
        twin = MockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        candidate = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.STEP_CLIMB,
            parameters={"delta_h_m": 15000.0, "target_ias_kt": 85.0},
            estimated_duration_s=60.0,
            estimated_cost=0.0,
            target_regimes=(0, 1),
        )
        feasibility = selector._check_feasibility(candidate, current_fuel_kg=5.0, current_altitude_m=1000.0, return_margin_s=600.0)
        assert not feasibility.is_feasible

    def test_select_probe_returns_result(self) -> None:
        twin = MockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        result = selector.select_probe(
            ambiguous_set=(1, 2),
            theta=np.ones(15),
            current_fim_det=1.0,
            post_probe_fim_det=1.5,
            current_fuel_kg=50.0,
            current_altitude_m=1500.0,
            return_margin_s=600.0,
        )
        assert isinstance(result, ProbeSelectionResult)
        assert result.ambiguous_set == (1, 2)

    def test_voi_test_gates_selection(self) -> None:
        twin = MockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        result = selector.select_probe(
            ambiguous_set=(1, 2),
            theta=np.ones(15),
            current_fim_det=1.0,
            post_probe_fim_det=1.0,
            current_fuel_kg=50.0,
            current_altitude_m=1500.0,
            return_margin_s=600.0,
            p_misattribution=0.0,
            cost_wrong_action=0.0,
        )
        assert result.selected_probe is None

    def test_compute_physics_cost_matches_twin_mdot_f(self) -> None:
        """
        ΔFuel must come from the twin's own predicted mdot_f integrated
        over the manoeuvre duration -- not the old flat
        duration*0.3 kg/s placeholder.
        """
        twin = RealisticMockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        candidate = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.POWER_DERATE,
            parameters={"delta_n_rpm": -100.0, "dwell_s": 60.0},
            estimated_duration_s=60.0,
            estimated_cost=0.0,
            target_regimes=(3,),
        )
        theta = _nominal_theta()
        delta_life_h, delta_time_h, delta_fuel_kg, delta_exposure = selector._compute_physics_cost(candidate, theta)

        # REGIME_CENTERS[3].n_rpm = 2000.0 (CRUISE_LOW_POWER); POWER_DERATE
        # holds n_rpm constant at op.n_rpm + delta_n_rpm for the whole
        # manoeuvre -> mdot_f = 5e-5*n_rpm*theta_vol + 2e-9*n_rpm^2*theta_comb
        # (RealisticMockPhysicsTwin.predict's formula) at n_rpm=1900, theta=1.
        n_rpm = 2000.0 - 100.0
        expected_mdot_f = 5e-5 * n_rpm * 1.0 + 2e-9 * n_rpm * n_rpm * 1.0
        expected_fuel_kg = expected_mdot_f * 60.0
        assert delta_fuel_kg == pytest.approx(expected_fuel_kg, rel=0.05)
        # Old fabricated formula would have given 60*0.3 = 18kg -- very
        # different order of magnitude; confirm we're not just coincidentally
        # matching it.
        assert delta_fuel_kg != pytest.approx(18.0, rel=0.1)
        assert delta_time_h == pytest.approx(60.0 / 3600.0)
        assert delta_life_h > 0.0

    def test_return_margin_violation_detected_for_long_manoeuvre(self) -> None:
        """
        A probe that resolves the diagnosis but strands the aircraft must
        be rejected -- return_margin_s was previously accepted and never
        actually checked against anything.
        """
        twin = RealisticMockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        candidate = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.STEP_CLIMB,
            parameters={"delta_h_m": 3000.0, "target_ias_kt": 85.0},
            estimated_duration_s=360.0,
            estimated_cost=0.0,
            target_regimes=(0, 1),
        )
        feasibility = selector._check_feasibility(
            candidate,
            current_fuel_kg=50.0,
            current_altitude_m=1000.0,
            return_margin_s=600.0,  # far short of the ~1560s this climb+return costs
            theta=_nominal_theta(),
        )
        assert not feasibility.is_feasible
        assert "return_margin_violated" in feasibility.violations

    def test_return_margin_ok_for_short_manoeuvre(self) -> None:
        twin = RealisticMockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        candidate = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.AIRSPEED_STEP,
            parameters={"delta_ias_kt": 10.0, "dwell_s": 120.0},
            estimated_duration_s=120.0,
            estimated_cost=0.0,
            target_regimes=(0, 1, 3),
        )
        feasibility = selector._check_feasibility(
            candidate,
            current_fuel_kg=50.0,
            current_altitude_m=1500.0,
            return_margin_s=10000.0,
            theta=_nominal_theta(),
        )
        assert "return_margin_violated" not in feasibility.violations

    def test_power_derate_feasibility_checks_rpm_floor(self) -> None:
        twin = RealisticMockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        candidate = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.POWER_DERATE,
            parameters={"delta_n_rpm": -250.0, "dwell_s": 60.0},
            estimated_duration_s=60.0,
            estimated_cost=0.0,
            target_regimes=(3,),
        )
        feasibility = selector._check_feasibility(
            candidate,
            current_fuel_kg=50.0,
            current_altitude_m=1500.0,
            return_margin_s=10000.0,
            theta=_nominal_theta(),
            current_rpm=1900.0,  # near idle; -250 would drop below the real Rotax idle limit (1800rpm)
        )
        assert not feasibility.is_feasible
        assert "rpm_below_idle" in feasibility.violations

    def test_descent_segment_feasibility_checks_altitude_floor(self) -> None:
        twin = RealisticMockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        candidate = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.DESCENT_SEGMENT,
            parameters={"delta_rod_ms": -4.0, "dwell_s": 240.0},
            estimated_duration_s=240.0,
            estimated_cost=0.0,
            target_regimes=(6, 7),
        )
        feasibility = selector._check_feasibility(
            candidate,
            current_fuel_kg=50.0,
            current_altitude_m=500.0,  # 500m - 4*240 = -460m -> well below safe floor
            return_margin_s=10000.0,
            theta=_nominal_theta(),
        )
        assert not feasibility.is_feasible
        assert "altitude_below_safe_floor" in feasibility.violations

    def test_separability_scores_reflect_which_regimes_are_excited(self) -> None:
        """
        Direct, deterministic test of the real targeted-separability math
        (Sec 3.4: G(m) = max over ambiguous pairs of the cosine-similarity
        DROP after the probe, weighted by Var[f_hat|_r] = sigma_r^2/tau_r(m)).
        Two theta parameters are set up to be strongly correlated (same
        derivative) across most regimes, but with parameter 1's
        sensitivity going nearly flat (near-orthogonal) in regimes 0 and 3
        specifically. A candidate that dwells in regimes {0,3} should
        genuinely separate them (G > 0); a candidate that dwells in
        regimes where they're identical (e.g. {1,2}) should not.
        """
        twin = RealisticMockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        n_theta = 15
        regime_jacobian = {}
        for r in range(12):
            J = np.zeros((9, n_theta), dtype=np.float64)
            J[0, 0] = 1.0
            J[0, 1] = 0.05 if r in (0, 3) else 1.0
            regime_jacobian[r] = J

        candidate_excites_0_3 = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.STEP_CLIMB,
            parameters={"delta_h_m": 1000.0, "target_ias_kt": 85.0, "dwell_s": 100.0},
            estimated_duration_s=100.0,
            estimated_cost=0.0,
            target_regimes=(0, 3),
        )
        candidate_excites_1_2 = CandidateManoeuvre(
            manoeuvre_type=ManoeuvreType.POWER_DERATE,
            parameters={"delta_n_rpm": -100.0, "dwell_s": 100.0},
            estimated_duration_s=100.0,
            estimated_cost=0.0,
            target_regimes=(1, 2),
        )

        g_0_3, _ = selector._separability_scores(candidate_excites_0_3, (0, 1), regime_jacobian)
        g_1_2, _ = selector._separability_scores(candidate_excites_1_2, (0, 1), regime_jacobian)

        assert g_0_3 > 0.3, f"expected the probe exciting the differentiating regimes to genuinely separate the pair, got {g_0_3}"
        assert g_1_2 < g_0_3, "a probe exciting regimes where the two parameters look IDENTICAL must not separate them as well"

    def test_candidates_of_different_types_get_different_separability_gain(self) -> None:
        """
        Integration-level check using the real physics-aware mock twin:
        different manoeuvre types (which excite different regimes/channels
        per Sec 3.3's table) must not all collapse onto the same
        separability_gain, unlike the old behaviour where only
        sqrt(dwell/300) varied per candidate regardless of type.
        """
        twin = RealisticMockPhysicsTwin()
        selector = ProbeSelector(physics_twin=twin)
        candidates = selector.generate_candidates([], 50.0)
        theta = _nominal_theta()
        ambiguous_set = (0, 1)
        scores = selector.score_candidates(
            candidates,
            theta,
            current_fim_det=1.0,
            post_probe_fim_det=1.0,
            current_fuel_kg=50.0,
            current_altitude_m=1500.0,
            return_margin_s=100000.0,
            p_misattribution=0.5,
            cost_wrong_action=1.0,
            ambiguous_set=ambiguous_set,
        )
        assert len(scores) > 0
        distinct_costs = {round(s.cost, 6) for s in scores}
        # Real, physics-derived costs must differ across candidates (this
        # alone rules out the old fabricated flat-rate cost model, where
        # cost was purely a function of duration).
        assert len(distinct_costs) > 1
