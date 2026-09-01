"""
Adversarial Stress Test Suite for Milestone 1 - Challenger 2.
Empirical validation of:
1. AmbiguousPairModel (cruise collinearity, orthogonal power residual decoupling, multi-regime FIM conditioning, CRLB bounds).
2. REGIME_GRID_V1 & QuasiSteadyDetector (20s continuous stability gating, derivative spikes, boundary transitions, noise stress, all 12 core + 3 operational bins, encounter counters, and borrow weights).
"""

from __future__ import annotations

import json
import math
import numpy as np
import pytest

from replan_to_learn.contracts.health import (
    HEALTH_PARAMETER_SPECS,
    NUM_HEALTH_PARAMETERS,
    PARAMETER_NAMES,
    AmbiguousPairModel,
    HealthParameters,
)
from replan_to_learn.contracts.regimes import (
    ALTITUDE_THRESHOLD_M,
    BIN_ALTITUDE_HIGH_POWER,
    BIN_ALTITUDE_LOW_POWER,
    BIN_ALTITUDE_MID_POWER,
    BIN_CLIMB_HIGH_POWER,
    BIN_CLIMB_LOW_POWER,
    BIN_CLIMB_MID_POWER,
    BIN_CRUISE_HIGH_POWER,
    BIN_CRUISE_LOW_POWER,
    BIN_CRUISE_MID_POWER,
    BIN_DESCENT_HIGH_POWER,
    BIN_DESCENT_LOW_POWER,
    BIN_DESCENT_MID_POWER,
    BIN_IDLE,
    BIN_INVALID,
    BIN_TRANSIENT,
    CORE_REGIME_BINS,
    IDLE_RPM_THRESHOLD,
    MAX_DERIV_CHT_K_PER_S,
    MAX_DERIV_MAP_PA_PER_S,
    MAX_DERIV_RPM_PER_S,
    MCP_POWER_KW,
    MIN_STABLE_DURATION_S,
    NUM_CORE_BINS,
    NUM_TOTAL_BINS,
    REGIME_GRID_DEFINITIONS,
    REGIME_GRID_VERSION,
    REGIME_NAMES,
    ROC_CRUISE_THRESHOLD_MS,
    AircraftExposureProfile,
    EncounterTracker,
    RegimeClassifier,
)


# ============================================================================
# 1. Stress Tests: AmbiguousPairModel & Sensitivity Decoupling
# ============================================================================

class TestAdversarialAmbiguousPairModel:
    """Adversarial stress testing of the cooling/combustion ambiguous pair."""

    def test_cruise_only_thermal_collinearity_exceeds_90_percent(self) -> None:
        """
        Verify that thermal-only channels in cruise exhibit severe collinearity (cos(phi) > 0.90).
        This proves mathematically why single-regime cruise cannot distinguish cooling from combustion.
        """
        j_cool_8, j_comb_8 = AmbiguousPairModel.compute_jacobian_columns("CRUISE", include_power_residual=False)
        assert len(j_cool_8) == 8
        assert len(j_comb_8) == 8

        cos_sim = AmbiguousPairModel.compute_collinearity("CRUISE", include_power_residual=False)
        assert cos_sim > 0.90, f"Expected collinearity > 0.90, got {cos_sim:.4f}"
        assert cos_sim == pytest.approx(0.91759, rel=1e-3)
        
        # Verify specific thermal components:
        # Both J_cool and J_comb have dominant positive CHT and Oil Temp sensitivities
        norm_cool = np.linalg.norm(j_cool_8)
        norm_comb = np.linalg.norm(j_comb_8)
        dot_product = np.dot(j_cool_8, j_comb_8)
        assert dot_product / (norm_cool * norm_comb) == pytest.approx(cos_sim)

    def test_power_balance_residual_orthogonality_and_decoupling(self) -> None:
        """
        Verify that the 9th channel (power balance residual r_N) is strictly orthogonal to theta_cool
        (d(r_N)/d(theta_cool) == 0.0) while strongly sensitive to theta_comb (d(r_N)/d(theta_comb) == -1.50).
        """
        j_cool_9, j_comb_9 = AmbiguousPairModel.compute_jacobian_columns("CRUISE", include_power_residual=True)
        assert len(j_cool_9) == 9
        assert len(j_comb_9) == 9

        # Power balance sensitivity of cooling is zero
        assert j_cool_9[8] == 0.0
        # Power balance sensitivity of combustion is negative and significant
        assert j_comb_9[8] == -1.50

        # Adding r_N dramatically drops collinearity from ~0.918 down to ~0.526
        cos_sim_with_rn = AmbiguousPairModel.compute_collinearity("CRUISE", include_power_residual=True)
        assert cos_sim_with_rn < 0.60
        assert cos_sim_with_rn == pytest.approx(0.5262, rel=1e-3)
        assert cos_sim_with_rn < AmbiguousPairModel.compute_collinearity("CRUISE", include_power_residual=False)

    def test_climb_and_descent_collinearity_resolution_below_30_percent(self) -> None:
        """
        Verify that in dynamic regimes (Climb and Descent) with orthogonal power-balance residual,
        collinearity drops strictly below 0.30:
        - Climb with r_N: cos(phi) = 0.2530 (< 0.30)
        - Descent with r_N: cos(phi) = 0.2623 (< 0.30)
        """
        cos_climb = AmbiguousPairModel.compute_collinearity("CLIMB", include_power_residual=True)
        cos_descent = AmbiguousPairModel.compute_collinearity("DESCENT", include_power_residual=True)

        assert cos_climb < 0.30, f"Expected Climb cos(phi) < 0.30, got {cos_climb:.4f}"
        assert cos_climb == pytest.approx(0.2530, rel=1e-3)

        assert cos_descent < 0.30, f"Expected Descent cos(phi) < 0.30, got {cos_descent:.4f}"
        assert cos_descent == pytest.approx(0.2623, rel=1e-3)

    def test_multi_regime_fisher_information_matrix_conditioning_spectrum(self) -> None:
        """
        Compare FIM condition numbers across four configurations:
        1. Single regime Cruise (thermal only) -> Ill-conditioned (cond ~ 23.35)
        2. Single regime Cruise (with power residual) -> Improved conditioning (cond ~ 5.71)
        3. Multi-regime without power residual -> Improved by thermal divergence across ROC (cond ~ 3.99)
        4. Multi-regime with power residual -> Optimal conditioning (cond ~ 3.41 < 5.0)
        """
        fim_cruise_thermal, cond_c_th = AmbiguousPairModel.compute_fisher_information_matrix(["CRUISE"], include_power_residual=False)
        fim_cruise_power, cond_c_pw = AmbiguousPairModel.compute_fisher_information_matrix(["CRUISE"], include_power_residual=True)
        fim_multi_thermal, cond_m_th = AmbiguousPairModel.compute_fisher_information_matrix(["CRUISE", "CLIMB", "DESCENT"], include_power_residual=False)
        fim_multi_power, cond_m_pw = AmbiguousPairModel.compute_fisher_information_matrix(["CRUISE", "CLIMB", "DESCENT"], include_power_residual=True)

        # Verify strict monotonic conditioning improvement
        assert cond_c_pw < cond_c_th
        assert cond_m_pw < cond_c_pw
        assert cond_m_pw < 5.0, f"Expected optimal FIM condition < 5.0, got {cond_m_pw:.2f}"
        assert cond_m_pw == pytest.approx(3.4104, rel=1e-3)

        # Check positive-definiteness of all FIMs
        for fim in [fim_cruise_thermal, fim_cruise_power, fim_multi_thermal, fim_multi_power]:
            eigenvalues = np.linalg.eigvalsh(fim)
            assert np.all(eigenvalues > 0.0), f"FIM not positive-definite: eigenvalues={eigenvalues}"

    def test_cramer_rao_lower_bound_covariance_bounds(self) -> None:
        """
        Verify that the Cramer-Rao Lower Bound (CRLB = inv(FIM)) yields finite, admissible
        parameter variance bounds for (theta_cool, theta_comb) under multi-regime data.
        """
        demo = AmbiguousPairModel.demonstrate_distinct_parameter_necessity()
        crlb_cov = np.array(demo["crlb_covariance_matrix"])
        assert crlb_cov.shape == (2, 2)
        assert np.all(np.isfinite(crlb_cov))

        # Variance of theta_cool and theta_comb must be positive and small (< 0.5)
        var_cool = crlb_cov[0, 0]
        var_comb = crlb_cov[1, 1]
        assert var_cool > 0.0
        assert var_comb > 0.0
        assert var_cool < 0.5
        assert var_comb < 0.5

        # Off-diagonal covariance magnitude must satisfy Cauchy-Schwarz |Cov| < sqrt(Var1 * Var2)
        cov_cool_comb = crlb_cov[0, 1]
        assert abs(cov_cool_comb) < math.sqrt(var_cool * var_comb)

    def test_invalid_regime_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown regime"):
            AmbiguousPairModel.compute_jacobian_columns("HYPERSONIC")


# ============================================================================
# 2. Stress Tests: REGIME_GRID_V1 Completeness & Bin Exhaustiveness
# ============================================================================

class TestAdversarialRegimeGridGridExhaustiveness:
    """Stress test the 12 core bins + 3 operational bins mapping and boundary definitions."""

    def test_all_15_bins_distinct_and_defined(self) -> None:
        """Verify that all 15 regime bins (0..14) have complete and unique specifications."""
        assert len(REGIME_GRID_DEFINITIONS) == 15
        bin_ids = list(REGIME_GRID_DEFINITIONS.keys())
        assert sorted(bin_ids) == list(range(15))

        names = set()
        for b_id, defn in REGIME_GRID_DEFINITIONS.items():
            assert defn.bin_id == b_id
            assert defn.name in REGIME_NAMES.values()
            assert defn.name not in names
            names.add(defn.name)

    @pytest.mark.parametrize("bin_id", range(12))
    def test_all_12_core_bins_reachable(self, bin_id: int) -> None:
        """Adversarially construct operating points that specifically target each of the 12 core bins."""
        classifier = RegimeClassifier(min_stable_duration_s=0.0)

        # Mapping test table: (rpm, p_brake_kw, roc_ms, alt_m)
        target_points = {
            # Climb Bins (0, 1, 2): alt <= 3000m, roc > +1.5 m/s
            0: (5200.0, 30.0, +3.0, 1500.0),   # CLIMB_LOW_POWER (< 45 kW)
            1: (5400.0, 60.0, +4.0, 1500.0),   # CLIMB_MID_POWER (45-75 kW)
            2: (5800.0, 90.0, +6.0, 1500.0),   # CLIMB_HIGH_POWER (> 75 kW)
            # Cruise Bins (3, 4, 5): alt <= 3000m, |roc| <= 1.5 m/s
            3: (4800.0, 35.0, 0.0, 2000.0),    # CRUISE_LOW_POWER (< 45 kW)
            4: (5000.0, 60.0, 0.0, 2000.0),    # CRUISE_MID_POWER (45-75 kW)
            5: (5500.0, 85.0, 0.0, 2000.0),    # CRUISE_HIGH_POWER (> 75 kW)
            # Descent Bins (6, 7, 8): alt <= 3000m, roc < -1.5 m/s
            6: (3500.0, 25.0, -3.0, 1800.0),   # DESCENT_LOW_POWER (< 45 kW)
            7: (4000.0, 55.0, -3.5, 1800.0),   # DESCENT_MID_POWER (45-75 kW)
            8: (4800.0, 80.0, -4.0, 1800.0),   # DESCENT_HIGH_POWER (> 75 kW)
            # High Altitude Bins (9, 10, 11): alt > 3000m
            9: (5000.0, 30.0, 0.0, 4500.0),    # ALTITUDE_LOW_POWER (< 45 kW)
            10: (5300.0, 65.0, +1.0, 4500.0),  # ALTITUDE_MID_POWER (45-75 kW)
            11: (5800.0, 95.0, -2.0, 4500.0),  # ALTITUDE_HIGH_POWER (> 75 kW)
        }

        rpm, p_kw, roc, alt = target_points[bin_id]
        detected_bin = classifier.determine_candidate_bin(rpm, p_kw, roc, alt)
        assert detected_bin == bin_id, (
            f"Failed for bin {bin_id} ({REGIME_NAMES[bin_id]}): "
            f"got {detected_bin} ({REGIME_NAMES[detected_bin]})"
        )

    @pytest.mark.parametrize("bin_id", [12, 13, 14])
    def test_all_3_operational_bins_reachable(self, bin_id: int) -> None:
        """Adversarially construct operating points targeting IDLE, TRANSIENT, and INVALID."""
        classifier = RegimeClassifier(min_stable_duration_s=20.0)

        if bin_id == BIN_IDLE:
            # Idle rpm < 1600.0
            res_bin = classifier.determine_candidate_bin(1400.0, 10.0, 0.0, 500.0)
            assert res_bin == BIN_IDLE
        elif bin_id == BIN_TRANSIENT:
            # Step with unstable derivative produces TRANSIENT
            classifier.step(0.0, 5000.0, 100000.0, 360.0, 2000.0, 60.0)
            # Step with large dRPM = 100 rpm/s > 30 rpm/s
            res_bin, _ = classifier.step(1.0, 5100.0, 100000.0, 360.0, 2000.0, 60.0)
            assert res_bin == BIN_TRANSIENT
        elif bin_id == BIN_INVALID:
            # Step with NaN sensor produces INVALID
            res_bin, _ = classifier.step(0.0, float("nan"), 100000.0, 360.0, 2000.0, 60.0)
            assert res_bin == BIN_INVALID


# ============================================================================
# 3. Stress Tests: QuasiSteadyDetector & 20-Second Continuous Stability Gate
# ============================================================================

class TestAdversarialQuasiSteadyDetector:
    """Stress testing derivative thresholds, noisy signals, rapid transients, and boundary conditions."""

    def test_exact_derivative_threshold_boundary_behavior(self) -> None:
        """
        Test strict inequality boundaries:
        - |dN/dt| < 30.0 rpm/s
        - |dp_im/dt| < 2000.0 Pa/s
        - |dT_hd/dt| < 0.15 K/s
        """
        classifier = RegimeClassifier(min_stable_duration_s=0.0)

        # 1. RPM boundary: 29.99 rpm/s is STABLE, 30.01 rpm/s is UNSTABLE
        is_st, _ = classifier.check_stability_derivatives(dt=1.0, d_rpm=29.99, d_map_pa=0.0, d_cht_k=0.0)
        assert is_st is True
        is_st, _ = classifier.check_stability_derivatives(dt=1.0, d_rpm=30.01, d_map_pa=0.0, d_cht_k=0.0)
        assert is_st is False
        # Exact threshold boundary (30.00)
        is_st, _ = classifier.check_stability_derivatives(dt=1.0, d_rpm=30.00, d_map_pa=0.0, d_cht_k=0.0)
        assert is_st is False

        # 2. MAP boundary: 1999.0 Pa/s is STABLE, 2001.0 Pa/s is UNSTABLE
        is_st, _ = classifier.check_stability_derivatives(dt=1.0, d_rpm=0.0, d_map_pa=1999.0, d_cht_k=0.0)
        assert is_st is True
        is_st, _ = classifier.check_stability_derivatives(dt=1.0, d_rpm=0.0, d_map_pa=2001.0, d_cht_k=0.0)
        assert is_st is False
        is_st, _ = classifier.check_stability_derivatives(dt=1.0, d_rpm=0.0, d_map_pa=2000.0, d_cht_k=0.0)
        assert is_st is False

        # 3. CHT boundary: 0.149 K/s is STABLE, 0.151 K/s is UNSTABLE
        is_st, _ = classifier.check_stability_derivatives(dt=1.0, d_rpm=0.0, d_map_pa=0.0, d_cht_k=0.149)
        assert is_st is True
        is_st, _ = classifier.check_stability_derivatives(dt=1.0, d_rpm=0.0, d_map_pa=0.0, d_cht_k=0.151)
        assert is_st is False
        is_st, _ = classifier.check_stability_derivatives(dt=1.0, d_rpm=0.0, d_map_pa=0.0, d_cht_k=0.150)
        assert is_st is False

    def test_single_transient_spike_at_second_19_resets_full_20s_gate(self) -> None:
        """
        A single out-of-spec derivative spike at t=19.0s MUST completely reset the 20s gate,
        requiring another full 20 continuous stable seconds before regime assignment.
        """
        classifier = RegimeClassifier(min_stable_duration_s=20.0)

        # 1. 19 stable steps (t=0..19)
        for s in range(20):
            bin_id, dur = classifier.step(
                t=float(s), rpm=5000.0, map_pa=100000.0, cht_k=365.0, alt_m=2000.0, p_brake_kw=60.0
            )
            if s < 20:
                assert bin_id == BIN_TRANSIENT

        # Duration accumulated is 19.0s at s=19
        assert classifier.current_stable_duration == pytest.approx(19.0)

        # 2. Step 20 (t=20): A brief throttle perturbation (MAP jumps 5 kPa in 1s)
        bin_id, dur = classifier.step(
            t=20.0, rpm=5000.0, map_pa=105000.0, cht_k=365.0, alt_m=2000.0, p_brake_kw=60.0
        )
        assert bin_id == BIN_TRANSIENT
        assert dur == 0.0
        assert classifier.current_stable_duration == 0.0

        # 3. Step 21..39 (19 more stable seconds) -> still TRANSIENT
        for s in range(21, 40):
            bin_id, dur = classifier.step(
                t=float(s), rpm=5000.0, map_pa=105000.0, cht_k=365.0, alt_m=2000.0, p_brake_kw=60.0
            )
            assert bin_id == BIN_TRANSIENT
            assert dur < 20.0

        # 4. Step 40 (t=40.0s, exactly 20.0s since reset at t=20.0) -> Valid CRUISE_MID_POWER
        bin_id, dur = classifier.step(
            t=40.0, rpm=5000.0, map_pa=105000.0, cht_k=365.0, alt_m=2000.0, p_brake_kw=60.0
        )
        assert bin_id == BIN_CRUISE_MID_POWER
        assert dur >= 20.0

    def test_regime_bin_switching_resets_stable_duration(self) -> None:
        """
        If operating point drifts from Cruise Low Power (Bin 3) to Cruise Mid Power (Bin 4)
        even while derivatives remain quasi-steady, the 20s counter must reset for the new bin.
        """
        classifier = RegimeClassifier(min_stable_duration_s=20.0)

        # 1. 15 seconds in Cruise Low Power (p_brake = 40 kW < 45 kW)
        for s in range(15):
            bin_id, dur = classifier.step(
                t=float(s), rpm=4500.0, map_pa=80000.0, cht_k=350.0, alt_m=1000.0, p_brake_kw=40.0
            )
            assert bin_id == BIN_TRANSIENT

        assert classifier.current_stable_duration == pytest.approx(14.0)

        # 2. Smooth throttle advancement crossing power boundary to 50 kW (Mid Power)
        # Power increases by 10 kW, but RPM increases by only 10 rpm (0.1 rpm/s), MAP by 500 Pa (500 Pa/s)
        # Derivatives are STABLE, but candidate bin changes from 3 to 4!
        bin_id, dur = classifier.step(
            t=15.0, rpm=4510.0, map_pa=80500.0, cht_k=350.0, alt_m=1000.0, p_brake_kw=50.0
        )
        assert bin_id == BIN_TRANSIENT
        # Stable duration MUST reset to dt (1.0s) for the new candidate bin
        assert dur == pytest.approx(1.0)
        assert classifier.current_stable_duration == pytest.approx(1.0)

    def test_high_frequency_sensor_noise_stress_rejection(self) -> None:
        """
        Adversarial test: Inject realistic Gaussian noise on sensor channels.
        Excessive noise exceeding derivative thresholds should keep the state in TRANSIENT,
        while low-pass / acceptable noise within thresholds allows convergence to core bin.
        """
        rng = np.random.default_rng(42)
        classifier_noisy = RegimeClassifier(min_stable_duration_s=20.0)

        # 1. High noise: CHT noise sigma = 0.5 K -> derivative ~ 0.5 K/s >> 0.15 K/s threshold
        unstable_count = 0
        for s in range(50):
            t = float(s)
            rpm = 5000.0 + rng.normal(0.0, 50.0)    # RPM jitter ~ 50 rpm/s > 30
            map_pa = 100000.0 + rng.normal(0.0, 3000.0) # MAP jitter ~ 3000 Pa/s > 2000
            cht_k = 365.0 + rng.normal(0.0, 0.5)    # CHT jitter ~ 0.5 K/s > 0.15
            b_id, dur = classifier_noisy.step(t, rpm, map_pa, cht_k, 2000.0, 60.0)
            if b_id == BIN_TRANSIENT:
                unstable_count += 1

        # Should be almost 100% transient due to noise exceeding limits
        assert unstable_count >= 48

        # 2. Clean signal with gentle smooth noise within limits
        # RPM sigma = 2 rpm/s (< 30), MAP sigma = 100 Pa/s (< 2000), CHT sigma = 0.01 K/s (< 0.15)
        classifier_gentle = RegimeClassifier(min_stable_duration_s=20.0)
        core_assigned = False
        for s in range(30):
            t = float(s)
            rpm = 5000.0 + rng.normal(0.0, 2.0)
            map_pa = 100000.0 + rng.normal(0.0, 100.0)
            cht_k = 365.0 + rng.normal(0.0, 0.01)
            b_id, dur = classifier_gentle.step(t, rpm, map_pa, cht_k, 2000.0, 60.0)
            if b_id == BIN_CRUISE_MID_POWER:
                core_assigned = True

        assert core_assigned is True

    def test_variable_dt_and_time_gaps(self) -> None:
        """
        Verify that QuasiSteadyDetector accurately accumulates time with variable dt (0.1s, 0.5s, 2.0s)
        and resets when dt > 5.0s (telemetry packet drop / time gap).
        """
        classifier = RegimeClassifier(min_stable_duration_s=20.0)

        # 41 steps at dt = 0.5s (40 intervals) = 20.0 seconds of elapsed duration
        t = 0.0
        for _ in range(41):
            t += 0.5
            b_id, dur = classifier.step(t, 5000.0, 100000.0, 365.0, 2000.0, 60.0)

        # At t=20.5s (step 41), exactly 20.0s of continuous stable duration has elapsed
        assert b_id == BIN_CRUISE_MID_POWER
        assert dur == pytest.approx(20.0)

        # Now inject a time jump of 10.0s (dt = 10.0 > 5.0)
        t += 10.0
        b_id, dur = classifier.step(t, 5000.0, 100000.0, 365.0, 2000.0, 60.0)
        assert b_id == BIN_TRANSIENT
        assert dur == 0.0


# ============================================================================
# 4. Stress Tests: EncounterTracker & Fleet-Borrow Weight Dynamics
# ============================================================================

class TestAdversarialEncounterTracker:
    """Stress testing multi-aircraft encounter tracking, persistence, and borrow weights."""

    def test_multi_aircraft_isolated_exposure_tracking(self) -> None:
        tracker = EncounterTracker()

        # Aircraft 1 flies 150s in Climb High Power (Bin 2)
        for s in range(150):
            tracker.record_step("UAV_001", BIN_CLIMB_HIGH_POWER, dt=1.0, stable_duration_s=25.0, t=float(s))

        # Aircraft 2 flies 60s in Cruise Mid Power (Bin 4) and 30s in Descent Low Power (Bin 6)
        for s in range(60):
            tracker.record_step("UAV_002", BIN_CRUISE_MID_POWER, dt=1.0, stable_duration_s=25.0, t=float(s))
        for s in range(30):
            tracker.record_step("UAV_002", BIN_DESCENT_LOW_POWER, dt=1.0, stable_duration_s=25.0, t=float(60 + s))

        p1 = tracker.get_or_create_profile("UAV_001")
        p2 = tracker.get_or_create_profile("UAV_002")

        assert p1.get_exposure(BIN_CLIMB_HIGH_POWER) == pytest.approx(150.0)
        assert p1.get_exposure(BIN_CRUISE_MID_POWER) == 0.0
        # Borrow weight saturates at 1.0 (150s >= 120s target)
        assert p1.compute_borrow_weight(BIN_CLIMB_HIGH_POWER, target_seconds=120.0) == 1.0

        assert p2.get_exposure(BIN_CRUISE_MID_POWER) == pytest.approx(60.0)
        assert p2.get_exposure(BIN_DESCENT_LOW_POWER) == pytest.approx(30.0)
        # Borrow weights: 60/120 = 0.50, 30/120 = 0.25
        assert p2.compute_borrow_weight(BIN_CRUISE_MID_POWER, target_seconds=120.0) == pytest.approx(0.50)
        assert p2.compute_borrow_weight(BIN_DESCENT_LOW_POWER, target_seconds=120.0) == pytest.approx(0.25)
        assert p2.compute_borrow_weight(BIN_CLIMB_HIGH_POWER, target_seconds=120.0) == 0.0

    def test_non_fingerprint_bins_never_accumulate_exposure(self) -> None:
        """
        Verify that IDLE (12), TRANSIENT (13), and INVALID (14) bins are strictly ignored
        by EncounterTracker even if erroneously passed with high stable_duration_s.
        """
        tracker = EncounterTracker()

        for b in [BIN_IDLE, BIN_TRANSIENT, BIN_INVALID, 999]:
            tracker.record_step("UAV_TEST", b, dt=100.0, stable_duration_s=50.0)

        profile = tracker.get_or_create_profile("UAV_TEST")
        for b_id in range(NUM_CORE_BINS):
            assert profile.get_exposure(b_id) == 0.0
            assert profile.compute_borrow_weight(b_id) == 0.0

    def test_exposure_profile_serialization_fidelity(self) -> None:
        """Verify full roundtrip JSON serialization and validation of fleet encounter profiles."""
        tracker = EncounterTracker()
        for b_id in range(NUM_CORE_BINS):
            tracker.record_step("UAV_FLEET_LEADER", b_id, dt=float(b_id * 15), stable_duration_s=25.0)

        json_str = tracker.export_json()
        deserialized = json.loads(json_str)
        assert deserialized["regime_grid_version"] == "REGIME_GRID_V1"
        assert deserialized["aircraft_count"] == 1

        tracker_restored = EncounterTracker()
        tracker_restored.load_json(json_str)
        p_restored = tracker_restored.get_or_create_profile("UAV_FLEET_LEADER")

        for b_id in range(NUM_CORE_BINS):
            expected = float(b_id * 15)
            assert p_restored.get_exposure(b_id) == pytest.approx(expected)
