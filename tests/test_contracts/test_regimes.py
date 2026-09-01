"""
Unit tests for replan_to_learn.contracts.regimes
Verifies REGIME_GRID_V1 12-core + 3 operational bins, the 20-second quasi-steady
stability detection criteria, and aircraft encounter tracking n_i(r).
"""

import json
import pytest
import numpy as np

from replan_to_learn.contracts.regimes import (
    BIN_ALTITUDE_HIGH_POWER,
    BIN_ALTITUDE_LOW_POWER,
    BIN_ALTITUDE_MID_POWER,
    BIN_CLIMB_HIGH_POWER,
    BIN_CLIMB_LOW_POWER,
    BIN_CLIMB_MID_POWER,
    BIN_CRUISE_HIGH_POWER,
    BIN_CRUISE_LOW_POWER,
    BIN_CRUISE_MID_POWER,
    BIN_DESCENT_LOW_POWER,
    BIN_IDLE,
    BIN_INVALID,
    BIN_TRANSIENT,
    CORE_REGIME_BINS,
    MAX_DERIV_CHT_K_PER_S,
    MAX_DERIV_MAP_PA_PER_S,
    MAX_DERIV_RPM_PER_S,
    MIN_STABLE_DURATION_S,
    NUM_CORE_BINS,
    NUM_TOTAL_BINS,
    REGIME_GRID_DEFINITIONS,
    REGIME_GRID_VERSION,
    REGIME_NAMES,
    AircraftExposureProfile,
    EncounterTracker,
    RegimeClassifier,
)


class TestRegimeGridDefinitions:
    """Test permanent REGIME_GRID_V1 structure."""

    def test_version_tag(self) -> None:
        assert REGIME_GRID_VERSION == "REGIME_GRID_V1"

    def test_bin_counts(self) -> None:
        assert len(CORE_REGIME_BINS) == 12
        assert len(REGIME_GRID_DEFINITIONS) == 15
        assert len(REGIME_NAMES) == 15

    def test_core_and_non_core_definitions(self) -> None:
        # Check bins 0..11 are core fingerprints
        for b in range(12):
            assert REGIME_GRID_DEFINITIONS[b].is_core_fingerprint is True
        # Check bins 12..14 are non-fingerprint
        for b in range(12, 15):
            assert REGIME_GRID_DEFINITIONS[b].is_core_fingerprint is False


class TestRegimeClassifierAndQuasiSteadyStability:
    """Test online stability detection and regime classification."""

    def test_quasi_steady_stability_20s_gate(self) -> None:
        classifier = RegimeClassifier(min_stable_duration_s=20.0)

        # Simulate 25 seconds of steady cruise at 5000 RPM, 105 kPa, 365 K, 2000m, 60 kW
        # dt = 1.0s
        for step in range(25):
            t = float(step)
            # Perfect steady state
            rpm = 5000.0
            map_pa = 105000.0
            cht_k = 365.0
            alt_m = 2000.0
            p_brake = 60.0  # 60% MCP -> MID POWER, CRUISE

            bin_id, stable_s = classifier.step(t, rpm, map_pa, cht_k, alt_m, p_brake)

            if step < 20:
                # Prior to 20 seconds, MUST emit TRANSIENT (13)
                assert bin_id == BIN_TRANSIENT
                assert stable_s == pytest.approx(float(step))
            else:
                # At step 20+ (stable duration >= 20.0s), MUST emit CRUISE_MID_POWER (Bin 4)
                assert bin_id == BIN_CRUISE_MID_POWER
                assert stable_s >= 20.0

    def test_transient_derivative_violation_resets_timer(self) -> None:
        classifier = RegimeClassifier(min_stable_duration_s=20.0)

        # 1. Run 22 steady seconds
        for step in range(22):
            bin_id, stable_s = classifier.step(
                t=float(step), rpm=5000.0, map_pa=105000.0, cht_k=365.0, alt_m=2000.0, p_brake=60.0
            )
        assert bin_id == BIN_CRUISE_MID_POWER

        # 2. Step 23: Sudden rapid throttle snap (MAP changes by 10 kPa in 1s > 2 kPa/s threshold)
        bin_id, stable_s = classifier.step(
            t=23.0, rpm=5100.0, map_pa=115000.0, cht_k=365.0, alt_m=2000.0, p_brake=70.0
        )
        assert bin_id == BIN_TRANSIENT
        assert stable_s == 0.0

    def test_climb_and_descent_bin_mapping(self) -> None:
        # Climb: ROC = +5 m/s (Alt 1000 -> 1005 in 1s), High Power (85 kW > 75 kW)
        climb_classifier = RegimeClassifier(min_stable_duration_s=0.0)
        climb_classifier.step(0.0, 5800.0, 150000.0, 370.0, 1000.0, 85.0)
        bin_id, _ = climb_classifier.step(1.0, 5800.0, 150000.0, 370.0, 1005.0, 85.0)
        assert bin_id == BIN_CLIMB_HIGH_POWER

        # Descent: ROC = -5 m/s (Alt 1005 -> 1000 in 1s), Low Power (30 kW < 45 kW)
        descent_classifier = RegimeClassifier(min_stable_duration_s=0.0)
        descent_classifier.step(0.0, 3500.0, 60000.0, 360.0, 1005.0, 30.0)
        bin_id, _ = descent_classifier.step(1.0, 3500.0, 60000.0, 360.0, 1000.0, 30.0)
        assert bin_id == BIN_DESCENT_LOW_POWER

    def test_idle_bin_mapping(self) -> None:
        classifier = RegimeClassifier(min_stable_duration_s=0.0)
        classifier.step(0.0, 1400.0, 40000.0, 330.0, 500.0, 10.0)
        bin_id, _ = classifier.step(1.0, 1400.0, 40000.0, 330.0, 500.0, 10.0)
        assert bin_id == BIN_IDLE


class TestAircraftEncounterTracker:
    """Test tracking n_i(r) and fleet borrow confidence weighting."""

    def test_encounter_accumulation(self) -> None:
        tracker = EncounterTracker()
        # Record 100 seconds in CRUISE_MID_POWER (Bin 4)
        for i in range(100):
            tracker.record_step(
                aircraft_id="UAV_ALPHA",
                regime_bin=BIN_CRUISE_MID_POWER,
                dt=1.0,
                stable_duration_s=25.0,  # stable
                t=float(i)
            )

        profile = tracker.get_or_create_profile("UAV_ALPHA")
        assert profile.get_exposure(BIN_CRUISE_MID_POWER) == pytest.approx(100.0)
        assert profile.get_exposure(BIN_CLIMB_HIGH_POWER) == 0.0

        # Borrow weight: 100 / 120 = ~0.833
        assert profile.compute_borrow_weight(BIN_CRUISE_MID_POWER, target_seconds=120.0) == pytest.approx(100.0 / 120.0, rel=1e-5)
        # Unflown regime has 0 borrow weight
        assert profile.compute_borrow_weight(BIN_CLIMB_HIGH_POWER) == 0.0

    def test_transient_steps_not_accumulated(self) -> None:
        tracker = EncounterTracker()
        # Attempt to record transient steps
        tracker.record_step("UAV_ALPHA", BIN_TRANSIENT, dt=10.0, stable_duration_s=5.0)
        profile = tracker.get_or_create_profile("UAV_ALPHA")
        for b in range(12):
            assert profile.get_exposure(b) == 0.0

    def test_json_export_and_load(self) -> None:
        tracker = EncounterTracker()
        tracker.record_step("UAV_1", BIN_CRUISE_MID_POWER, dt=50.0, stable_duration_s=25.0)
        json_str = tracker.export_json()

        tracker2 = EncounterTracker()
        tracker2.load_json(json_str)
        p = tracker2.get_or_create_profile("UAV_1")
        assert p.get_exposure(BIN_CRUISE_MID_POWER) == pytest.approx(50.0)

    def test_json_load_version_mismatch_raises(self) -> None:
        tracker = EncounterTracker()
        bad_json = json.dumps({"regime_grid_version": "REGIME_GRID_V2_INCOMPATIBLE"})
        with pytest.raises(ValueError, match="Regime grid version mismatch"):
            tracker.load_json(bad_json)

    def test_invalid_telemetry_returns_bin_invalid(self) -> None:
        classifier = RegimeClassifier(min_stable_duration_s=0.0)
        bin_id, dur = classifier.step(0.0, float("nan"), 100000.0, 360.0)
        assert bin_id == BIN_INVALID

        bin_id, dur = classifier.step(1.0, 5000.0, 100000.0, 360.0, is_valid_telemetry=False)
        assert bin_id == BIN_INVALID

    def test_high_altitude_extension_bins(self) -> None:
        classifier = RegimeClassifier(min_stable_duration_s=0.0)
        # Alt = 4000m (>3000m threshold)
        # Low power (<45 kW)
        bin_low = classifier.determine_candidate_bin(5000.0, 30.0, 0.0, 4000.0)
        assert bin_low == BIN_ALTITUDE_LOW_POWER

        # Mid power (60 kW)
        bin_mid = classifier.determine_candidate_bin(5000.0, 60.0, 0.0, 4000.0)
        assert bin_mid == BIN_ALTITUDE_MID_POWER

        # High power (90 kW)
        bin_high = classifier.determine_candidate_bin(5000.0, 90.0, 0.0, 4000.0)
        assert bin_high == BIN_ALTITUDE_HIGH_POWER
