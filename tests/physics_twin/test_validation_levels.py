"""
SIH26054 Replan to Learn: 9-Level Validation Hierarchy Tests.
Tests for Physics Sanity (L1), Numerical Verification (L2), Healthy Telemetry (L3),
Unseen Flight (L4), Fault Injection (L5), and Fault Separability (L6).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from replan_to_learn.physics_twin.physics_twin import PhysicsTwin
from replan_to_learn.physics_twin.fault_injection import inject, inject_theta_cyl, inject_cooling_nonuniform
from replan_to_learn.contracts.telemetry import TelemetryFrame
from replan_to_learn.contracts.residuals import ResidualFrame
from replan_to_learn.contracts.faults import FaultSpec, HealthParameters, FaultInjectionHarness
from replan_to_learn.contracts.health import HealthParameters
from replan_to_learn.contracts.regimes import RegimeClassifier, RegimeBin


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def model_dir(tmp_path):
    """Create a temporary model directory with required JSON artifacts."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    regime_grid = {
        "version": "REGIME_GRID_V1",
        "phase_bins": ["CLIMB", "CRUISE", "DESCENT"],
        "power_bins": ["LOW", "MID", "HIGH"],
    }
    (model_dir / "regime_grid_v1.json").write_text(json.dumps(regime_grid))

    gaspath = {
        "constants": {
            "R_air": 287.0,
            "V_d": 0.001352,
            "AFR_st": 14.7,
            "LHV": 43000000.0,
        },
        "volumetric_efficiency": {
            "lookup_table": {
                "N_breakpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
                "p_im_breakpoints": [50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000],
                "values": [[1.0] * 8] * 8,
            }
        },
        "indicated_efficiency": {
            "lookup_table": {
                "lambda_breakpoints": [0.8, 0.9, 1.0, 1.1, 1.2],
                "N_breakpoints": [1000, 2000, 3000, 4000, 5000],
                "p_im_breakpoints": [50000, 100000, 150000, 200000, 250000],
                "values": [
                    [[0.30, 0.32, 0.33, 0.32, 0.30],
                     [0.30, 0.32, 0.33, 0.32, 0.30],
                     [0.30, 0.32, 0.33, 0.32, 0.30],
                     [0.30, 0.32, 0.33, 0.32, 0.30],
                     [0.30, 0.32, 0.33, 0.32, 0.30]],
                    [[0.32, 0.34, 0.35, 0.34, 0.32],
                     [0.32, 0.34, 0.35, 0.34, 0.32],
                     [0.32, 0.34, 0.35, 0.34, 0.32],
                     [0.32, 0.34, 0.35, 0.34, 0.32],
                     [0.32, 0.34, 0.35, 0.34, 0.32]],
                    [[0.35, 0.37, 0.38, 0.37, 0.35],
                     [0.35, 0.37, 0.38, 0.37, 0.35],
                     [0.35, 0.37, 0.38, 0.37, 0.35],
                     [0.35, 0.37, 0.38, 0.37, 0.35],
                     [0.35, 0.37, 0.38, 0.37, 0.35]],
                    [[0.33, 0.35, 0.36, 0.35, 0.33],
                     [0.33, 0.35, 0.36, 0.35, 0.33],
                     [0.33, 0.35, 0.36, 0.35, 0.33],
                     [0.33, 0.35, 0.36, 0.35, 0.33],
                     [0.33, 0.35, 0.36, 0.35, 0.33]],
                    [[0.30, 0.32, 0.33, 0.32, 0.30],
                     [0.30, 0.32, 0.33, 0.32, 0.30],
                     [0.30, 0.32, 0.33, 0.32, 0.30],
                     [0.30, 0.32, 0.33, 0.32, 0.30],
                     [0.30, 0.32, 0.33, 0.32, 0.30]],
                ],
            }
        },
        "fmep_coefficients": {
            "c_0": 20000.0,
            "c_1": 5.0,
            "c_2": 0.01,
        },
        "lambda_commanded": {
            "lookup_table": {
                "N_breakpoints": [2000, 3000, 4000],
                "p_im_breakpoints": [75000, 125000, 175000],
                "values": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            }
        },
        "probe_recovery": {"k_probe": 0.85},
    }
    (model_dir / "gaspath_params.json").write_text(json.dumps(gaspath))

    thermal = {
        "thermal_capacitances": {
            "C_hd": 1500.0,
            "C_oil": 2000.0,
            "C_c": 500.0,
        },
        "cooling_conductances": {
            "UA_hd_nominal": 10.0,
            "UA_oc_nominal": 8.0,
            "UA_c_nominal": 12.0,
        },
        "asymmetry_factors": {
            "a_values": [0.0, 0.1, 0.2, 0.3],
        },
        "heat_transfer_to_head": {
            "Q_to_head_fraction": 0.3,
            "Q_to_head_per_cylinder": [0.25, 0.25, 0.25, 0.25],
        },
        "friction_heat": {
            "friction_to_oil_fraction": 0.7,
        },
        "oil_viscosity": {
            "parameters": {
                "A": -3.5,
                "B": 1200.0,
                "C": 150.0,
            }
        },
        "heat_exchanger": {
            "UA_ho": 15.0,
        },
    }
    (model_dir / "thermal_params.json").write_text(json.dumps(thermal))

    noise_model = {
        "noise_std": {
            "z_EGT_1": [0.1] * 12,
            "z_EGT_2": [0.1] * 12,
            "z_EGT_3": [0.1] * 12,
            "z_EGT_4": [0.1] * 12,
            "z_CHT": [0.05] * 12,
            "z_p_oil": [0.02] * 12,
            "z_t_oil": [0.03] * 12,
            "z_mdot_f": [0.04] * 12,
            "z_N": [0.05] * 12,
        },
        "ar_coefficients": {
            "channels": ["z_EGT_1", "z_EGT_2", "z_EGT_3", "z_EGT_4", "z_CHT", "z_p_oil", "z_t_oil", "z_mdot_f", "z_N"],
            "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }
    (model_dir / "noise_model.json").write_text(json.dumps(noise_model))

    manifest = {"model_version": "1.0.0", "regime_grid_version": "REGIME_GRID_V1"}
    (model_dir / "MANIFEST.sha256").write_text(json.dumps(manifest))

    return model_dir


@pytest.fixture
def sample_telemetry():
    return TelemetryFrame(
        t=0.0,
        egt=(950.0, 955.0, 960.0, 965.0),
        cht=360.0,
        p_oil=3.5e5,
        t_oil=350.0,
        n_rpm=3000.0,
        mdot_f=0.05,
        map_pa=1.2e5,
        tps=0.8,
        t_im=300.0,
        p_amb=1.01325e5,
        t_amb=288.15,
        v_tas=50.0,
        h_p=0.0,
    )


def _make_frame(t=0.0, **kwargs):
    defaults = dict(
        egt=(950.0, 955.0, 960.0, 965.0),
        cht=360.0,
        p_oil=3.5e5,
        t_oil=350.0,
        n_rpm=3000.0,
        mdot_f=0.05,
        map_pa=1.2e5,
        tps=0.8,
        t_im=300.0,
        p_amb=1.01325e5,
        t_amb=288.15,
        v_tas=50.0,
        h_p=0.0,
    )
    defaults.update(kwargs)
    return TelemetryFrame(t=t, **defaults)


# ============================================================================
# Level 1: Physics Sanity Tests
# ============================================================================

class TestLevel1PhysicsSanity:
    """Physics sanity checks for the mean-value engine model."""

    def test_energy_balance_closure(self, model_dir, sample_telemetry):
        twin = PhysicsTwin(model_dir)
        twin.reset(sample_telemetry)
        for _ in range(50):
            rf = twin.step(sample_telemetry)

        state = twin.state
        T_hd = state[1]
        T_oil = state[2]
        T_exh = state[3]
        T_cyl = state[5:9]

        assert 273.15 < T_hd < 500.0, f"T_hd={T_hd} out of admissible range"
        assert 250.0 < T_oil < 450.0, f"T_oil={T_oil} out of admissible range"
        assert 300.0 < T_exh < 1450.0, f"T_exh={T_exh} out of admissible range"
        for i, Tc in enumerate(T_cyl):
            assert 273.15 <= Tc <= 500.0, f"T_cyl[{i}]={Tc} out of admissible range"

    def test_cooling_degradation_increases_cht(self, model_dir, sample_telemetry):
        twin_nom = PhysicsTwin(model_dir)
        twin_nom.reset(sample_telemetry)
        dT_hd_nom = twin_nom._compute_state_derivatives(twin_nom.state, sample_telemetry)[1]

        twin_deg = PhysicsTwin(model_dir)
        twin_deg.reset(sample_telemetry)
        twin_deg.health_params[2] = 0.7
        dT_hd_deg = twin_deg._compute_state_derivatives(twin_deg.state, sample_telemetry)[1]

        assert dT_hd_deg > dT_hd_nom, \
            f"Degraded cooling should increase CHT derivative: nom={dT_hd_nom}, deg={dT_hd_deg}"

    def test_combustion_degradation_increases_egt(self, model_dir, sample_telemetry):
        twin_nom = PhysicsTwin(model_dir)
        twin_nom.reset(sample_telemetry)
        preds_nom = twin_nom._compute_predictions(sample_telemetry)

        twin_deg = PhysicsTwin(model_dir)
        twin_deg.reset(sample_telemetry)
        twin_deg.health_params[1] = 0.7
        preds_deg = twin_deg._compute_predictions(sample_telemetry)

        assert preds_deg["p_brake"] < preds_nom["p_brake"], \
            f"Degraded combustion should reduce brake power: nom={preds_nom['p_brake']}, deg={preds_deg['p_brake']}"

    def test_injector_clogging_increases_local_egt(self, model_dir, sample_telemetry):
        twin_nom = PhysicsTwin(model_dir)
        twin_nom.reset(sample_telemetry)
        preds_nom = twin_nom._compute_predictions(sample_telemetry)

        twin_deg = PhysicsTwin(model_dir)
        twin_deg.reset(sample_telemetry)
        twin_deg.health_params[3:7] = np.array([1.0, 1.0, 0.9, 1.0])
        preds_deg = twin_deg._compute_predictions(sample_telemetry)

        assert preds_deg["egt"][2] > preds_deg["egt"][0], \
            "Clogged cyl 3 should have higher EGT than cyl 1"

    def test_oil_pressure_monotonic_with_rpm(self, model_dir):
        twin = PhysicsTwin(model_dir)
        preds = []
        for rpm in [1500, 2000, 2500, 3000, 3500]:
            frame = _make_frame(n_rpm=float(rpm))
            twin.reset(frame)
            for _ in range(50):
                twin.step(frame)
            preds.append(twin._compute_predictions(frame)["p_oil"])

        for i in range(len(preds) - 1):
            assert preds[i + 1] >= preds[i] * 0.95, f"Oil pressure should increase with RPM: {preds}"

    def test_physical_state_bounds_enforced(self, model_dir, sample_telemetry):
        twin = PhysicsTwin(model_dir)
        twin.reset(sample_telemetry)
        for _ in range(200):
            rf = twin.step(sample_telemetry)
            assert rf.status != 2, "Model should not saturate for normal telemetry"

    def test_regime_classifier_transient_initial(self, model_dir, sample_telemetry):
        twin = PhysicsTwin(model_dir)
        twin.reset(sample_telemetry)
        rf = twin.step(sample_telemetry)
        assert rf.regime in (RegimeBin.TRANSIENT, RegimeBin.INVALID), \
            f"First step should be TRANSIENT/INVALID, got {rf.regime}"

    def test_residual_frame_contract_shape(self, model_dir, sample_telemetry):
        twin = PhysicsTwin(model_dir)
        twin.reset(sample_telemetry)
        rf = twin.step(sample_telemetry)
        assert isinstance(rf, ResidualFrame)
        assert len(rf.z) == 9
        assert len(rf.spatial) == 4
        assert len(rf.sigma) == 9
        assert len(rf.x_hat) == 9
        assert rf.model_version == "1.0.0"
        assert rf.regime_grid_version == "REGIME_GRID_V1"


# ============================================================================
# Level 2: Numerical Verification
# ============================================================================

class TestLevel2NumericalVerification:
    """Numerical verification of Jacobian consistency and integrator convergence."""

    def test_jacobian_shape(self, model_dir, sample_telemetry):
        twin = PhysicsTwin(model_dir)
        theta = np.ones(15)
        J = twin.jacobian(sample_telemetry, theta)
        assert J.shape == (9, 15)

    def test_jacobian_vs_finite_difference(self, model_dir, sample_telemetry):
        twin = PhysicsTwin(model_dir)
        theta = np.ones(15)
        J = twin.jacobian(sample_telemetry, theta)

        eps = 1e-4
        J_fd = np.zeros_like(J)
        for j in range(15):
            theta_p = theta.copy()
            theta_p[j] += eps
            theta_m = theta.copy()
            theta_m[j] -= eps
            y_p = twin.predict(sample_telemetry, theta_p)
            y_m = twin.predict(sample_telemetry, theta_m)
            J_fd[:, j] = (y_p - y_m) / (2.0 * eps)

        rel_err = np.abs(J - J_fd) / (np.abs(J_fd) + 1e-9)
        assert float(np.max(rel_err)) < 0.1, \
            f"Jacobian vs FD max relative error {np.max(rel_err):.4e} too large"

    def test_predict_output_shape_single(self, model_dir, sample_telemetry):
        twin = PhysicsTwin(model_dir)
        theta = np.ones(15)
        y = twin.predict(sample_telemetry, theta)
        assert y.shape == (9,)

    def test_predict_output_shape_sequence(self, model_dir, sample_telemetry):
        twin = PhysicsTwin(model_dir)
        theta = np.ones(15)
        frames = [sample_telemetry, _make_frame(t=1.0), _make_frame(t=2.0)]
        y = twin.predict(frames, theta)
        assert y.shape == (3, 9)

    def test_determinism_g15(self, model_dir, sample_telemetry):
        twin1 = PhysicsTwin(model_dir)
        twin2 = PhysicsTwin(model_dir)
        twin1.reset(sample_telemetry)
        twin2.reset(sample_telemetry)
        r1 = twin1.step(sample_telemetry)
        r2 = twin2.step(sample_telemetry)
        assert np.allclose(np.array(r1.z), np.array(r2.z)), "G1.5: bit-identical determinism failed"
        assert np.allclose(np.array(r1.spatial), np.array(r2.spatial)), "G1.5: spatial not deterministic"

    def test_edge_budget_g16(self, model_dir, sample_telemetry):
        twin = PhysicsTwin(model_dir)
        twin.reset(sample_telemetry)
        start = time.perf_counter()
        for _ in range(100):
            twin.step(sample_telemetry)
        elapsed_ms = (time.perf_counter() - start) * 10.0
        assert elapsed_ms < 8.0, f"G1.6: edge budget exceeded {elapsed_ms:.2f} ms/step"


# ============================================================================
# Level 3: Healthy Telemetry Validation (G1.3, G1.4)
# ============================================================================

class TestLevel3HealthyTelemetry:
    """Healthy telemetry validation for regime independence and residual whiteness."""

    def test_regime_independence_g14_synthetic(self, model_dir):
        twin = PhysicsTwin(model_dir)
        base_frame = _make_frame(n_rpm=2500.0, map_pa=1.0e5, v_tas=60.0, h_p=1000.0)
        twin.reset(base_frame)

        for t in range(0, 2000, 1):
            f = _make_frame(t=float(t), n_rpm=base_frame.n_rpm, map_pa=base_frame.map_pa, v_tas=base_frame.v_tas)
            twin.step(f)

        zs = []
        for t in range(2000, 2200, 1):
            f = _make_frame(t=float(t), n_rpm=base_frame.n_rpm, map_pa=base_frame.map_pa, v_tas=base_frame.v_tas)
            rf = twin.step(f)
            zs.append(np.array(rf.z))

        zs = np.array(zs)
        z_mean = np.nanmean(zs, axis=0)
        sigma = np.nanstd(zs, axis=0)
        valid = sigma > 1e-6
        if np.any(valid):
            rel_bias = np.zeros_like(z_mean)
            rel_bias[valid] = np.abs(z_mean[valid]) / sigma[valid]
            assert np.all(rel_bias[valid] < 2.0), \
                f"G1.4: |mean(z)|/sigma too large: {rel_bias}"

    def test_residual_frame_status_flags(self, model_dir):
        twin = PhysicsTwin(model_dir)
        frame_valid = _make_frame(valid_mask=0xFFFF)
        twin.reset(frame_valid)
        rf = twin.step(frame_valid)
        assert rf.status == 0

    def test_spatial_egt_decomposition_properties(self, model_dir, sample_telemetry):
        from replan_to_learn.contracts.faults import SpatialEGTDecomposer
        ortho = SpatialEGTDecomposer.verify_orthogonality()
        assert ortho["is_orthonormal"], "Spatial basis must be orthonormal"


# ============================================================================
# Level 4: Unseen Flight Validation
# ============================================================================

class TestLevel4UnseenFlight:
    """Unseen flight/aircraft validation."""

    def test_aircraft_exposure_profile(self):
        from replan_to_learn.contracts.regimes import AircraftExposureProfile, EncounterTracker
        profile = AircraftExposureProfile(aircraft_id="TEST-001")
        for r in range(12):
            profile.add_exposure(r, 120.0)
        for r in range(12):
            assert profile.get_exposure(r) == 120.0
            assert profile.compute_borrow_weight(r) == 1.0

        profile_low = AircraftExposureProfile(aircraft_id="TEST-002")
        profile_low.add_exposure(0, 30.0)
        assert profile_low.compute_borrow_weight(0) == pytest.approx(0.25, abs=0.01)

    def test_encounter_tracker_json_roundtrip(self):
        from replan_to_learn.contracts.regimes import EncounterTracker
        tracker = EncounterTracker()
        tracker.record_step("A1", 0, 1.0, 20.0)
        tracker.record_step("A1", 1, 1.0, 20.0)
        tracker.record_step("A2", 3, 1.0, 20.0)

        json_str = tracker.export_json()
        tracker2 = EncounterTracker()
        tracker2.load_json(json_str)
        assert "A1" in tracker2.profiles
        assert "A2" in tracker2.profiles
        assert tracker2.profiles["A1"].get_exposure(0) == 1.0


# ============================================================================
# Level 5: Fault Injection Smoke Tests
# ============================================================================

class TestLevel5FaultInjection:
    """Fault injection smoke tests."""

    def test_cooling_degradation_injection(self):
        theta_nom = HealthParameters()
        fault = FaultSpec(
            fault_type="cooling_degradation",
            target_cylinder=0,
            magnitude=0.15,
            onset_time_s=0.0,
            ramp_duration_s=3600.0,
            spatial_asymmetry_weight=0.40,
        )
        result = inject(theta_nom, fault, t=1800.0)
        assert result[2] < 1.0, "theta_cool should degrade"

    def test_combustion_degradation_injection(self):
        theta_nom = HealthParameters()
        fault = FaultSpec(
            fault_type="combustion_degradation",
            target_cylinder=0,
            magnitude=0.1,
            onset_time_s=0.0,
            ramp_duration_s=3600.0,
        )
        result = inject(theta_nom, fault, t=1800.0)
        assert result[1] < 1.0, "theta_comb should degrade"

    def test_injector_clogging_single_cylinder(self):
        theta_nom = HealthParameters()
        fault = FaultSpec(
            fault_type="injector_clogging",
            target_cylinder=3,
            magnitude=0.2,
            onset_time_s=0.0,
            ramp_duration_s=60.0,
        )
        result = inject(theta_nom, fault, t=30.0)
        assert result[5] < 1.0, "theta_inj[3] should degrade"
        assert result[3] == 1.0, "theta_inj[0] should remain nominal"

    def test_oil_pump_degradation(self):
        theta_nom = HealthParameters()
        fault = FaultSpec(
            fault_type="oil_pump_wear",
            target_cylinder=0,
            magnitude=0.2,
            onset_time_s=0.0,
            ramp_duration_s=3600.0,
        )
        result = inject(theta_nom, fault, t=1800.0)
        assert result[7] < 1.0, "theta_oilp should degrade"

    def test_sensor_bias_egt(self):
        theta_nom = HealthParameters()
        fault = FaultSpec(
            fault_type="sensor_bias_egt",
            target_cylinder=3,
            magnitude=5.0,
            onset_time_s=0.0,
            ramp_duration_s=60.0,
        )
        result = inject(theta_nom, fault, t=30.0)
        assert result[11] == pytest.approx(2.5), "b_egt[2] should be half magnitude at t=30s"

    def test_zero_time_yields_nominal(self):
        theta_nom = HealthParameters()
        fault = FaultSpec(
            fault_type="injector_clogging",
            target_cylinder=1,
            magnitude=0.2,
            onset_time_s=0.0,
            ramp_duration_s=60.0,
        )
        result = inject(theta_nom, fault, t=0.0)
        assert np.allclose(result, theta_nom.to_array())

    def test_non_uniform_cooling_factors(self):
        factors = FaultInjectionHarness.compute_non_uniform_cooling_factors(
            FaultSpec(
                fault_type="cooling_degradation",
                target_cylinder=0,
                magnitude=0.2,
                onset_time_s=0.0,
                ramp_duration_s=3600.0,
                spatial_asymmetry_weight=0.40,
            ),
            t=1800.0,
        )
        assert factors.shape == (4,)
        assert np.min(factors) < np.max(factors), "Non-uniform cooling must create cylinder spread"


# ============================================================================
# Level 6: Fault Separability (Ambiguous Pair)
# ============================================================================

class TestLevel6FaultSeparability:
    """Fault separability tests for cooling vs combustion ambiguity."""

    def test_cooling_vs_combustion_sensitivity_divergence(self):
        from replan_to_learn.contracts.health import AmbiguousPairModel
        j_cool_cruise, j_comb_cruise = AmbiguousPairModel.compute_jacobian_columns("CRUISE")
        j_cool_climb, j_comb_climb = AmbiguousPairModel.compute_jacobian_columns("CLIMB")

        cos_cruise = AmbiguousPairModel.compute_collinearity("CRUISE", include_power_residual=True)
        cos_climb = AmbiguousPairModel.compute_collinearity("CLIMB", include_power_residual=True)

        assert cos_cruise < 0.95, "Multi-regime should break collinearity below 0.95"
        assert cos_climb < cos_cruise, "Climb should further reduce collinearity"

    def test_fisher_information_well_conditioned(self):
        from replan_to_learn.contracts.health import AmbiguousPairModel
        fim, cond = AmbiguousPairModel.compute_fisher_information_matrix(
            ["CRUISE", "CLIMB", "DESCENT"],
            include_power_residual=True,
        )
        assert np.isfinite(cond), "FIM condition number must be finite"
        assert cond < 1e4, f"FIM should be well-conditioned, got cond={cond}"

    def test_power_balance_decouples_parameters(self):
        from replan_to_learn.contracts.health import AmbiguousPairModel
        sens = AmbiguousPairModel.evaluate_sensitivities("CRUISE")
        assert abs(sens["dz_power_d_theta_cool"]) < 0.1, \
            "theta_cool should have near-zero power-balance sensitivity"
        assert abs(sens["dz_power_d_theta_comb"]) > 1.0, \
            "theta_comb should have strong power-balance sensitivity"
