"""
Unit tests for replan_to_learn.contracts.faults
Verifies fault injection harness (non-uniform cooling w >= 0.35, injector faults,
sensor biases) and spatial EGT orthogonal basis decomposition (alpha, beta, s_t).
"""

import pytest
import numpy as np

from replan_to_learn.contracts.health import HealthParameters
from replan_to_learn.contracts.faults import (
    BASIS_COMMON_MODE,
    BASIS_GRADIENT,
    NOMINAL_CYL_AERODYNAMIC_ASYMMETRY,
    FaultInjectionHarness,
    FaultSpec,
    SpatialEGTDecomposer,
)


class TestFaultSpecAndInjection:
    """Test fault specifications and parameter degradation models."""

    def test_cooling_fault_requires_non_uniform_asymmetry(self) -> None:
        # w < 0.35 MUST raise ValueError to prevent unrealistic uniform simplification
        with pytest.raises(ValueError, match="w >= 0.35"):
            FaultSpec(
                fault_type="cooling_degradation",
                magnitude=0.15,
                spatial_asymmetry_weight=0.20,  # Invalid: < 0.35
            )

        # w >= 0.35 is valid
        f_valid = FaultSpec(
            fault_type="cooling_degradation",
            magnitude=0.15,
            spatial_asymmetry_weight=0.40,
        )
        assert f_valid.spatial_asymmetry_weight == 0.40

    def test_cooling_degradation_injection(self, nominal_health_parameters: HealthParameters) -> None:
        fault = FaultSpec(
            fault_type="cooling_degradation",
            magnitude=0.15,
            onset_time_s=100.0,
            ramp_duration_s=50.0,
            spatial_asymmetry_weight=0.40,
        )

        # Before onset: nominal
        hp_0 = FaultInjectionHarness.inject(nominal_health_parameters, fault, t=50.0)
        assert hp_0.theta_cool == pytest.approx(1.0)

        # Mid-ramp (t = 125s, 50% ramp): theta_cool drops by 0.075 -> 0.925
        hp_mid = FaultInjectionHarness.inject(nominal_health_parameters, fault, t=125.0)
        assert hp_mid.theta_cool == pytest.approx(0.925, rel=1e-4)

        # Fully degraded (t = 200s): theta_cool drops by 0.15 -> 0.85
        hp_full = FaultInjectionHarness.inject(nominal_health_parameters, fault, t=200.0)
        assert hp_full.theta_cool == pytest.approx(0.85, rel=1e-4)

        # Verify non-uniform per-cylinder conductance factors
        ua_cyl = FaultInjectionHarness.compute_non_uniform_cooling_factors(fault, t=200.0)
        assert len(ua_cyl) == 4
        # Cylinders 1 and 2 (front) have different cooling delta than 3 and 4 (rear)
        assert ua_cyl[0] != ua_cyl[3]

    def test_single_cylinder_injector_clogging(self, nominal_health_parameters: HealthParameters) -> None:
        fault = FaultSpec(
            fault_type="injector_clogging",
            target_cylinder=2,  # Cylinder 2 (index 1)
            magnitude=0.10,     # 10% delivery drop
            onset_time_s=0.0,
            ramp_duration_s=0.0,
        )
        hp = FaultInjectionHarness.inject(nominal_health_parameters, fault, t=10.0)
        assert hp.theta_inj[0] == pytest.approx(1.0)
        assert hp.theta_inj[1] == pytest.approx(0.90)  # Degraded
        assert hp.theta_inj[2] == pytest.approx(1.0)
        assert hp.theta_inj[3] == pytest.approx(1.0)

    def test_sensor_bias_injection(self, nominal_health_parameters: HealthParameters) -> None:
        fault = FaultSpec(
            fault_type="sensor_bias_egt",
            target_cylinder=3,
            magnitude=15.0,  # +15 K drift
            onset_time_s=0.0,
            ramp_duration_s=0.0,
        )
        hp = FaultInjectionHarness.inject(nominal_health_parameters, fault, t=10.0)
        assert hp.b_egt[2] == pytest.approx(15.0)
        assert hp.b_egt[0] == 0.0


    def test_all_cylinders_injector_clogging(self, nominal_health_parameters: HealthParameters) -> None:
        fault = FaultSpec(
            fault_type="injector_clogging",
            target_cylinder=0,  # All cylinders
            magnitude=0.10,
            onset_time_s=0.0,
            ramp_duration_s=0.0,
        )
        hp = FaultInjectionHarness.inject(nominal_health_parameters, fault, t=10.0)
        assert all(inj == pytest.approx(0.90) for inj in hp.theta_inj)

    def test_all_cylinders_sensor_bias_egt(self, nominal_health_parameters: HealthParameters) -> None:
        fault = FaultSpec(
            fault_type="sensor_bias_egt",
            target_cylinder=0,
            magnitude=10.0,
            onset_time_s=0.0,
            ramp_duration_s=0.0,
        )
        hp = FaultInjectionHarness.inject(nominal_health_parameters, fault, t=10.0)
        assert all(b == pytest.approx(10.0) for b in hp.b_egt)

    def test_sensor_bias_cht_and_poil(self, nominal_health_parameters: HealthParameters) -> None:
        f_cht = FaultSpec(fault_type="sensor_bias_cht", magnitude=5.0, onset_time_s=0.0, ramp_duration_s=0.0)
        hp_cht = FaultInjectionHarness.inject(nominal_health_parameters, f_cht, t=1.0)
        assert hp_cht.b_cht == pytest.approx(5.0)

        f_poil = FaultSpec(fault_type="sensor_bias_poil", magnitude=10000.0, onset_time_s=0.0, ramp_duration_s=0.0)
        hp_poil = FaultInjectionHarness.inject(nominal_health_parameters, f_poil, t=1.0)
        assert hp_poil.b_poil == pytest.approx(10000.0)

    def test_oil_pump_wear_and_friction_increase(self, nominal_health_parameters: HealthParameters) -> None:
        f_oil = FaultSpec(fault_type="oil_pump_wear", magnitude=0.15, onset_time_s=0.0, ramp_duration_s=0.0)
        hp_oil = FaultInjectionHarness.inject(nominal_health_parameters, f_oil, t=1.0)
        assert hp_oil.theta_oilp == pytest.approx(0.85)

        f_fric = FaultSpec(fault_type="friction_increase", magnitude=0.20, onset_time_s=0.0, ramp_duration_s=0.0)
        hp_fric = FaultInjectionHarness.inject(nominal_health_parameters, f_fric, t=1.0)
        assert hp_fric.theta_fric == pytest.approx(1.20)

    def test_combustion_degradation_injection(self, nominal_health_parameters: HealthParameters) -> None:
        f_comb = FaultSpec(fault_type="combustion_degradation", magnitude=0.10, onset_time_s=0.0, ramp_duration_s=0.0)
        hp_comb = FaultInjectionHarness.inject(nominal_health_parameters, f_comb, t=1.0)
        assert hp_comb.theta_comb == pytest.approx(0.90)

    def test_unsupported_fault_type(self, nominal_health_parameters: HealthParameters) -> None:
        f_bad = FaultSpec(fault_type="warp_core_breach", magnitude=0.10, onset_time_s=0.0, ramp_duration_s=0.0)
        with pytest.raises(ValueError, match="Unsupported fault type"):
            FaultInjectionHarness.inject(nominal_health_parameters, f_bad, t=1.0)

    def test_invalid_target_cylinder(self) -> None:
        with pytest.raises(ValueError, match="target_cylinder"):
            FaultSpec(fault_type="injector_clogging", target_cylinder=5)


class TestSpatialEGTDecomposer:
    """Test orthogonal basis decomposition and projection."""

    def test_orthonormal_basis_properties(self) -> None:
        ortho_check = SpatialEGTDecomposer.verify_orthogonality()
        assert ortho_check["is_orthonormal"] is True
        assert ortho_check["norm_common_mode"] == pytest.approx(1.0, abs=1e-12)
        assert ortho_check["norm_gradient"] == pytest.approx(1.0, abs=1e-12)
        assert abs(ortho_check["dot_common_gradient"]) < 1e-12

    def test_invalid_input_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 4 EGT residuals"):
            SpatialEGTDecomposer.decompose([1.0, 2.0])

    def test_all_nan_handling(self) -> None:
        alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose([float("nan")] * 4)
        assert alpha == 0.0
        assert beta == 0.0
        assert norm_inf == 0.0
        assert dom_cyl == 0

    def test_partial_nan_handling(self) -> None:
        alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose([10.0, float("nan"), 10.0, float("nan")])
        assert alpha == pytest.approx(10.0)
        assert dom_cyl == 0

    def test_pure_common_mode_decomposition(self) -> None:
        # Uniform EGT residual: z = [2.0, 2.0, 2.0, 2.0]
        # Projected onto 1 = [0.5, 0.5, 0.5, 0.5] -> alpha = 4.0
        # Gradient projection -> beta = 0.0
        # Sparse residual -> s_t = 0
        z = np.array([2.0, 2.0, 2.0, 2.0])
        alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose(z)

        assert alpha == pytest.approx(4.0, abs=1e-6)
        assert beta == pytest.approx(0.0, abs=1e-6)
        assert norm_inf == pytest.approx(0.0, abs=1e-6)

    def test_pure_fore_aft_cooling_gradient_decomposition(self) -> None:
        # Pure gradient pattern proportional to BASIS_GRADIENT
        z = 5.0 * BASIS_GRADIENT
        alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose(z)

        assert alpha == pytest.approx(0.0, abs=1e-6)
        assert beta == pytest.approx(5.0, abs=1e-6)
        assert norm_inf == pytest.approx(0.0, abs=1e-6)

    def test_isolated_single_cylinder_fault_localization(self) -> None:
        # Cylinder 3 has severe lean condition: z = [0.0, 0.0, 6.0, 0.0]
        z = np.array([0.0, 0.0, 6.0, 0.0])
        alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose(z)

        # Decomposes into common-mode + gradient + sparse component
        assert alpha > 0.0
        assert norm_inf > 0.0
        # Dominant cylinder MUST correctly identify cylinder 3
        assert dom_cyl == 3

    def test_sparse_residual_orthogonality(self) -> None:
        # For arbitrary residual vector, verify s_t is orthogonal to 1 and g
        z = np.array([1.2, -3.4, 5.6, -0.8])
        alpha, beta, _, _ = SpatialEGTDecomposer.decompose(z)

        # Reconstruct s_t
        s_t = z - (alpha * BASIS_COMMON_MODE + beta * BASIS_GRADIENT)

        assert np.dot(s_t, BASIS_COMMON_MODE) == pytest.approx(0.0, abs=1e-12)
        assert np.dot(s_t, BASIS_GRADIENT) == pytest.approx(0.0, abs=1e-12)
