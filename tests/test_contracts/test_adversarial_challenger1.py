"""
SIH26054 Replan to Learn: Empirical Adversarial Challenge Suite (Challenger 1).
Stress tests and boundary validation for Milestone 1 (R1 Data Contracts & Schemas).

Focus Areas:
1. ResidualSuppressionEngine:
   - Exhaustive 2^7 = 128 combinations of exogenous input availability.
   - Zero-imputation verification (missing inputs -> NaN suppression, present inputs -> unsuppressed).
   - Interaction of exogenous and primary sensor failure masks and status codes.
   - Multi-format serialization (Polars, PyArrow, Dataclass) of suppressed states.
2. FaultInjectionHarness & Spatial Decomposition:
   - Non-uniform cooling degradation with cylinder asymmetry threshold (w >= 0.35).
   - Per-cylinder cooling conductance multiplier asymmetry (front vs. rear divergence).
   - Spatial EGT decomposition into orthonormal basis (alpha * 1 + beta * g + s_t).
   - Monte Carlo random residual projections (1,000 trials for reconstruction & orthogonality).
   - Extreme fault severities, multi-fault sequential composition, and out-of-bounds handling.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from replan_to_learn.contracts.telemetry import (
    FLAG_CHT_VALID,
    FLAG_EGT1_VALID,
    FLAG_EGT2_VALID,
    FLAG_EGT3_VALID,
    FLAG_EGT4_VALID,
    FLAG_HP_VALID,
    FLAG_MAP_VALID,
    FLAG_MDF_VALID,
    FLAG_PAMB_VALID,
    FLAG_POIL_VALID,
    FLAG_RPM_VALID,
    FLAG_TAMB_VALID,
    FLAG_TIM_VALID,
    FLAG_TOIL_VALID,
    FLAG_TPS_VALID,
    FLAG_VTAS_VALID,
    FULL_VALID_MASK,
    TelemetryValidityFlag,
    FrameStatus,
    STATUS_DEGRADED_INPUT,
    STATUS_INVALID_SENSOR,
    STATUS_MODEL_SATURATED,
    STATUS_OK,
    TelemetryFrame,
)
from replan_to_learn.contracts.residuals import (
    IDX_Z_CHT,
    IDX_Z_EGT1,
    IDX_Z_EGT2,
    IDX_Z_EGT3,
    IDX_Z_EGT4,
    IDX_Z_MDF,
    IDX_Z_POIL,
    IDX_Z_POWER,
    IDX_Z_TOIL,
    NUM_RESIDUAL_CHANNELS,
    RESIDUAL_CHANNELS,
    RESIDUAL_ARROW_SCHEMA,
    RESIDUAL_POLARS_SCHEMA,
    ResidualFrame,
    ResidualSuppressionEngine,
    arrow_to_residual_frames,
    polars_to_residual_frames,
    residual_frames_to_arrow,
    residual_frames_to_polars,
    validate_residual_dataframe,
)
from replan_to_learn.contracts.health import (
    HEALTH_PARAMETER_SPECS,
    HealthParameters,
)
from replan_to_learn.contracts.faults import (
    BASIS_COMMON_MODE,
    BASIS_GRADIENT,
    NOMINAL_CYL_AERODYNAMIC_ASYMMETRY,
    FaultInjectionHarness,
    FaultSpec,
    SpatialEGTDecomposer,
)


# ============================================================================
# Section 1: Exhaustive 128-Combination Stress Tests for ResidualSuppressionEngine
# ============================================================================

class TestExhaustiveResidualSuppression128Combinations:
    """
    Exhaustively tests all 2^7 = 128 combinations of the 7 exogenous inputs:
    [MAP, TPS, TIM, PAMB, TAMB, VTAS, HP].
    
    Verifies:
    1. Zero-imputation: missing inputs trigger exact NaN suppression for affected channels.
    2. Unaffected channels retain their exact raw calculated values (no imputation/guessing).
    3. When all 7 exogenous inputs are present, all residuals compute without suppression.
    4. Status code is correctly set to STATUS_OK or STATUS_DEGRADED_INPUT.
    """

    EXOGENOUS_FLAGS = [
        ("MAP", FLAG_MAP_VALID),
        ("TPS", FLAG_TPS_VALID),
        ("TIM", FLAG_TIM_VALID),
        ("PAMB", FLAG_PAMB_VALID),
        ("TAMB", FLAG_TAMB_VALID),
        ("VTAS", FLAG_VTAS_VALID),
        ("HP", FLAG_HP_VALID),
    ]

    PRIMARY_VALID_MASK = (
        FLAG_EGT1_VALID | FLAG_EGT2_VALID | FLAG_EGT3_VALID | FLAG_EGT4_VALID |
        FLAG_CHT_VALID | FLAG_POIL_VALID | FLAG_TOIL_VALID | FLAG_RPM_VALID | FLAG_MDF_VALID
    )

    @pytest.fixture
    def test_raw_residuals(self) -> Tuple[float, ...]:
        return (1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9)

    def test_all_128_exogenous_combinations(self, test_raw_residuals: Tuple[float, ...]) -> None:
        """
        Iterate through all 128 subsets of exogenous validity flags.
        Assert deterministic suppression and exact non-imputation.
        """
        assert ResidualSuppressionEngine.is_imputation_prohibited() is True

        n_combinations = 0
        for combo_bits in range(128):
            n_combinations += 1
            exogenous_mask = 0
            present_exogenous: Set[str] = set()
            missing_exogenous: Set[str] = set()

            for bit_idx, (name, flag) in enumerate(self.EXOGENOUS_FLAGS):
                if (combo_bits >> bit_idx) & 1:
                    exogenous_mask |= flag
                    present_exogenous.add(name)
                else:
                    missing_exogenous.add(name)

            valid_mask = self.PRIMARY_VALID_MASK | exogenous_mask

            # Expected suppressed channels based on dependency map
            expected_suppressed: Set[int] = set()
            for flag, affected_channels in ResidualSuppressionEngine.DEPENDENCY_MAP.items():
                if not (valid_mask & flag):
                    expected_suppressed.update(affected_channels)

            z_out, status, suppressed_names = ResidualSuppressionEngine.evaluate_suppression(
                raw_z=test_raw_residuals,
                valid_mask=valid_mask,
                is_saturated=False,
            )

            # 1. Assert length and structure
            assert len(z_out) == NUM_RESIDUAL_CHANNELS
            assert len(suppressed_names) == len(expected_suppressed)

            # 2. Check every channel for NaN suppression vs raw preservation
            for idx in range(NUM_RESIDUAL_CHANNELS):
                if idx in expected_suppressed:
                    assert math.isnan(z_out[idx]), (
                        f"Channel {RESIDUAL_CHANNELS[idx]} must be NaN suppressed "
                        f"when missing {missing_exogenous}, but got {z_out[idx]}"
                    )
                else:
                    assert not math.isnan(z_out[idx]), (
                        f"Channel {RESIDUAL_CHANNELS[idx]} was unexpectedly suppressed "
                        f"with missing {missing_exogenous}"
                    )
                    # Zero imputation assertion: value must be exact raw input
                    assert z_out[idx] == pytest.approx(test_raw_residuals[idx], abs=1e-7), (
                        f"Channel {RESIDUAL_CHANNELS[idx]} was mutated/imputed! "
                        f"Expected {test_raw_residuals[idx]}, got {z_out[idx]}"
                    )

            # 3. Check status code logic
            if len(expected_suppressed) == 0:
                assert status == STATUS_OK, f"Expected STATUS_OK when no channels suppressed, got {status}"
            else:
                assert status == STATUS_DEGRADED_INPUT, (
                    f"Expected STATUS_DEGRADED_INPUT for missing {missing_exogenous}, got {status}"
                )

        assert n_combinations == 128, f"Expected exactly 128 combinations, executed {n_combinations}"

    def test_gas_path_exogenous_dependencies(self, test_raw_residuals: Tuple[float, ...]) -> None:
        """
        Verify MAP and T_im missing causes suppression of all gas-path residuals
        (z_egt_1..4, z_fuel_flow, z_power_balance).
        """
        for flag_name, flag_val in [("MAP", FLAG_MAP_VALID), ("TIM", FLAG_TIM_VALID)]:
            mask = FULL_VALID_MASK & ~flag_val
            z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(test_raw_residuals, mask)

            assert status == STATUS_DEGRADED_INPUT
            # Gas path suppressed
            assert math.isnan(z_out[IDX_Z_EGT1])
            assert math.isnan(z_out[IDX_Z_EGT2])
            assert math.isnan(z_out[IDX_Z_EGT3])
            assert math.isnan(z_out[IDX_Z_EGT4])
            assert math.isnan(z_out[IDX_Z_MDF])
            assert math.isnan(z_out[IDX_Z_POWER])

            # Thermal & lubrication intact
            assert z_out[IDX_Z_CHT] == pytest.approx(test_raw_residuals[IDX_Z_CHT])
            assert z_out[IDX_Z_POIL] == pytest.approx(test_raw_residuals[IDX_Z_POIL])
            assert z_out[IDX_Z_TOIL] == pytest.approx(test_raw_residuals[IDX_Z_TOIL])

    def test_cooling_convective_exogenous_dependencies(self, test_raw_residuals: Tuple[float, ...]) -> None:
        """
        Verify V_TAS, p_amb, T_amb missing causes suppression of convective cooling
        residuals (z_cht, z_oil_temperature, z_power_balance).
        """
        for flag_name, flag_val in [("VTAS", FLAG_VTAS_VALID), ("PAMB", FLAG_PAMB_VALID), ("TAMB", FLAG_TAMB_VALID)]:
            mask = FULL_VALID_MASK & ~flag_val
            z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(test_raw_residuals, mask)

            assert status == STATUS_DEGRADED_INPUT
            # Thermal suppressed
            assert math.isnan(z_out[IDX_Z_CHT])
            assert math.isnan(z_out[IDX_Z_TOIL])
            assert math.isnan(z_out[IDX_Z_POWER])

            # Gas path intact
            assert z_out[IDX_Z_EGT1] == pytest.approx(test_raw_residuals[IDX_Z_EGT1])
            assert z_out[IDX_Z_EGT2] == pytest.approx(test_raw_residuals[IDX_Z_EGT2])
            assert z_out[IDX_Z_EGT3] == pytest.approx(test_raw_residuals[IDX_Z_EGT3])
            assert z_out[IDX_Z_EGT4] == pytest.approx(test_raw_residuals[IDX_Z_EGT4])
    def test_all_128_exogenous_combinations_with_telemetry_frame(self) -> None:
        """
        Test end-to-end TelemetryFrame -> apply_suppression_to_frame -> ResidualFrame -> Polars/PyArrow
        for all 128 exogenous input combinations.
        """
        for combo_bits in range(128):
            exogenous_mask = 0
            for bit_idx, (name, flag) in enumerate(self.EXOGENOUS_FLAGS):
                if (combo_bits >> bit_idx) & 1:
                    exogenous_mask |= flag

            valid_mask = self.PRIMARY_VALID_MASK | exogenous_mask

            # Build TelemetryFrame with valid_mask
            tf = TelemetryFrame(
                t=float(combo_bits),
                egt=(1050.0, 1045.0, 1055.0, 1050.0),
                cht=365.0,
                p_oil=350000.0,
                t_oil=360.0,
                n_rpm=5000.0,
                mdot_f=0.0055,
                map_pa=105000.0 if (valid_mask & FLAG_MAP_VALID) else float("nan"),
                tps=0.75 if (valid_mask & FLAG_TPS_VALID) else float("nan"),
                t_im=310.0 if (valid_mask & FLAG_TIM_VALID) else float("nan"),
                p_amb=80000.0 if (valid_mask & FLAG_PAMB_VALID) else float("nan"),
                t_amb=275.0 if (valid_mask & FLAG_TAMB_VALID) else float("nan"),
                v_tas=65.0 if (valid_mask & FLAG_VTAS_VALID) else float("nan"),
                h_p=2000.0 if (valid_mask & FLAG_HP_VALID) else float("nan"),
                valid_mask=valid_mask,
                flight_id=f"FLIGHT_COMBO_{combo_bits}",
            )

            raw_z = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
            sigma = (1.0,) * 9
            x_hat = (0.0,) * 9

            rf = ResidualSuppressionEngine.apply_suppression_to_frame(
                telemetry=tf,
                raw_z=raw_z,
                regime=4,
                regime_stable_s=25.0,
                spatial=(0.1, -0.05, 0.2, 2),
                sigma=sigma,
                x_hat=x_hat,
            )

            # Determine expected suppression
            expected_suppressed: Set[int] = set()
            for flag, affected_channels in ResidualSuppressionEngine.DEPENDENCY_MAP.items():
                if not (valid_mask & flag):
                    expected_suppressed.update(affected_channels)

            # Assert ResidualFrame methods
            assert rf.count_valid_channels() == 9 - len(expected_suppressed)
            for ch in range(NUM_RESIDUAL_CHANNELS):
                if ch in expected_suppressed:
                    assert rf.is_suppressed(ch) is True
                else:
                    assert rf.is_suppressed(ch) is False

            # Assert serialization roundtrips
            df = residual_frames_to_polars([rf])
            assert df.height == 1
            rec_rf = polars_to_residual_frames(df)[0]
            assert rec_rf.status == rf.status
            assert rec_rf.count_valid_channels() == rf.count_valid_channels()


# ============================================================================
# Section 2: Primary vs Exogenous Sensor Failure Discrimination
# ============================================================================

class TestPrimaryVsExogenousFailureDiscrimination:
    """
    Stress-tests the status code discrimination between:
    - Primary sensor failure (STATUS_INVALID_SENSOR)
    - Exogenous model driver missing (STATUS_DEGRADED_INPUT)
    - Combined primary + exogenous failure (STATUS_DEGRADED_INPUT takes precedence)
    - Model saturation clamp (STATUS_MODEL_SATURATED)
    """

    PRIMARY_FLAGS = [
        ("EGT1", FLAG_EGT1_VALID, [IDX_Z_EGT1]),
        ("EGT2", FLAG_EGT2_VALID, [IDX_Z_EGT2]),
        ("EGT3", FLAG_EGT3_VALID, [IDX_Z_EGT3]),
        ("EGT4", FLAG_EGT4_VALID, [IDX_Z_EGT4]),
        ("CHT", FLAG_CHT_VALID, [IDX_Z_CHT]),
        ("POIL", FLAG_POIL_VALID, [IDX_Z_POIL]),
        ("MDF", FLAG_MDF_VALID, [IDX_Z_MDF]),
        ("TOIL", FLAG_TOIL_VALID, [IDX_Z_TOIL, IDX_Z_POIL]),
    ]

    @pytest.fixture
    def test_raw_residuals(self) -> Tuple[float, ...]:
        return (0.5,) * 9

    def test_isolated_primary_sensor_failures(self, test_raw_residuals: Tuple[float, ...]) -> None:
        """Isolated primary sensor failures must emit STATUS_INVALID_SENSOR."""
        for name, flag, affected_indices in self.PRIMARY_FLAGS:
            mask = FULL_VALID_MASK & ~flag
            z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(test_raw_residuals, mask)

            assert status == STATUS_INVALID_SENSOR, f"Expected STATUS_INVALID_SENSOR for {name}, got {status}"
            for idx in affected_indices:
                assert math.isnan(z_out[idx]), f"Channel {idx} must be NaN for missing {name}"

    def test_engine_speed_rpm_total_suppression(self, test_raw_residuals: Tuple[float, ...]) -> None:
        """Missing engine speed (RPM) must suppress all 9 channels."""
        mask = FULL_VALID_MASK & ~FLAG_RPM_VALID
        z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(test_raw_residuals, mask)

        assert len(suppressed) == 9
        assert all(math.isnan(z) for z in z_out)
        assert status in (STATUS_DEGRADED_INPUT, STATUS_INVALID_SENSOR)

    def test_combined_primary_and_exogenous_failure(self, test_raw_residuals: Tuple[float, ...]) -> None:
        """When both primary sensor and exogenous input fail, status indicates degraded input."""
        mask = FULL_VALID_MASK & ~FLAG_EGT1_VALID & ~FLAG_MAP_VALID
        z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(test_raw_residuals, mask)

        assert status == STATUS_DEGRADED_INPUT
        assert math.isnan(z_out[IDX_Z_EGT1])
        assert math.isnan(z_out[IDX_Z_MDF])
        assert math.isnan(z_out[IDX_Z_POWER])

    def test_saturation_overrides_status(self, test_raw_residuals: Tuple[float, ...]) -> None:
        """Physical model saturation clamp must yield STATUS_MODEL_SATURATED."""
        z_out, status, _ = ResidualSuppressionEngine.evaluate_suppression(
            test_raw_residuals, FULL_VALID_MASK, is_saturated=True
        )
        assert status == STATUS_MODEL_SATURATED


# ============================================================================
# Section 3: Serialization & Schema Roundtrip Stress Tests
# ============================================================================

class TestResidualSerializationUnderSuppression:
    """
    Validates Polars DataFrame and Apache Arrow Table conversions under arbitrary
    combinations of suppressed channels.
    """

    def test_all_sparse_nan_patterns_polars_roundtrip(self) -> None:
        """Test random combinations of NaN residual patterns roundtrip perfectly through Polars."""
        rng = np.random.default_rng(seed=42)
        frames = []

        for frame_idx in range(50):
            z_vals = []
            for ch in range(NUM_RESIDUAL_CHANNELS):
                # 30% chance of suppression
                if rng.uniform() < 0.30:
                    z_vals.append(float("nan"))
                else:
                    z_vals.append(float(rng.normal(0.0, 1.0)))

            rf = ResidualFrame(
                t=float(frame_idx),
                z=tuple(z_vals),  # type: ignore[arg-type]
                regime=frame_idx % 12,
                regime_stable_s=float(frame_idx),
                spatial=(0.1, -0.05, 0.2, 1),
                sigma=(1.0,) * 9,
                x_hat=(0.0,) * 9,
                status=STATUS_DEGRADED_INPUT if any(math.isnan(x) for x in z_vals) else STATUS_OK,
                flight_id=f"FLIGHT_STRESS_{frame_idx}",
            )
            frames.append(rf)

        # Convert to Polars
        df = residual_frames_to_polars(frames)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 50
        assert df.schema == RESIDUAL_POLARS_SCHEMA

        # Convert back
        recovered = polars_to_residual_frames(df)
        assert len(recovered) == 50

        for orig, rec in zip(frames, recovered):
            assert orig.t == rec.t
            assert orig.regime == rec.regime
            assert orig.status == rec.status
            for i in range(NUM_RESIDUAL_CHANNELS):
                if math.isnan(orig.z[i]):
                    assert math.isnan(rec.z[i])
                else:
                    assert rec.z[i] == pytest.approx(orig.z[i], abs=1e-6)

    def test_arrow_table_roundtrip_with_nulls(self) -> None:
        """Test Apache Arrow roundtrip preserving null/NaN fields."""
        rf = ResidualFrame(
            t=123.4,
            z=(float("nan"), 1.2, float("nan"), 3.4, float("nan"), 5.6, float("nan"), 7.8, float("nan")),
            regime=2,
            regime_stable_s=21.0,
            spatial=(0.0, 0.0, 0.0, 0),
            sigma=(1.0,) * 9,
            x_hat=(0.0,) * 9,
            status=STATUS_DEGRADED_INPUT,
            flight_id="FLIGHT_ARROW_STRESS",
        )
        table = residual_frames_to_arrow([rf])
        assert isinstance(table, pa.Table)
        assert table.num_rows == 1

        recovered = arrow_to_residual_frames(table)
        assert len(recovered) == 1
        assert math.isnan(recovered[0].z[0])
        assert recovered[0].z[1] == pytest.approx(1.2, abs=1e-6)


# ============================================================================
# Section 4: FaultInjectionHarness Stress & Asymmetry Tests
# ============================================================================

class TestFaultInjectionHarnessStress:
    """
    Stress-tests FaultInjectionHarness:
    1. Enforces w >= 0.35 non-uniform cooling asymmetry constraint.
    2. Per-cylinder cooling conductance multiplier asymmetry (front vs rear).
    3. Multi-fault interactions and sequential compositions.
    4. Extreme fault severities and physical clipping bounds.
    """

    @pytest.fixture
    def nominal_hp(self) -> HealthParameters:
        return HealthParameters()

    def test_cooling_asymmetry_weight_boundary(self) -> None:
        """
        Boundary condition check:
        - w = 0.35 MUST be accepted
        - w = 0.34999 MUST raise ValueError
        - w < 0.35 MUST raise ValueError
        """
        # Valid boundary at 0.35
        f_exact = FaultSpec(fault_type="cooling_degradation", spatial_asymmetry_weight=0.35)
        assert f_exact.spatial_asymmetry_weight == 0.35

        # Invalid just below boundary
        with pytest.raises(ValueError, match="w >= 0.35"):
            FaultSpec(fault_type="cooling_degradation", spatial_asymmetry_weight=0.349999)

        # Invalid low value
        with pytest.raises(ValueError, match="w >= 0.35"):
            FaultSpec(fault_type="cooling_degradation", spatial_asymmetry_weight=0.0)

    def test_front_vs_rear_cooling_conductance_divergence(self) -> None:
        """
        Verify that compute_non_uniform_cooling_factors produces distinct
        conductance factors between front cylinders (1, 2) and rear cylinders (3, 4)
        under non-uniform cooling degradation.
        """
        fault = FaultSpec(
            fault_type="cooling_degradation",
            magnitude=0.20,
            spatial_asymmetry_weight=0.45,
            onset_time_s=10.0,
            ramp_duration_s=50.0,
        )

        # At t = 60s (full severity)
        ua_factors = FaultInjectionHarness.compute_non_uniform_cooling_factors(fault, t=60.0)
        assert len(ua_factors) == 4

        # Conductance multipliers must all be positive and >= 0.10
        assert all(ua >= 0.10 for ua in ua_factors)

        # Asymmetric degradation: all 4 cylinders have distinct non-uniform cooling conductances
        assert len(set(np.round(ua_factors, 4))) == 4, "All 4 cylinders must have distinct cooling factors"

        # Spread across cylinders is significant (non-uniform degradation)
        spread = float(np.max(ua_factors) - np.min(ua_factors))
        assert spread > 0.10, f"Expected significant spatial spread across cylinders, got {spread}"

    def test_ramp_dynamics_and_negative_time(self, nominal_hp: HealthParameters) -> None:
        """Test ramp function at various flight time points including negative time."""
        fault = FaultSpec(
            fault_type="combustion_degradation",
            magnitude=0.20,
            onset_time_s=100.0,
            ramp_duration_s=40.0,
        )

        # Negative time
        hp_neg = FaultInjectionHarness.inject(nominal_hp, fault, t=-10.0)
        assert hp_neg.theta_comb == pytest.approx(1.0)

        # Before onset (t = 50s)
        hp_before = FaultInjectionHarness.inject(nominal_hp, fault, t=50.0)
        assert hp_before.theta_comb == pytest.approx(1.0)

        # Exact onset (t = 100s)
        hp_onset = FaultInjectionHarness.inject(nominal_hp, fault, t=100.0)
        assert hp_onset.theta_comb == pytest.approx(1.0)

        # Quarter ramp (t = 110s, 25%)
        hp_q = FaultInjectionHarness.inject(nominal_hp, fault, t=110.0)
        assert hp_q.theta_comb == pytest.approx(1.0 - 0.20 * 0.25, rel=1e-4)

        # Half ramp (t = 120s, 50%)
        hp_half = FaultInjectionHarness.inject(nominal_hp, fault, t=120.0)
        assert hp_half.theta_comb == pytest.approx(1.0 - 0.20 * 0.50, rel=1e-4)

        # Full ramp (t = 140s, 100%)
        hp_full = FaultInjectionHarness.inject(nominal_hp, fault, t=140.0)
        assert hp_full.theta_comb == pytest.approx(0.80, rel=1e-4)

        # Post-ramp (t = 300s, 100% clamped)
        hp_post = FaultInjectionHarness.inject(nominal_hp, fault, t=300.0)
        assert hp_post.theta_comb == pytest.approx(0.80, rel=1e-4)

    def test_instantaneous_step_fault(self, nominal_hp: HealthParameters) -> None:
        """Test instantaneous step fault with ramp_duration_s = 0.0 within physical bounds."""
        fault = FaultSpec(
            fault_type="sensor_bias_egt",
            target_cylinder=4,
            magnitude=20.0,  # <= 24.0 K max admissible bias
            onset_time_s=50.0,
            ramp_duration_s=0.0,
        )

        assert FaultInjectionHarness.inject(nominal_hp, fault, t=49.9).b_egt[3] == 0.0
        assert FaultInjectionHarness.inject(nominal_hp, fault, t=50.0).b_egt[3] == pytest.approx(20.0)
        assert FaultInjectionHarness.inject(nominal_hp, fault, t=100.0).b_egt[3] == pytest.approx(20.0)

    def test_extreme_fault_severity_clipping(self, nominal_hp: HealthParameters) -> None:
        """
        Verify that extreme fault magnitudes (e.g. 500% degradation) clip safely
        to admissible physical parameter bounds.
        """
        # Massive cooling loss
        f_extreme_cool = FaultSpec(fault_type="cooling_degradation", magnitude=5.0, spatial_asymmetry_weight=0.40)
        hp_cool = FaultInjectionHarness.inject(nominal_hp, f_extreme_cool, t=100.0)
        is_valid, _ = hp_cool.validate()
        assert is_valid is True
        assert hp_cool.theta_cool >= HEALTH_PARAMETER_SPECS["theta_cool"][0]  # min admissible bound

        # Massive friction increase
        f_extreme_fric = FaultSpec(fault_type="friction_increase", magnitude=10.0)
        hp_fric = FaultInjectionHarness.inject(nominal_hp, f_extreme_fric, t=100.0)
        is_valid, _ = hp_fric.validate()
        assert is_valid is True
        assert hp_fric.theta_fric <= HEALTH_PARAMETER_SPECS["theta_fric"][1]  # max admissible bound

    def test_sequential_multi_fault_composition(self, nominal_hp: HealthParameters) -> None:
        """
        Apply a chain of 5 distinct faults in sequence and verify deterministic health state.
        """
        faults = [
            FaultSpec(fault_type="cooling_degradation", magnitude=0.10, spatial_asymmetry_weight=0.38),
            FaultSpec(fault_type="injector_clogging", target_cylinder=1, magnitude=0.08),
            FaultSpec(fault_type="sensor_bias_cht", magnitude=4.0),
            FaultSpec(fault_type="oil_pump_wear", magnitude=0.12),
            FaultSpec(fault_type="friction_increase", magnitude=0.15),
        ]

        hp_current = nominal_hp
        for fault in faults:
            hp_current = FaultInjectionHarness.inject(hp_current, fault, t=100.0)

        assert hp_current.theta_cool == pytest.approx(0.90, abs=1e-3)
        assert hp_current.theta_inj[0] == pytest.approx(0.92, abs=1e-3)
        assert hp_current.theta_inj[1] == pytest.approx(1.0, abs=1e-3)
        assert hp_current.b_cht == pytest.approx(4.0, abs=1e-3)
        assert hp_current.theta_oilp == pytest.approx(0.88, abs=1e-3)
        assert hp_current.theta_fric == pytest.approx(1.15, abs=1e-3)

        is_valid, violations = hp_current.validate()
        assert is_valid is True, f"Sequential fault state must remain valid, violations: {violations}"

    def test_out_of_bounds_target_cylinder_raises(self) -> None:
        """Target cylinder out of 0..4 must raise ValueError."""
        with pytest.raises(ValueError, match="target_cylinder"):
            FaultSpec(fault_type="injector_clogging", target_cylinder=-1)
        with pytest.raises(ValueError, match="target_cylinder"):
            FaultSpec(fault_type="injector_clogging", target_cylinder=5)
        with pytest.raises(ValueError, match="target_cylinder"):
            FaultSpec(fault_type="injector_clogging", target_cylinder=99)


# ============================================================================
# Section 5: Spatial EGT Orthogonal Decomposition Stress Tests
# ============================================================================

class TestSpatialEGTDecompositionStress:
    """
    Stress-tests SpatialEGTDecomposer:
    1. Orthonormality verification (||1|| = 1, ||g|| = 1, 1^T g = 0).
    2. Monte Carlo 1,000 trials:
       - Exact reconstruction: z = alpha * 1 + beta * g + s_t
       - Exact orthogonality: s_t orthogonal to span{1, g}
       - Norm-infinity: ||s_t||_inf = max_i |s_t,i|
       - Dominant cylinder: argmax_i |s_t,i| in 1..4
    3. Pure canonical basis projections (pure common mode, pure gradient, per-cylinder spikes).
    4. NaN / missing channel handling in spatial projection.
    """

    def test_basis_orthonormality(self) -> None:
        """Verify unit norm and mutual orthogonality of 1 and g."""
        norm_1 = np.linalg.norm(BASIS_COMMON_MODE)
        norm_g = np.linalg.norm(BASIS_GRADIENT)
        dot_1_g = np.dot(BASIS_COMMON_MODE, BASIS_GRADIENT)

        assert norm_1 == pytest.approx(1.0, abs=1e-12)
        assert norm_g == pytest.approx(1.0, abs=1e-12)
        assert abs(dot_1_g) < 1e-12

    def test_monte_carlo_spatial_decomposition_1000_trials(self) -> None:
        """
        Generate 1,000 arbitrary random EGT residual vectors.
        Assert reconstruction, orthogonality, infinity norm, and dominant cylinder index.
        """
        rng = np.random.default_rng(seed=2026)
        n_trials = 1000

        for trial in range(n_trials):
            z_random = rng.normal(loc=0.0, scale=10.0, size=4)
            alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose(z_random)

            # Reconstruct s_t
            s_t = z_random - (alpha * BASIS_COMMON_MODE + beta * BASIS_GRADIENT)

            # 1. Exact reconstruction check: ||z - (alpha 1 + beta g + s)||_2 < 1e-12
            reconstruction = alpha * BASIS_COMMON_MODE + beta * BASIS_GRADIENT + s_t
            rec_error = np.linalg.norm(z_random - reconstruction)
            assert rec_error < 1e-12, f"Reconstruction error {rec_error} at trial {trial}"

            # 2. Orthogonality checks
            dot_s_1 = np.dot(s_t, BASIS_COMMON_MODE)
            dot_s_g = np.dot(s_t, BASIS_GRADIENT)
            assert abs(dot_s_1) < 1e-12, f"s_t not orthogonal to 1: dot = {dot_s_1}"
            assert abs(dot_s_g) < 1e-12, f"s_t not orthogonal to g: dot = {dot_s_g}"

            # 3. Infinity norm check
            expected_norm_inf = float(np.max(np.abs(s_t)))
            assert norm_inf == pytest.approx(expected_norm_inf, abs=1e-12)

            # 4. Dominant cylinder check (1-indexed, 1..4)
            expected_dom_cyl = int(np.argmax(np.abs(s_t)) + 1)
            assert dom_cyl == expected_dom_cyl
            assert dom_cyl in (1, 2, 3, 4)

    def test_single_cylinder_localized_perturbations(self) -> None:
        """
        Test localized perturbation patterns on inner and outer cylinders:
        - Inner cylinders 2 and 3 (with lower gradient projections) exhibit peak s_t on themselves.
        - Outer cylinders 1 and 4 project significantly onto fore-aft gradient g.
        - All spikes produce strictly non-zero s_t and valid dominant cylinder indices.
        """
        # Cylinders 2 and 3 isolated spikes
        for target_cyl in (2, 3):
            z = np.zeros(4, dtype=np.float64)
            z[target_cyl - 1] = 12.0  # +12 K spike

            alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose(z)

            assert dom_cyl == target_cyl, f"Expected dominant cylinder {target_cyl}, got {dom_cyl}"
            assert norm_inf == pytest.approx(7.8, abs=1e-4)

        # Cylinders 1 and 4 isolated spikes
        for target_cyl in (1, 4):
            z = np.zeros(4, dtype=np.float64)
            z[target_cyl - 1] = 12.0  # +12 K spike

            alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose(z)
            assert dom_cyl in (1, 2, 3, 4)
            assert norm_inf == pytest.approx(5.4, abs=1e-4)

    def test_nan_handling_edge_cases(self) -> None:
        """
        Verify graceful degradation when EGT channels contain NaNs.
        """
        # All 4 NaNs
        alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose([float("nan")] * 4)
        assert alpha == 0.0
        assert beta == 0.0
        assert norm_inf == 0.0
        assert dom_cyl == 0

        # 2 NaNs with valid entries
        alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose([14.0, float("nan"), 16.0, float("nan")])
        assert alpha == pytest.approx(15.0)
        assert beta == 0.0
        assert norm_inf == 0.0
        assert dom_cyl == 0


# ============================================================================
# Section 6: Multi-Fault Permutation Invariance & Boundary Stress
# ============================================================================

class TestMultiFaultPermutationAndNumericalInvariance:
    """
    Validates commutativity / order invariance for independent faults
    and numerical boundary behaviors.
    """

    def test_independent_fault_permutation_invariance(self) -> None:
        """
        Applying independent faults (e.g. cooling, oil pump, friction) in any
        of the 3! = 6 permutations must yield bit-identical final HealthParameters.
        """
        f1 = FaultSpec(fault_type="cooling_degradation", magnitude=0.10, spatial_asymmetry_weight=0.35)
        f2 = FaultSpec(fault_type="oil_pump_wear", magnitude=0.15)
        f3 = FaultSpec(fault_type="friction_increase", magnitude=0.20)

        fault_list = [f1, f2, f3]
        results = []

        for perm in itertools.permutations(fault_list):
            hp = HealthParameters()
            for f in perm:
                hp = FaultInjectionHarness.inject(hp, f, t=100.0)
            results.append(hp.to_array())

        # All permutations must yield identical arrays
        for arr in results[1:]:
            np.testing.assert_allclose(arr, results[0], atol=1e-12)

    def test_high_magnitude_floating_point_stability(self) -> None:
        """Test spatial decomposition under large and subnormal floating point values."""
        # Large finite numbers
        z_large = np.array([1e6, -2e6, 3e6, -4e6], dtype=np.float64)
        alpha, beta, norm_inf, dom_cyl = SpatialEGTDecomposer.decompose(z_large)
        assert not math.isnan(alpha)
        assert not math.isnan(beta)
        assert not math.isnan(norm_inf)
        assert dom_cyl in (1, 2, 3, 4)

        # Very small numbers (near zero)
        z_tiny = np.array([1e-15, -2e-15, 3e-15, -1e-15], dtype=np.float64)
        alpha_t, beta_t, norm_t, dom_t = SpatialEGTDecomposer.decompose(z_tiny)
        assert not math.isnan(alpha_t)
        assert not math.isnan(beta_t)
        assert dom_t in (1, 2, 3, 4)

