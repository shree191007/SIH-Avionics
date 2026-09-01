"""
Unit tests for replan_to_learn.contracts.residuals
Verifies the ResidualFrame contract, dual-format schemas, and the strict
ResidualSuppressionEngine programmatic zero-imputation policy.
"""

import math
import pytest
import numpy as np
import polars as pl
import pyarrow as pa

from replan_to_learn.contracts.telemetry import (
    FLAG_CHT_VALID,
    FLAG_EGT1_VALID,
    FLAG_EGT2_VALID,
    FLAG_EGT3_VALID,
    FLAG_EGT4_VALID,
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


class TestResidualFrameContract:
    """Test ResidualFrame frozen dataclass properties."""

    def test_residual_frame_creation(self, sample_residual_frame: ResidualFrame) -> None:
        rf = sample_residual_frame
        assert rf.t == 100.0
        assert len(rf.z) == 9
        assert rf.regime == 4
        assert rf.regime_stable_s == 25.0
        assert rf.status == STATUS_OK
        assert rf.count_valid_channels() == 9
        assert isinstance(rf.z_array, np.ndarray)

    def test_residual_frame_immutability(self, sample_residual_frame: ResidualFrame) -> None:
        with pytest.raises(Exception):
            sample_residual_frame.regime = 0  # type: ignore[misc]


class TestResidualSuppressionEngine:
    """Test strict zero-imputation policy and dependency graph suppression."""

    def test_zero_imputation_policy_attestation(self) -> None:
        assert ResidualSuppressionEngine.is_imputation_prohibited() is True

    def test_nominal_full_valid_mask(self) -> None:
        raw_z = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(raw_z, FULL_VALID_MASK)
        assert status == STATUS_OK
        assert len(suppressed) == 0
        assert all(not math.isnan(val) for val in z_out)

    def test_missing_map_suppression(self) -> None:
        """Missing MAP suppresses gas-path residuals (EGT 1..4, fuel flow, power balance)."""
        raw_z = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        # Clear MAP flag
        mask_without_map = FULL_VALID_MASK & ~FLAG_MAP_VALID
        z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(raw_z, mask_without_map)

        assert status == STATUS_DEGRADED_INPUT
        # Gas path suppressed
        assert math.isnan(z_out[IDX_Z_EGT1])
        assert math.isnan(z_out[IDX_Z_EGT2])
        assert math.isnan(z_out[IDX_Z_EGT3])
        assert math.isnan(z_out[IDX_Z_EGT4])
        assert math.isnan(z_out[IDX_Z_MDF])
        assert math.isnan(z_out[IDX_Z_POWER])
        # Thermal & oil residuals remain unaffected
        assert not math.isnan(z_out[IDX_Z_CHT])
        assert not math.isnan(z_out[IDX_Z_POIL])
        assert not math.isnan(z_out[IDX_Z_TOIL])

    def test_missing_ambient_temperature_suppression(self) -> None:
        """Missing T_amb suppresses thermal residuals (CHT, T_oil, power balance)."""
        raw_z = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        mask_without_tamb = FULL_VALID_MASK & ~FLAG_TAMB_VALID
        z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(raw_z, mask_without_tamb)

        assert status == STATUS_DEGRADED_INPUT
        assert math.isnan(z_out[IDX_Z_CHT])
        assert math.isnan(z_out[IDX_Z_TOIL])
        assert math.isnan(z_out[IDX_Z_POWER])
        # Gas path unaffected
        assert not math.isnan(z_out[IDX_Z_EGT1])
        assert not math.isnan(z_out[IDX_Z_MDF])

    def test_missing_engine_speed_suppresses_all_residuals(self) -> None:
        """Missing RPM invalidates all 9 residual channels."""
        raw_z = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        mask_without_rpm = FULL_VALID_MASK & ~FLAG_RPM_VALID
        z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(raw_z, mask_without_rpm)

        assert status in (STATUS_DEGRADED_INPUT, STATUS_INVALID_SENSOR)
        assert len(suppressed) == 9
        assert all(math.isnan(val) for val in z_out)

    def test_single_primary_sensor_failure(self) -> None:
        """Single failed EGT probe suppresses only that probe."""
        raw_z = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        mask_egt2_failed = FULL_VALID_MASK & ~FLAG_EGT2_VALID
        z_out, status, suppressed = ResidualSuppressionEngine.evaluate_suppression(raw_z, mask_egt2_failed)

        assert status == STATUS_INVALID_SENSOR
        assert math.isnan(z_out[IDX_Z_EGT2])
        assert not math.isnan(z_out[IDX_Z_EGT1])
        assert not math.isnan(z_out[IDX_Z_EGT3])
        assert not math.isnan(z_out[IDX_Z_CHT])

    def test_saturation_clamp_status(self) -> None:
        raw_z = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        z_out, status, _ = ResidualSuppressionEngine.evaluate_suppression(raw_z, FULL_VALID_MASK, is_saturated=True)
        assert status == STATUS_MODEL_SATURATED


class TestResidualSchemas:
    """Test Polars and Arrow serialization of ResidualFrame with NaNs/nulls."""

    def test_polars_schema_roundtrip_with_suppressed_nans(self) -> None:
        # Create a frame with suppressed channels
        z_with_nans = (float("nan"), 0.15, float("nan"), 0.20, 0.05, float("nan"), 0.12, 0.08, float("nan"))
        rf = ResidualFrame(
            t=50.0,
            z=z_with_nans,
            regime=3,
            regime_stable_s=22.0,
            spatial=(0.1, -0.05, 0.2, 2),
            sigma=(1.0,) * 9,
            x_hat=(0.0,) * 9,
            status=STATUS_DEGRADED_INPUT,
            flight_id="FLIGHT_SUPPRESSED",
        )
        df = residual_frames_to_polars([rf])
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1
        assert df.schema == RESIDUAL_POLARS_SCHEMA

        # Check null count in Polars
        assert df["z_egt_1"].is_null()[0] is True
        assert df["z_egt_2"].is_null()[0] is False

        # Convert back
        recovered = polars_to_residual_frames(df)
        assert len(recovered) == 1
        assert math.isnan(recovered[0].z[0])
        assert recovered[0].z[1] == pytest.approx(0.15, rel=1e-5)
        assert recovered[0].status == STATUS_DEGRADED_INPUT

    def test_arrow_schema_roundtrip(self, sample_residual_frame: ResidualFrame) -> None:
        table = residual_frames_to_arrow([sample_residual_frame])
        assert isinstance(table, pa.Table)
        assert table.num_rows == 1

        recovered = arrow_to_residual_frames(table)
        assert len(recovered) == 1
        assert recovered[0].t == sample_residual_frame.t
        assert recovered[0].regime == sample_residual_frame.regime

    def test_apply_suppression_to_frame(self, sample_telemetry_frame: TelemetryFrame) -> None:
        raw_z = (0.1,) * 9
        sigma = (1.0,) * 9
        x_hat = (0.0,) * 9
        rf = ResidualSuppressionEngine.apply_suppression_to_frame(
            telemetry=sample_telemetry_frame,
            raw_z=raw_z,
            regime=4,
            regime_stable_s=25.0,
            spatial=(0.0, 0.0, 0.0, 1),
            sigma=sigma,
            x_hat=x_hat,
        )
        assert rf.status == STATUS_OK
        assert rf.count_valid_channels() == 9

    def test_validate_residual_dataframe(self, sample_residual_frame: ResidualFrame) -> None:
        df = residual_frames_to_polars([sample_residual_frame])
        validated_df = validate_residual_dataframe(df)
        assert validated_df.height == 1

    def test_validate_residual_dataframe_missing_column_raises(self) -> None:
        df_bad = pl.DataFrame({"timestamp": [1.0], "z_egt_1": [0.1]})
        with pytest.raises(ValueError, match="Missing required residual column"):
            validate_residual_dataframe(df_bad)

    def test_residual_frame_shape_validations(self) -> None:
        with pytest.raises(ValueError, match="Residual vector z must contain 9 elements"):
            ResidualFrame(
                t=0.0,
                z=(0.1, 0.2),  # Invalid len
                regime=0,
                regime_stable_s=0.0,
                spatial=(0.0, 0.0, 0.0, 0),
                sigma=(1.0,) * 9,
                x_hat=(0.0,) * 9,
                status=0,
            )
