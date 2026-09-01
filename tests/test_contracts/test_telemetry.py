"""
Unit tests for replan_to_learn.contracts.telemetry
Verifies primary channels, exogenous drivers, unit conversions, validity bitmasks,
frozen dataclass contracts, and Polars/PyArrow schema compliance.
"""

import math
import pytest
import numpy as np
import polars as pl
import pyarrow as pa

from replan_to_learn.contracts.telemetry import (
    ALL_SENSOR_SPECS,
    EXOGENOUS_SENSOR_SPECS,
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
    FrameStatus,
    PRIMARY_SENSOR_SPECS,
    STATUS_DEGRADED_INPUT,
    STATUS_INVALID_SENSOR,
    STATUS_MODEL_SATURATED,
    STATUS_OK,
    SensorSpec,
    TELEMETRY_ARROW_SCHEMA,
    TELEMETRY_POLARS_SCHEMA,
    TelemetryFrame,
    TelemetryValidityFlag,
    arrow_to_telemetry_frames,
    bar_to_pa,
    celsius_to_kelvin,
    feet_to_meters,
    hpa_to_pa,
    inhg_to_pa,
    kelvin_to_celsius,
    kg_s_to_liters_per_hour,
    knots_to_m_s,
    kpa_to_pa,
    liters_per_hour_to_kg_s,
    m_s_to_knots,
    meters_to_feet,
    pa_to_bar,
    pa_to_hpa,
    pa_to_inhg,
    pa_to_kpa,
    polars_to_telemetry_frames,
    rad_s_to_rpm,
    rpm_to_rad_s,
    telemetry_frames_to_arrow,
    telemetry_frames_to_polars,
    validate_telemetry_dataframe,
)


class TestUnitConversions:
    """Test mathematical and physical unit conversion utilities."""

    def test_temperature_conversions(self) -> None:
        assert celsius_to_kelvin(0.0) == pytest.approx(273.15, rel=1e-6)
        assert celsius_to_kelvin(100.0) == pytest.approx(373.15, rel=1e-6)
        assert celsius_to_kelvin(-40.0) == pytest.approx(233.15, rel=1e-6)

        assert kelvin_to_celsius(273.15) == pytest.approx(0.0, abs=1e-6)
        assert kelvin_to_celsius(373.15) == pytest.approx(100.0, abs=1e-6)
        # Roundtrip
        t_c = 85.5
        assert kelvin_to_celsius(celsius_to_kelvin(t_c)) == pytest.approx(t_c, abs=1e-6)

    def test_pressure_conversions(self) -> None:
        # Bar <-> Pa
        assert bar_to_pa(1.0) == pytest.approx(100000.0, rel=1e-6)
        assert pa_to_bar(500000.0) == pytest.approx(5.0, rel=1e-6)
        assert pa_to_bar(bar_to_pa(3.45)) == pytest.approx(3.45, rel=1e-6)

        # kPa <-> Pa
        assert kpa_to_pa(105.0) == pytest.approx(105000.0, rel=1e-6)
        assert pa_to_kpa(80000.0) == pytest.approx(80.0, rel=1e-6)

        # inHg <-> Pa
        assert inhg_to_pa(29.92126) == pytest.approx(101325.0, rel=1e-4)
        assert pa_to_inhg(101325.0) == pytest.approx(29.92126, rel=1e-4)

        # hPa <-> Pa
        assert hpa_to_pa(1013.25) == pytest.approx(101325.0, rel=1e-6)
        assert pa_to_hpa(100000.0) == pytest.approx(1000.0, rel=1e-6)

    def test_speed_and_flow_conversions(self) -> None:
        # RPM <-> rad/s
        assert rpm_to_rad_s(60.0) == pytest.approx(2.0 * math.pi, rel=1e-6)
        assert rad_s_to_rpm(2.0 * math.pi) == pytest.approx(60.0, rel=1e-6)
        assert rad_s_to_rpm(rpm_to_rad_s(5000.0)) == pytest.approx(5000.0, rel=1e-6)

        # Knots <-> m/s
        assert knots_to_m_s(100.0) == pytest.approx(51.44444, rel=1e-4)
        assert m_s_to_knots(51.44444) == pytest.approx(100.0, rel=1e-4)

        # Feet <-> Meters
        assert feet_to_meters(1000.0) == pytest.approx(304.8, rel=1e-6)
        assert meters_to_feet(304.8) == pytest.approx(1000.0, rel=1e-6)

        # L/h <-> kg/s (with 0.720 kg/L density)
        assert liters_per_hour_to_kg_s(36.0, density_kg_l=0.720) == pytest.approx(0.0072, rel=1e-6)
        assert kg_s_to_liters_per_hour(0.0072, density_kg_l=0.720) == pytest.approx(36.0, rel=1e-6)


class TestSensorSpecsAndBounds:
    """Test physical validity limits, noise, and disconnect triggers."""

    def test_primary_sensor_specs_count(self) -> None:
        # 6 primary quantities across 9 scalar channels (4 EGT + CHT + p_oil + T_oil + N + mdot_f)
        assert len(PRIMARY_SENSOR_SPECS) == 9
        expected_keys = {"egt_1", "egt_2", "egt_3", "egt_4", "cht", "oil_pressure", "oil_temperature", "engine_speed_rpm", "fuel_flow_kg_s"}
        assert set(PRIMARY_SENSOR_SPECS.keys()) == expected_keys

    def test_exogenous_sensor_specs_count(self) -> None:
        # 7 exogenous channels (MAP, TPS, T_im, p_amb, T_amb, V_TAS, h_p)
        assert len(EXOGENOUS_SENSOR_SPECS) == 7
        expected_keys = {"map_pa", "throttle_position", "intake_temp_k", "ambient_pressure_pa", "ambient_temp_k", "true_airspeed_ms", "pressure_altitude_m"}
        assert set(EXOGENOUS_SENSOR_SPECS.keys()) == expected_keys

    def test_sensor_bounds_checking(self) -> None:
        egt_spec = PRIMARY_SENSOR_SPECS["egt_1"]
        # Nominal
        assert egt_spec.is_in_admissible_range(1000.0) is True
        assert egt_spec.is_disconnected_or_failed(1000.0) is False
        # Below admissible (cold start or warning) but above disconnect
        assert egt_spec.is_in_admissible_range(300.0) is False
        assert egt_spec.is_disconnected_or_failed(300.0) is False
        # Disconnected (e.g. 0 K or open circuit > 1423 K)
        assert egt_spec.is_disconnected_or_failed(200.0) is True
        assert egt_spec.is_disconnected_or_failed(1500.0) is True
        assert egt_spec.is_disconnected_or_failed(float("nan")) is True


class TestTelemetryFrameContract:
    """Test TelemetryFrame frozen dataclass contract and validation."""

    def test_telemetry_frame_creation(self, sample_telemetry_frame: TelemetryFrame) -> None:
        f = sample_telemetry_frame
        assert f.t == 100.0
        assert len(f.egt) == 4
        assert f.egt_1 == 1050.0
        assert f.egt_4 == 1050.0
        assert f.cht == 365.0
        assert f.p_oil == 350000.0
        assert f.n_rpm == 5000.0
        assert f.mdot_f == 0.0055
        assert f.are_primary_sensors_valid() is True
        assert f.are_exogenous_inputs_valid() is True

    def test_telemetry_frame_frozen_immutability(self, sample_telemetry_frame: TelemetryFrame) -> None:
        with pytest.raises(Exception):
            sample_telemetry_frame.t = 200.0  # type: ignore[misc]

    def test_compute_validity_mask_nominal(self, sample_telemetry_frame: TelemetryFrame) -> None:
        computed_mask = sample_telemetry_frame.compute_validity_mask()
        assert (computed_mask & FULL_VALID_MASK) == FULL_VALID_MASK

    def test_compute_validity_mask_with_sensor_disconnect(self) -> None:
        # Create frame with disconnected CHT (e.g. 100 K) and missing MAP (NaN)
        f = TelemetryFrame(
            t=0.0,
            egt=(1050.0, 1050.0, 1050.0, 1050.0),
            cht=100.0,             # Disconnected
            p_oil=350000.0,
            t_oil=360.0,
            n_rpm=5000.0,
            mdot_f=0.0055,
            map_pa=float("nan"),   # Missing
            tps=0.75,
            t_im=310.0,
            p_amb=80000.0,
            t_amb=275.0,
            v_tas=65.0,
            h_p=2000.0,
        )
        mask = f.compute_validity_mask()
        # CHT flag should NOT be set
        assert not (mask & FLAG_CHT_VALID)
        # MAP flag should NOT be set
        assert not (mask & FLAG_MAP_VALID)
        # EGT and RPM should be set
        assert bool(mask & FLAG_EGT1_VALID)
        assert bool(mask & FLAG_RPM_VALID)


class TestMultiFormatSchemas:
    """Test Polars pl.Schema and PyArrow pa.schema integration."""

    def test_polars_roundtrip(self, sample_telemetry_sequence: list) -> None:
        df = telemetry_frames_to_polars(sample_telemetry_sequence)
        assert isinstance(df, pl.DataFrame)
        assert df.height == len(sample_telemetry_sequence)
        assert df.schema == TELEMETRY_POLARS_SCHEMA

        # Convert back
        recovered_frames = polars_to_telemetry_frames(df)
        assert len(recovered_frames) == len(sample_telemetry_sequence)
        assert recovered_frames[0].t == sample_telemetry_sequence[0].t
        assert recovered_frames[0].egt == sample_telemetry_sequence[0].egt

    def test_arrow_roundtrip(self, sample_telemetry_sequence: list) -> None:
        table = telemetry_frames_to_arrow(sample_telemetry_sequence)
        assert isinstance(table, pa.Table)
        assert table.num_rows == len(sample_telemetry_sequence)
        assert table.schema.names == TELEMETRY_ARROW_SCHEMA.names

        # Convert back
        recovered_frames = arrow_to_telemetry_frames(table)
        assert len(recovered_frames) == len(sample_telemetry_sequence)
        assert recovered_frames[0].n_rpm == sample_telemetry_sequence[0].n_rpm

    def test_validate_telemetry_dataframe(self, sample_telemetry_frame: TelemetryFrame) -> None:
        df = telemetry_frames_to_polars([sample_telemetry_frame])
        validated_df = validate_telemetry_dataframe(df)
        assert validated_df.height == 1
        assert validated_df["validity_bitmask"][0] == FULL_VALID_MASK

    def test_invalid_egt_length_raises(self) -> None:
        with pytest.raises(ValueError, match="EGT must contain exactly 4 cylinders"):
            TelemetryFrame(
                t=0.0,
                egt=(1000.0, 1000.0),  # len 2 invalid
                cht=360.0,
                p_oil=350000.0,
                t_oil=360.0,
                n_rpm=5000.0,
                mdot_f=0.005,
                map_pa=100000.0,
                tps=0.7,
                t_im=300.0,
                p_amb=80000.0,
                t_amb=275.0,
                v_tas=60.0,
                h_p=1000.0,
            )

    def test_validate_telemetry_dataframe_missing_column_raises(self) -> None:
        df_bad = pl.DataFrame({"timestamp": [100.0], "cht": [360.0]})
        with pytest.raises(ValueError, match="Missing required telemetry column"):
            validate_telemetry_dataframe(df_bad)

    def test_empty_dataframe_and_table_conversions(self) -> None:
        empty_df = telemetry_frames_to_polars([])
        assert empty_df.height == 0
        assert empty_df.schema == TELEMETRY_POLARS_SCHEMA

        empty_frames = polars_to_telemetry_frames(empty_df)
        assert len(empty_frames) == 0

    def test_validity_flag_helpers(self, sample_telemetry_frame: TelemetryFrame) -> None:
        f = sample_telemetry_frame
        assert f.is_valid(TelemetryValidityFlag.EGT1_VALID) is True
        assert f.is_valid(FLAG_EGT1_VALID) is True
        assert f.is_valid(TelemetryValidityFlag.NONE) is False
