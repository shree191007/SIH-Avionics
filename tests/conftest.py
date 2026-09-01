"""E2E Test Suite Fixtures and Mock Data Generators for SIH26054 Phase 1.

Provides reusable fixtures, telemetry sample generators, temporary artifact directories,
mock calibration manifests, and synthetic flight data generators.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from tests.constants import (
    ALL_CHANNELS_VALID_MASK,
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
    REGIME_GRID_VERSION_V1,
    RESIDUAL_SCHEMA_VERSION_V1,
    STATUS_DEGRADED_INPUT,
    STATUS_INVALID_SENSOR,
    STATUS_MODEL_SATURATED,
    STATUS_OK,
    TELEMETRY_SCHEMA_VERSION_V1,
)

# Ensure src/ and workspace root are on sys.path for test discovery and imports
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = WORKSPACE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


# ============================================================================
# Shared Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def workspace_root() -> Path:
    """Returns the absolute path to the workspace root directory."""
    return WORKSPACE_ROOT


@pytest.fixture
def temp_artifact_registry(tmp_path: Path) -> Path:
    """Creates a temporary artifact registry with all 7 mandatory subtrees."""
    registry_root = tmp_path / "artifacts"
    subtrees = [
        "raw",
        "processed",
        "calibration",
        "models",
        "fingerprints",
        "replay",
        "evaluation",
    ]
    for subtree in subtrees:
        (registry_root / subtree).mkdir(parents=True, exist_ok=True)
    return registry_root


@pytest.fixture
def mock_calibration_manifest_hash() -> str:
    """Returns a deterministic 64-character SHA256 hex digest for calibration."""
    content = b"ROTAX_915_IS_A_CALIBRATION_PARAMS_V1_2026_08"
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def sample_8field_manifest_dict(mock_calibration_manifest_hash: str) -> Dict[str, Any]:
    """Returns a valid dictionary containing all 8 required provenance fields."""
    return {
        "dataset_id": "DS_RAW_NGAFID_2026_08",
        "flight_id": "7152",
        "aircraft_id": "AC_01",
        "engine_id": "ROTAX_915_IS_SN1042",
        "model_version": "1.0.0",
        "regime_grid_version": REGIME_GRID_VERSION_V1,
        "calibration_manifest_hash": mock_calibration_manifest_hash,
        "schema_version": TELEMETRY_SCHEMA_VERSION_V1,
        "extensions": {
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "payload_file": "flight_7152.parquet",
            "payload_format": "PARQUET",
            "payload_file_sha256": "a" * 64,
            "canonical_data_sha256": "b" * 64,
            "row_count": 1800,
            "duration_s": 1800.0,
            "parent_manifest_hashes": [],
            "split_group": "train",
            "status_flags": ["HEALTHY", "COMPLETE"],
        },
    }


@pytest.fixture
def nominal_telemetry_dict() -> Dict[str, Any]:
    """Returns a single nominal 1 Hz telemetry sample as a dictionary."""
    return {
        "timestamp": 100.0,
        "egt_1": 1053.15,  # 780 °C in K
        "egt_2": 1048.15,  # 775 °C in K
        "egt_3": 1050.15,  # 777 °C in K
        "egt_4": 1055.15,  # 782 °C in K
        "cht": 368.15,     # 95 °C in K
        "oil_pressure": 350000.0,  # 3.5 bar in Pa
        "oil_temperature": 363.15, # 90 °C in K
        "engine_speed_rpm": 5000.0,
        "fuel_flow_kg_s": 0.0055,  # ~25 L/h
        # 7 Exogenous inputs
        "map_pa": 125000.0,        # 125 kPa (cruise boost)
        "throttle_position": 0.75, # 75%
        "intake_temp_k": 310.15,   # 37 °C
        "ambient_pressure_pa": 85000.0, # ~1500m MSL
        "ambient_temp_k": 288.15,  # 15 °C
        "true_airspeed_ms": 65.0,  # ~126 kts
        "pressure_altitude_m": 1500.0,
        "validity_bitmask": ALL_CHANNELS_VALID_MASK,
        "flight_id": "7152",
        "aircraft_id": "AC_01",
        "engine_id": "ROTAX_915_IS_SN1042",
    }


@pytest.fixture
def synthetic_flight_dataframe() -> pl.DataFrame:
    """Generates a synthetic 300-second nominal cruise flight Polars DataFrame."""
    n_samples = 300
    t = np.arange(n_samples, dtype=np.float64)
    rng = np.random.default_rng(seed=42)

    # Primary channels with mild realistic Gaussian noise
    egt_1 = 1053.15 + rng.normal(0, 1.5, n_samples)
    egt_2 = 1048.15 + rng.normal(0, 1.5, n_samples)
    egt_3 = 1050.15 + rng.normal(0, 1.5, n_samples)
    egt_4 = 1055.15 + rng.normal(0, 1.5, n_samples)
    cht = 368.15 + rng.normal(0, 0.5, n_samples)
    oil_pressure = 350000.0 + rng.normal(0, 1000.0, n_samples)
    oil_temperature = 363.15 + rng.normal(0, 0.3, n_samples)
    engine_speed_rpm = 5000.0 + rng.normal(0, 5.0, n_samples)
    fuel_flow_kg_s = 0.0055 + rng.normal(0, 0.00005, n_samples)

    # 7 Exogenous inputs
    map_pa = 125000.0 + rng.normal(0, 100.0, n_samples)
    throttle_position = np.full(n_samples, 0.75, dtype=np.float64)
    intake_temp_k = 310.15 + rng.normal(0, 0.2, n_samples)
    ambient_pressure_pa = 85000.0 + rng.normal(0, 50.0, n_samples)
    ambient_temp_k = 288.15 + rng.normal(0, 0.1, n_samples)
    true_airspeed_ms = 65.0 + rng.normal(0, 0.2, n_samples)
    pressure_altitude_m = 1500.0 + rng.normal(0, 1.0, n_samples)

    validity_bitmask = np.full(n_samples, ALL_CHANNELS_VALID_MASK, dtype=np.uint32)
    flight_id = ["flight_001"] * n_samples
    aircraft_id = ["AC_01"] * n_samples
    engine_id = ["ROTAX_915_IS_SN1042"] * n_samples

    return pl.DataFrame({
        "timestamp": t,
        "egt_1": egt_1,
        "egt_2": egt_2,
        "egt_3": egt_3,
        "egt_4": egt_4,
        "cht": cht,
        "oil_pressure": oil_pressure,
        "oil_temperature": oil_temperature,
        "engine_speed_rpm": engine_speed_rpm,
        "fuel_flow_kg_s": fuel_flow_kg_s,
        "map_pa": map_pa,
        "throttle_position": throttle_position,
        "intake_temp_k": intake_temp_k,
        "ambient_pressure_pa": ambient_pressure_pa,
        "ambient_temp_k": ambient_temp_k,
        "true_airspeed_ms": true_airspeed_ms,
        "pressure_altitude_m": pressure_altitude_m,
        "validity_bitmask": validity_bitmask,
        "flight_id": flight_id,
        "aircraft_id": aircraft_id,
        "engine_id": engine_id,
    })


@pytest.fixture
def multi_aircraft_fleet_dataframe() -> pl.DataFrame:
    """Generates a synthetic fleet DataFrame with 3 aircraft and 6 flights."""
    records: List[Dict[str, Any]] = []
    fleet_configs = [
        ("AC_01", "ROTAX_915_SN01", ["FL_101", "FL_102"]),
        ("AC_02", "ROTAX_915_SN02", ["FL_201", "FL_202"]),
        ("AC_03", "ROTAX_915_SN03", ["FL_301", "FL_302"]),
    ]

    rng = np.random.default_rng(seed=123)

    for ac_id, eng_id, flight_ids in fleet_configs:
        for fl_id in flight_ids:
            n_rows = 100
            for sec in range(n_rows):
                records.append({
                    "timestamp": float(sec),
                    "egt_1": 1050.0 + rng.normal(0, 1.0),
                    "egt_2": 1045.0 + rng.normal(0, 1.0),
                    "egt_3": 1048.0 + rng.normal(0, 1.0),
                    "egt_4": 1052.0 + rng.normal(0, 1.0),
                    "cht": 368.0 + rng.normal(0, 0.5),
                    "oil_pressure": 350000.0 + rng.normal(0, 500.0),
                    "oil_temperature": 363.0 + rng.normal(0, 0.3),
                    "engine_speed_rpm": 5000.0 + rng.normal(0, 5.0),
                    "fuel_flow_kg_s": 0.0055,
                    "map_pa": 125000.0,
                    "throttle_position": 0.75,
                    "intake_temp_k": 310.0,
                    "ambient_pressure_pa": 85000.0,
                    "ambient_temp_k": 288.0,
                    "true_airspeed_ms": 65.0,
                    "pressure_altitude_m": 1500.0,
                    "validity_bitmask": ALL_CHANNELS_VALID_MASK,
                    "flight_id": fl_id,
                    "aircraft_id": ac_id,
                    "engine_id": eng_id,
                })

    return pl.DataFrame(records)


@pytest.fixture
def synthetic_pyarrow_telemetry_table(synthetic_flight_dataframe: pl.DataFrame) -> pa.Table:
    """Converts the synthetic Polars DataFrame to a PyArrow Table."""
    return synthetic_flight_dataframe.to_arrow()


# ============================================================================
# Contract Unit Test Fixtures
# ============================================================================

from replan_to_learn.contracts.telemetry import TelemetryFrame, FULL_VALID_MASK
from replan_to_learn.contracts.residuals import ResidualFrame, STATUS_OK
from replan_to_learn.contracts.health import HealthParameters


@pytest.fixture
def sample_telemetry_frame() -> TelemetryFrame:
    """Fixture providing a standard nominal cruise TelemetryFrame."""
    return TelemetryFrame(
        t=100.0,
        egt=(1050.0, 1045.0, 1055.0, 1050.0),
        cht=365.0,
        p_oil=350000.0,
        t_oil=360.0,
        n_rpm=5000.0,
        mdot_f=0.0055,
        map_pa=105000.0,
        tps=0.75,
        t_im=310.0,
        p_amb=80000.0,
        t_amb=275.0,
        v_tas=65.0,
        h_p=2000.0,
        valid_mask=FULL_VALID_MASK,
        flight_id="FLIGHT_001",
        aircraft_id="AIRCRAFT_A",
        engine_id="ENGINE_ROTAX_915_001",
    )


@pytest.fixture
def sample_telemetry_sequence() -> List[TelemetryFrame]:
    """Fixture providing a 30-second quasi-steady cruise telemetry sequence."""
    frames = []
    base_t = 0.0
    for i in range(30):
        t = base_t + float(i)
        noise_egt = (float(np.random.normal(0, 0.5)), float(np.random.normal(0, 0.5)),
                     float(np.random.normal(0, 0.5)), float(np.random.normal(0, 0.5)))
        frame = TelemetryFrame(
            t=t,
            egt=(1050.0 + noise_egt[0], 1048.0 + noise_egt[1], 1052.0 + noise_egt[2], 1050.0 + noise_egt[3]),
            cht=365.0 + float(np.random.normal(0, 0.1)),
            p_oil=350000.0 + float(np.random.normal(0, 500)),
            t_oil=360.0 + float(np.random.normal(0, 0.1)),
            n_rpm=5000.0 + float(np.random.normal(0, 1.0)),
            mdot_f=0.0055 + float(np.random.normal(0, 0.00001)),
            map_pa=105000.0 + float(np.random.normal(0, 50)),
            tps=0.75,
            t_im=310.0,
            p_amb=80000.0,
            t_amb=275.0,
            v_tas=65.0,
            h_p=2000.0,
            valid_mask=FULL_VALID_MASK,
            flight_id="FLIGHT_TEST_SEQ",
            aircraft_id="AIRCRAFT_A",
            engine_id="ENGINE_ROTAX_001",
        )
        frames.append(frame)
    return frames


@pytest.fixture
def sample_residual_frame() -> ResidualFrame:
    """Fixture providing a standard nominal ResidualFrame."""
    return ResidualFrame(
        t=100.0,
        z=(0.12, -0.05, 0.08, -0.15, 0.20, -0.10, 0.05, 0.15, -0.08),
        regime=4,
        regime_stable_s=25.0,
        spatial=(0.0, 0.0, 0.0, 1),
        sigma=(8.0, 8.0, 8.0, 8.0, 2.5, 12000.0, 1.5, 0.00012, 2.0),
        x_hat=(105000.0, 365.0, 360.0, 1050.0, 523.6, 365.0, 365.0, 365.0, 365.0),
        status=STATUS_OK,
        model_version="1.0.0",
        regime_grid_version="REGIME_GRID_V1",
        flight_id="FLIGHT_001",
    )


@pytest.fixture
def nominal_health_parameters() -> HealthParameters:
    """Fixture providing as-new nominal health parameters."""
    return HealthParameters()

