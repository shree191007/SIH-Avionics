"""
SIH26054 Phase 1: Unit Tests for R2 Provenance Manifest Schema, Generator & Verification.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import tempfile
from typing import Any, Dict

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError
import pytest

from replan_to_learn.provenance.hasher import canonical_json_dumps, payload_file_sha256
from replan_to_learn.provenance.manifest import (
    PARQUET_MANIFEST_KEY,
    REGIME_GRID_VERSION_V1,
    ManifestGenerator,
    ProvenanceAuditMetadata,
    ProvenanceManifest,
    VerificationResult,
    embed_manifest_in_schema,
    read_manifest_from_parquet,
    verify_manifest,
    write_parquet_with_manifest,
)


@pytest.fixture
def valid_manifest_dict() -> Dict[str, Any]:
    return {
        "dataset_id": "DS_RAW_NGAFID_2026_08",
        "flight_id": "7152",
        "aircraft_id": "AC_01",
        "engine_id": "ROTAX_915_IS_SN1042",
        "model_version": "1.0.0",
        "regime_grid_version": REGIME_GRID_VERSION_V1,
        "calibration_manifest_hash": "a" * 64,
        "schema_version": "TELEMETRY_SCHEMA_V1",
        "extensions": {
            "created_at_utc": "2026-08-26T21:00:00.000000Z",
            "created_by": "test_suite",
            "payload_file": "flight_7152.parquet",
            "payload_format": "PARQUET",
            "payload_file_sha256": "b" * 64,
            "canonical_data_sha256": "c" * 64,
            "row_count": 1800,
            "duration_s": 1800.0,
            "parent_manifest_hashes": ["d" * 64],
            "split_group": "train",
            "status_flags": ["HEALTHY", "COMPLETE"],
            "extra_metadata": {"pilot_notes": "nominal climb and cruise"},
        },
    }


# ============================================================================
# 1. 8 Required Fields Validation Tests
# ============================================================================

def test_manifest_instantiation_with_valid_dict(valid_manifest_dict: Dict[str, Any]):
    manifest = ProvenanceManifest.from_dict(valid_manifest_dict)
    assert manifest.dataset_id == "DS_RAW_NGAFID_2026_08"
    assert manifest.flight_id == "7152"
    assert manifest.aircraft_id == "AC_01"
    assert manifest.engine_id == "ROTAX_915_IS_SN1042"
    assert manifest.model_version == "1.0.0"
    assert manifest.regime_grid_version == REGIME_GRID_VERSION_V1
    assert manifest.calibration_manifest_hash == "a" * 64
    assert manifest.schema_version == "TELEMETRY_SCHEMA_V1"
    assert manifest.row_count == 1800
    assert manifest.split_group == "train"


def test_manifest_missing_required_fields_raises_validation_error():
    required_fields = [
        "dataset_id", "flight_id", "aircraft_id", "engine_id",
        "model_version", "regime_grid_version",
        "calibration_manifest_hash", "schema_version"
    ]
    base_data = {
        "dataset_id": "DS_RAW_01",
        "flight_id": "FL_101",
        "aircraft_id": "AC_01",
        "engine_id": "ENG_01",
        "model_version": "1.0.0",
        "regime_grid_version": "REGIME_GRID_V1",
        "calibration_manifest_hash": "NONE",
        "schema_version": "TELEMETRY_SCHEMA_V1",
    }
    for field in required_fields:
        corrupt_data = dict(base_data)
        del corrupt_data[field]
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(corrupt_data)


@pytest.mark.parametrize("invalid_dataset_id", ["", "a", "a" * 150, "invalid/slash", "bad#char"])
def test_manifest_invalid_dataset_id_rejected(valid_manifest_dict: Dict[str, Any], invalid_dataset_id: str):
    d = dict(valid_manifest_dict)
    d["dataset_id"] = invalid_dataset_id
    with pytest.raises(ValidationError):
        ProvenanceManifest.from_dict(d)


@pytest.mark.parametrize("invalid_flight_id", ["", "a" * 70, "bad space"])
def test_manifest_invalid_flight_id_rejected(valid_manifest_dict: Dict[str, Any], invalid_flight_id: str):
    d = dict(valid_manifest_dict)
    d["flight_id"] = invalid_flight_id
    with pytest.raises(ValidationError):
        ProvenanceManifest.from_dict(d)


def test_manifest_flight_id_multi_is_accepted(valid_manifest_dict: Dict[str, Any]):
    d = dict(valid_manifest_dict)
    d["flight_id"] = "MULTI"
    manifest = ProvenanceManifest.from_dict(d)
    assert manifest.flight_id == "MULTI"


@pytest.mark.parametrize("invalid_aircraft_id", ["", "a", "a" * 70, "AC/01", "AC 01"])
def test_manifest_invalid_aircraft_id_rejected(valid_manifest_dict: Dict[str, Any], invalid_aircraft_id: str):
    d = dict(valid_manifest_dict)
    d["aircraft_id"] = invalid_aircraft_id
    with pytest.raises(ValidationError):
        ProvenanceManifest.from_dict(d)


@pytest.mark.parametrize("invalid_model_version", ["", "invalid_ver", "1.0", "v1"])
def test_manifest_invalid_model_version_rejected(valid_manifest_dict: Dict[str, Any], invalid_model_version: str):
    d = dict(valid_manifest_dict)
    d["model_version"] = invalid_model_version
    with pytest.raises(ValidationError):
        ProvenanceManifest.from_dict(d)


@pytest.mark.parametrize("valid_model_version", ["1.0.0", "v1.2.3", "2.1.0-alpha.1", "9f8e7d6", "a" * 40])
def test_manifest_valid_model_versions_accepted(valid_manifest_dict: Dict[str, Any], valid_model_version: str):
    d = dict(valid_manifest_dict)
    d["model_version"] = valid_model_version
    manifest = ProvenanceManifest.from_dict(d)
    assert manifest.model_version == valid_model_version


@pytest.mark.parametrize("invalid_regime_grid", ["", "REGIME_GRID", "GRID_V1", "REGIME_GRID_V0", "REGIME_GRID_VX"])
def test_manifest_invalid_regime_grid_version_rejected(valid_manifest_dict: Dict[str, Any], invalid_regime_grid: str):
    d = dict(valid_manifest_dict)
    d["regime_grid_version"] = invalid_regime_grid
    with pytest.raises(ValidationError):
        ProvenanceManifest.from_dict(d)


@pytest.mark.parametrize("invalid_calib_hash", ["", "NONE_WRONG", "123", "a" * 63, "a" * 65, "g" * 64])
def test_manifest_invalid_calibration_hash_rejected(valid_manifest_dict: Dict[str, Any], invalid_calib_hash: str):
    d = dict(valid_manifest_dict)
    d["calibration_manifest_hash"] = invalid_calib_hash
    with pytest.raises(ValidationError):
        ProvenanceManifest.from_dict(d)


def test_manifest_calibration_hash_none_accepted(valid_manifest_dict: Dict[str, Any]):
    d = dict(valid_manifest_dict)
    d["calibration_manifest_hash"] = "NONE"
    manifest = ProvenanceManifest.from_dict(d)
    assert manifest.calibration_manifest_hash == "NONE"


# ============================================================================
# 2. ManifestGenerator Tests
# ============================================================================

def test_manifest_generator_create_raw_manifest(synthetic_flight_dataframe: pl.DataFrame):
    manifest = ManifestGenerator.create_raw_manifest(
        dataset_id="DS_RAW_SYNTHETIC",
        flight_id="flight_001",
        aircraft_id="AC_01",
        engine_id="ROTAX_915_IS_SN1042",
        data=synthetic_flight_dataframe,
    )
    assert manifest.dataset_id == "DS_RAW_SYNTHETIC"
    assert manifest.calibration_manifest_hash == "NONE"
    assert manifest.row_count == 300
    assert manifest.duration_s == 299.0
    assert len(manifest.canonical_data_sha256) == 64
    assert manifest.schema_version == "TELEMETRY_SCHEMA_V1"


def test_manifest_generator_create_processed_manifest(synthetic_flight_dataframe: pl.DataFrame):
    raw_hash = "1" * 64
    calib_hash = "2" * 64
    manifest = ManifestGenerator.create_processed_manifest(
        dataset_id="DS_PROC_SYNTHETIC",
        flight_id="flight_001",
        aircraft_id="AC_01",
        engine_id="ROTAX_915_IS_SN1042",
        model_version="1.0.0",
        calibration_manifest_hash=calib_hash,
        parent_manifest_hashes=[raw_hash],
        data=synthetic_flight_dataframe,
        split_group="train",
    )
    assert manifest.dataset_id == "DS_PROC_SYNTHETIC"
    assert manifest.calibration_manifest_hash == calib_hash
    assert manifest.parent_manifest_hashes == [raw_hash]
    assert manifest.split_group == "train"


# ============================================================================
# 3. Serialization & Parquet Roundtrips
# ============================================================================

def test_manifest_to_canonical_json_roundtrip(valid_manifest_dict: Dict[str, Any]):
    manifest = ProvenanceManifest.from_dict(valid_manifest_dict)
    canon_json = manifest.to_canonical_json()
    assert isinstance(canon_json, str)
    reloaded = ProvenanceManifest.from_json(canon_json)
    assert reloaded.dataset_id == manifest.dataset_id
    assert reloaded.compute_manifest_hash() == manifest.compute_manifest_hash()


def test_manifest_save_and_load_sidecar_file(valid_manifest_dict: Dict[str, Any], tmp_path: Path):
    manifest = ProvenanceManifest.from_dict(valid_manifest_dict)
    save_path = tmp_path / "test_manifest.json"
    manifest.save(save_path)
    assert save_path.exists()

    loaded = ProvenanceManifest.load(save_path)
    assert loaded.compute_manifest_hash() == manifest.compute_manifest_hash()


def test_parquet_metadata_embedding_and_extraction(
    synthetic_flight_dataframe: pl.DataFrame,
    valid_manifest_dict: Dict[str, Any],
    tmp_path: Path
):
    manifest = ProvenanceManifest.from_dict(valid_manifest_dict)
    pq_path = tmp_path / "telemetry_with_meta.parquet"

    write_parquet_with_manifest(synthetic_flight_dataframe, pq_path, manifest)
    assert pq_path.exists()

    # Verify zero-copy manifest extraction from Parquet header
    extracted_manifest = read_manifest_from_parquet(pq_path)
    assert extracted_manifest.dataset_id == manifest.dataset_id
    assert extracted_manifest.aircraft_id == manifest.aircraft_id
    assert extracted_manifest.compute_manifest_hash() == manifest.compute_manifest_hash()


# ============================================================================
# 4. Automated Verification Engine (verify_manifest) Tests
# ============================================================================

def test_verify_manifest_happy_path(
    synthetic_flight_dataframe: pl.DataFrame,
    tmp_path: Path
):
    pq_path = tmp_path / "valid_artifact.parquet"
    manifest = ManifestGenerator.create_raw_manifest(
        dataset_id="DS_VERIFY_TEST",
        flight_id="FL_001",
        aircraft_id="AC_01",
        engine_id="ENG_01",
        data=synthetic_flight_dataframe,
    )
    # Write Parquet with embedded manifest
    write_parquet_with_manifest(synthetic_flight_dataframe, pq_path, manifest)

    # 1. Verify directly from embedded Parquet metadata
    result_embedded = verify_manifest(pq_path)
    assert result_embedded.is_valid is True
    assert result_embedded.status == "PASSED"
    assert len(result_embedded.errors) == 0

    # 2. Verify with explicit manifest object matching embedded
    result_explicit = verify_manifest(pq_path, manifest)
    assert result_explicit.is_valid is True
    assert result_explicit.status == "PASSED"


def test_verify_manifest_detects_tampered_parquet_data(
    synthetic_flight_dataframe: pl.DataFrame,
    tmp_path: Path
):
    manifest = ManifestGenerator.create_raw_manifest(
        dataset_id="DS_TAMPER_TEST",
        flight_id="FL_001",
        aircraft_id="AC_01",
        engine_id="ENG_01",
        data=synthetic_flight_dataframe,
    )
    pq_path = tmp_path / "tampered_artifact.parquet"

    # Save original
    write_parquet_with_manifest(synthetic_flight_dataframe, pq_path, manifest)

    # Create tampered dataframe with 1 modified value
    tampered_df = synthetic_flight_dataframe.with_columns(
        pl.when(pl.col("timestamp") == 100.0)
        .then(pl.col("egt_1") + 50.0)
        .otherwise(pl.col("egt_1"))
        .alias("egt_1")
    )
    # Overwrite file with tampered table
    pq.write_table(tampered_df.to_arrow(), pq_path)

    # Verify against original manifest expecting canonical data hash failure
    result = verify_manifest(pq_path, manifest)
    assert result.is_valid is False
    assert result.status == "FAILED"
    assert any("Canonical data SHA256 mismatch" in err for err in result.errors)


def test_verify_manifest_detects_nonexistent_file(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.parquet"
    result = verify_manifest(missing_path)
    assert result.is_valid is False
    assert result.status == "FAILED"
    assert "does not exist" in result.errors[0]
