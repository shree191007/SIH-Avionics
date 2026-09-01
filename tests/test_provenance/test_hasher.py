"""
SIH26054 Phase 1: Unit Tests for R2 Cryptographic & Canonical Hashing Engine (hasher.py).
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import polars as pl
import pyarrow as pa
import pytest

from replan_to_learn.provenance.hasher import (
    canonical_data_sha256,
    canonical_json_dumps,
    canonical_json_loads,
    compute_manifest_hash,
    payload_file_sha256,
)
from replan_to_learn.provenance.manifest import (
    REGIME_GRID_VERSION_V1,
    ProvenanceManifest,
)


# ============================================================================
# 1. RFC 8785 JSON Canonicalization Scheme (JCS) Tests
# ============================================================================

def test_canonical_json_dumps_lexicographical_sorting():
    """Verify dictionary keys are sorted lexicographically by UTF-8 Unicode code points."""
    raw_dict = {
        "z_channel": 1,
        "a_channel": 2,
        "m_channel": {"beta": 10, "alpha": 20},
        "b_channel": [3, 2, 1],
    }
    dumped = canonical_json_dumps(raw_dict)
    expected = '{"a_channel":2,"b_channel":[3,2,1],"m_channel":{"alpha":20,"beta":10},"z_channel":1}'
    assert dumped == expected


def test_canonical_json_dumps_compact_separators():
    """Verify separators have no whitespace after colon or comma."""
    raw_dict = {"key1": "val1", "key2": [1, 2, 3]}
    dumped = canonical_json_dumps(raw_dict)
    assert ": " not in dumped
    assert ", " not in dumped


def test_canonical_json_dumps_rejects_nan_and_inf():
    """Verify NaN, Infinity, and -Infinity are strictly rejected per RFC 8785."""
    with pytest.raises(ValueError, match="NaN and Infinity are not permitted"):
        canonical_json_dumps({"invalid_val": float("nan")})

    with pytest.raises(ValueError, match="NaN and Infinity are not permitted"):
        canonical_json_dumps({"invalid_val": float("inf")})

    with pytest.raises(ValueError, match="NaN and Infinity are not permitted"):
        canonical_json_dumps({"invalid_val": float("-inf")})


def test_canonical_json_dumps_datetime_normalization():
    """Verify datetime objects are normalized to ISO 8601 UTC with 'Z' suffix."""
    dt_naive = datetime.datetime(2026, 8, 26, 21, 0, 0)
    dt_utc = datetime.datetime(2026, 8, 26, 21, 0, 0, tzinfo=datetime.timezone.utc)

    dumped_naive = canonical_json_dumps({"time": dt_naive})
    dumped_utc = canonical_json_dumps({"time": dt_utc})

    assert dumped_naive == '{"time":"2026-08-26T21:00:00Z"}'
    assert dumped_utc == '{"time":"2026-08-26T21:00:00Z"}'


def test_canonical_json_dumps_and_loads_dataclass_and_pydantic():
    """Verify dataclasses and Pydantic models serialize cleanly to canonical JSON."""
    @dataclasses.dataclass
    class SampleConfig:
        name: str
        rate_hz: int
        sub: Dict[str, Any]

    cfg = SampleConfig(name="telemetry", rate_hz=10, sub={"b": 2, "a": 1})
    dumped = canonical_json_dumps(cfg)
    assert dumped == '{"name":"telemetry","rate_hz":10,"sub":{"a":1,"b":2}}'

    parsed = canonical_json_loads(dumped)
    assert parsed["name"] == "telemetry"
    assert parsed["sub"]["a"] == 1


# ============================================================================
# 2. File Checksum Computation (payload_file_sha256) Tests
# ============================================================================

def test_payload_file_sha256_streaming_equality(tmp_path: Path):
    """Verify chunked file SHA256 computation matches standard single-pass hashlib."""
    test_file = tmp_path / "large_dummy.bin"
    payload = b"SIH26054_TELEMETRY_PAYLOAD_" * 35000
    test_file.write_bytes(payload)

    expected_hash = hashlib.sha256(payload).hexdigest().lower()
    computed_hash = payload_file_sha256(test_file, chunk_size=4096)

    assert computed_hash == expected_hash
    assert len(computed_hash) == 64


def test_payload_file_sha256_file_not_found(tmp_path: Path):
    """Verify FileNotFoundError on non-existent path."""
    missing = tmp_path / "missing_file.parquet"
    with pytest.raises(FileNotFoundError):
        payload_file_sha256(missing)


def test_payload_file_sha256_is_directory_error(tmp_path: Path):
    """Verify IsADirectoryError when passing a directory."""
    with pytest.raises(IsADirectoryError):
        payload_file_sha256(tmp_path)


def test_payload_file_sha256_tamper_detection(tmp_path: Path):
    """Verify modifying a single byte changes the file SHA256 completely."""
    f1 = tmp_path / "clean.bin"
    f2 = tmp_path / "tampered.bin"

    data1 = bytearray(b"AIRCRAFT_01_NOMINAL_CRUISE_RUN")
    data2 = bytearray(b"AIRCRAFT_01_NOMINAL_CRUISE_RUN")
    data2[5] = ord("X")

    f1.write_bytes(data1)
    f2.write_bytes(data2)

    hash1 = payload_file_sha256(f1)
    hash2 = payload_file_sha256(f2)

    assert hash1 != hash2


# ============================================================================
# 3. Canonical Arrow Data Stream Hashing (canonical_data_sha256) Tests
# ============================================================================

def test_canonical_data_sha256_polars_vs_arrow_equivalence():
    """Verify pl.DataFrame and pa.Table containing identical data yield identical SHA256."""
    pydict = {
        "timestamp": [0.0, 1.0, 2.0, 3.0],
        "egt_1": [1050.0, 1052.5, 1051.0, 1053.0],
        "cht": [450.0, 451.0, 450.5, 452.0],
        "oil_pressure": [350000.0, 351000.0, 349500.0, 350500.0],
    }
    df = pl.DataFrame(pydict)
    table = pa.Table.from_pydict(pydict)

    hash_df = canonical_data_sha256(df)
    hash_table = canonical_data_sha256(table)

    assert hash_df == hash_table
    assert len(hash_df) == 64


def test_canonical_data_sha256_column_order_independence():
    """Verify column ordering differences do NOT affect canonical hash (columns sorted lexicographically)."""
    table1 = pa.Table.from_pydict({
        "oil_pressure": [350000.0, 351000.0],
        "cht": [450.0, 451.0],
        "egt_1": [1050.0, 1052.5],
    })
    table2 = pa.Table.from_pydict({
        "egt_1": [1050.0, 1052.5],
        "oil_pressure": [350000.0, 351000.0],
        "cht": [450.0, 451.0],
    })

    hash1 = canonical_data_sha256(table1)
    hash2 = canonical_data_sha256(table2)

    assert hash1 == hash2


def test_canonical_data_sha256_metadata_independence():
    """Verify adding schema metadata does NOT change the data hash (metadata is stripped)."""
    table_plain = pa.Table.from_pydict({"val": [1.0, 2.0, 3.0]})
    
    meta_schema = table_plain.schema.with_metadata({b"custom.write_time": b"2026-08-26T21:00:00Z"})
    table_with_meta = table_plain.replace_schema_metadata(meta_schema.metadata)

    hash_plain = canonical_data_sha256(table_plain)
    hash_with_meta = canonical_data_sha256(table_with_meta)

    assert hash_plain == hash_with_meta


def test_canonical_data_sha256_chunking_independence():
    """Verify chunked table vs single-chunk table yields identical hash."""
    batch1 = pa.RecordBatch.from_pydict({"col": [1.0, 2.0]})
    batch2 = pa.RecordBatch.from_pydict({"col": [3.0, 4.0]})
    chunked_table = pa.Table.from_batches([batch1, batch2])
    single_chunk_table = chunked_table.combine_chunks()

    hash_chunked = canonical_data_sha256(chunked_table)
    hash_single = canonical_data_sha256(single_chunk_table)

    assert hash_chunked == hash_single


def test_canonical_data_sha256_record_batch_reader_and_sequence():
    """Verify pa.RecordBatchReader and list of batches can be hashed directly."""
    batch1 = pa.RecordBatch.from_pydict({"a": [10, 20]})
    batch2 = pa.RecordBatch.from_pydict({"a": [30, 40]})
    batches = [batch1, batch2]

    reader = pa.RecordBatchReader.from_batches(batch1.schema, batches)
    hash_reader = canonical_data_sha256(reader)

    hash_list = canonical_data_sha256(batches)
    assert hash_reader == hash_list
    assert len(hash_reader) == 64


def test_canonical_data_sha256_detects_data_mutation():
    """Verify mutating a single floating point value alters the hash."""
    df1 = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    df2 = pl.DataFrame({"a": [1.0, 2.0000001, 3.0]})

    assert canonical_data_sha256(df1) != canonical_data_sha256(df2)


def test_canonical_data_sha256_empty_dataset():
    """Verify canonical hashing of empty table / dataframe produces deterministic 64-char hex."""
    empty_df = pl.DataFrame({"a": []}, schema={"a": pl.Float64})
    h = canonical_data_sha256(empty_df)
    assert len(h) == 64
    assert h == canonical_data_sha256(empty_df.to_arrow())


def test_canonical_data_sha256_unsupported_type_raises():
    """Verify TypeError on invalid input types."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        canonical_data_sha256("invalid string")


# ============================================================================
# 4. Manifest Self-Hash (compute_manifest_hash) Tests
# ============================================================================

def test_compute_manifest_hash_omits_manifest_hash_field():
    """Verify compute_manifest_hash strips self-referential manifest_hash field."""
    base_dict = {
        "dataset_id": "DS_01",
        "flight_id": "FL_101",
        "aircraft_id": "AC_01",
        "engine_id": "ENG_01",
        "model_version": "1.0.0",
        "regime_grid_version": REGIME_GRID_VERSION_V1,
        "calibration_manifest_hash": "NONE",
        "schema_version": "TELEMETRY_SCHEMA_V1",
    }
    dict_with_hash = dict(base_dict)
    dict_with_hash["manifest_hash"] = "9" * 64

    hash_clean = compute_manifest_hash(base_dict)
    hash_with = compute_manifest_hash(dict_with_hash)

    assert hash_clean == hash_with
    assert len(hash_clean) == 64


def test_compute_manifest_hash_pydantic_manifest_consistency():
    """Verify compute_manifest_hash works consistently with ProvenanceManifest objects."""
    manifest = ProvenanceManifest(
        dataset_id="DS_RAW_01",
        flight_id="FL_01",
        aircraft_id="AC_01",
        engine_id="ENG_01",
        model_version="1.0.0",
        regime_grid_version=REGIME_GRID_VERSION_V1,
        calibration_manifest_hash="NONE",
        schema_version="TELEMETRY_SCHEMA_V1",
    )
    h1 = manifest.compute_manifest_hash()
    h2 = compute_manifest_hash(manifest)
    h3 = compute_manifest_hash(manifest.to_dict())

    assert h1 == h2 == h3
    assert len(h1) == 64
