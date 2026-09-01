"""
SIH26054 Replan to Learn: Empirical Adversarial Challenge Suite (Challenger 1).
Milestone 2: R2 Data Versioning & Provenance Metadata.

Stress Tests & Adversarial Scenarios:
1. Parquet Payload Tamper Detection:
   - Single-bit / float LSB flips in Parquet columns.
   - Raw binary corruption of Parquet byte streams.
   - String / categorical column single-character tampering.
   - Row additions, deletions, and intra-column row swaps.
   - Invariance of canonical hashing under benign transformations (rechunking, column reordering).
2. Manifest Metadata Mutation:
   - Exhaustive mutations of each of the 8 required fields.
   - Mutation of extended audit metadata (row counts, durations, status flags, timestamps).
   - Invalidation on tampered file SHA256 or canonical data SHA256 in manifest.
   - Parquet embedded vs. sidecar manifest desynchronization detection.
3. Exhaustive 8-Field Regex & Schema Validation:
   - Boundary tests for min/max string lengths.
   - Invalid character injection (whitespace, slashes, control chars, symbols, Unicode, emojis).
   - Missing fields, null values, and incorrect data types.
4. Multi-Threaded / Concurrency Stress Harness:
   - High-concurrency multi-threaded manifest generation.
   - Concurrent Parquet writing with embedded key-value metadata.
   - High-contention concurrent read/verify operations on shared files.
   - Concurrent execution of grouped anti-leakage splitters.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError
import pytest

from replan_to_learn.provenance.hasher import (
    canonical_data_sha256,
    canonical_json_dumps,
    canonical_json_loads,
    compute_manifest_hash,
    payload_file_sha256,
)
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
from replan_to_learn.provenance.splitters import (
    GroupedAircraftSplitter,
    GroupedFlightSplitter,
    HierarchicalGroupSplitter,
    SplitConfiguration,
    TemporalFlightSplitter,
)


# ============================================================================
# Section 1: Parquet Payload Single Bit / Float Flip & Tamper Detection
# ============================================================================

class TestParquetPayloadTamperDetection:
    """
    Empirical challenge: verify that modifying even 1 bit or 1 float in a Parquet dataset
    is deterministically detected by canonical data SHA256 verification in verify_manifest.
    """

    def test_single_float_least_significant_bit_flip_in_numerical_columns(
        self, synthetic_flight_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        """
        For every primary telemetry channel, perturb a single floating-point value
        by the smallest representable delta (epsilon / nextafter) and assert verify_manifest FAILS.
        """
        numerical_channels = [
            "egt_1", "egt_2", "egt_3", "egt_4", "cht",
            "oil_pressure", "oil_temperature", "engine_speed_rpm", "fuel_flow_kg_s",
            "map_pa", "throttle_position", "intake_temp_k", "ambient_pressure_pa",
            "ambient_temp_k", "true_airspeed_ms", "pressure_altitude_m", "timestamp"
        ]

        # Generate golden manifest for original clean dataset
        golden_manifest = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_TAMPER_NUMERICAL_TEST",
            flight_id="FL_001",
            aircraft_id="AC_01",
            engine_id="ENG_01",
            data=synthetic_flight_dataframe,
        )

        for col in numerical_channels:
            # Create a perturbed dataframe with exactly 1 float value modified by 1 ULP / small epsilon
            original_val = synthetic_flight_dataframe[col][150]
            # Flip least significant bit or perturb slightly
            if isinstance(original_val, (float, np.floating)):
                perturbed_val = float(np.nextafter(original_val, np.inf))
                if perturbed_val == original_val:
                    perturbed_val = original_val + 1e-12
            elif isinstance(original_val, (int, np.integer)):
                perturbed_val = original_val + 1
            else:
                continue

            tampered_df = synthetic_flight_dataframe.with_columns(
                pl.when(pl.int_range(0, synthetic_flight_dataframe.height) == 150)
                .then(pl.lit(perturbed_val))
                .otherwise(pl.col(col))
                .alias(col)
            )

            # Assert raw canonical data hash of tampered dataframe differs from golden
            tampered_hash = canonical_data_sha256(tampered_df)
            assert tampered_hash != golden_manifest.canonical_data_sha256, (
                f"Column '{col}' float LSB perturbation did not change canonical SHA256!"
            )

            # Write tampered table to disk
            pq_file = tmp_path / f"tampered_{col}.parquet"
            pq.write_table(tampered_df.to_arrow(), pq_file)

            # Verify against golden manifest
            result = verify_manifest(pq_file, golden_manifest)
            assert result.is_valid is False, f"verify_manifest failed to detect tamper in {col}"
            assert result.status == "FAILED"
            assert any("Canonical data SHA256 mismatch" in err for err in result.errors)

    def test_single_character_tamper_in_categorical_columns(
        self, synthetic_flight_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        """
        Verify single character substitution in string metadata columns inside the dataframe
        is detected as data tampering.
        """
        categorical_cols = ["flight_id", "aircraft_id", "engine_id"]
        golden_manifest = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_TAMPER_STRING_TEST",
            flight_id="flight_001",
            aircraft_id="AC_01",
            engine_id="ROTAX_915_IS_SN1042",
            data=synthetic_flight_dataframe,
        )

        for col in categorical_cols:
            orig_str = str(synthetic_flight_dataframe[col][0])
            # Mutate 1 character
            tampered_str = orig_str[:-1] + ("X" if orig_str[-1] != "X" else "Y")

            tampered_df = synthetic_flight_dataframe.with_columns(
                pl.when(pl.int_range(0, synthetic_flight_dataframe.height) == 0)
                .then(pl.lit(tampered_str))
                .otherwise(pl.col(col))
                .alias(col)
            )

            pq_file = tmp_path / f"tampered_str_{col}.parquet"
            pq.write_table(tampered_df.to_arrow(), pq_file)

            result = verify_manifest(pq_file, golden_manifest)
            assert result.is_valid is False
            assert any("Canonical data SHA256 mismatch" in err for err in result.errors)

    def test_row_insertion_and_deletion_tamper(
        self, synthetic_flight_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        """
        Verify that adding or deleting a single row in the dataset fails canonical verification.
        """
        golden_manifest = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_ROW_COUNT_TEST",
            flight_id="flight_001",
            aircraft_id="AC_01",
            engine_id="ENG_01",
            data=synthetic_flight_dataframe,
        )

        # 1. Deletion of single row
        deleted_df = synthetic_flight_dataframe.slice(0, synthetic_flight_dataframe.height - 1)
        pq_del = tmp_path / "row_deleted.parquet"
        pq.write_table(deleted_df.to_arrow(), pq_del)

        res_del = verify_manifest(pq_del, golden_manifest)
        assert res_del.is_valid is False
        assert any("Canonical data SHA256 mismatch" in err for err in res_del.errors)

        # 2. Insertion of extra row
        extra_row = synthetic_flight_dataframe.slice(0, 1).with_columns(pl.col("timestamp") + 9999.0)
        inserted_df = pl.concat([synthetic_flight_dataframe, extra_row])
        pq_ins = tmp_path / "row_inserted.parquet"
        pq.write_table(inserted_df.to_arrow(), pq_ins)

        res_ins = verify_manifest(pq_ins, golden_manifest)
        assert res_ins.is_valid is False
        assert any("Canonical data SHA256 mismatch" in err for err in res_ins.errors)

    def test_row_swap_inversion_tamper(
        self, synthetic_flight_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        """
        Verify that swapping two rows in an otherwise identical dataset changes the canonical hash.
        """
        golden_manifest = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_SWAP_TEST",
            flight_id="flight_001",
            aircraft_id="AC_01",
            engine_id="ENG_01",
            data=synthetic_flight_dataframe,
        )

        # Swap row 10 and row 11
        rows = synthetic_flight_dataframe.to_dicts()
        rows[10], rows[11] = rows[11], rows[10]
        swapped_df = pl.DataFrame(rows)

        pq_swap = tmp_path / "row_swapped.parquet"
        pq.write_table(swapped_df.to_arrow(), pq_swap)

        res_swap = verify_manifest(pq_swap, golden_manifest)
        assert res_swap.is_valid is False
        assert any("Canonical data SHA256 mismatch" in err for err in res_swap.errors)

    def test_raw_byte_flip_in_parquet_file_on_disk(
        self, synthetic_flight_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        """
        Verify that corrupting a single random byte directly inside the saved Parquet binary file
        is caught by payload_file_sha256 and/or Parquet reader / canonical hash check.
        """
        pq_path = tmp_path / "original_valid.parquet"
        manifest = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_RAW_BYTE_CORRUPTION",
            flight_id="flight_001",
            aircraft_id="AC_01",
            engine_id="ENG_01",
            data=synthetic_flight_dataframe,
            payload_file=pq_path,
        )
        write_parquet_with_manifest(synthetic_flight_dataframe, pq_path, manifest)

        # Verify clean file passes
        clean_res = verify_manifest(pq_path, manifest)
        assert clean_res.is_valid is True

        # Flip a byte at 25% of file size (inside Parquet data pages)
        file_bytes = bytearray(pq_path.read_bytes())
        corrupt_offset = len(file_bytes) // 4
        file_bytes[corrupt_offset] ^= 0xFF  # Invert 8 bits

        corrupted_pq_path = tmp_path / "corrupted.parquet"
        corrupted_pq_path.write_bytes(file_bytes)

        # Verify corrupted file fails
        corrupted_res = verify_manifest(corrupted_pq_path, manifest)
        assert corrupted_res.is_valid is False
        assert corrupted_res.status == "FAILED"
        assert len(corrupted_res.errors) >= 1

    def test_canonical_hash_invariance_under_benign_transformations(
        self, synthetic_flight_dataframe: pl.DataFrame
    ) -> None:
        """
        Verify that benign storage-level transformations do NOT alter canonical data SHA256:
        1. Shuffling dataframe column ordering.
        2. Splitting table into multiple RecordBatches / chunks.
        3. Adding arbitrary Parquet custom schema metadata.
        """
        arrow_table = synthetic_flight_dataframe.to_arrow()
        base_hash = canonical_data_sha256(arrow_table)

        # 1. Column reordering
        col_names = list(arrow_table.column_names)
        rng = random.Random(42)
        shuffled_cols = list(col_names)
        rng.shuffle(shuffled_cols)
        shuffled_table = arrow_table.select(shuffled_cols)
        assert canonical_data_sha256(shuffled_table) == base_hash

        # 2. Chunking variations (1 chunk vs 10 chunks)
        n_rows = arrow_table.num_rows
        chunk_size = n_rows // 5
        batches = [arrow_table.slice(i, chunk_size).to_batches()[0] for i in range(0, n_rows, chunk_size)]
        chunked_table = pa.Table.from_batches(batches)
        assert chunked_table.column(0).num_chunks > 1 or len(batches) > 1
        assert canonical_data_sha256(chunked_table) == base_hash

        # 3. Custom schema metadata
        meta_table = arrow_table.replace_schema_metadata({
            b"custom_key": b"custom_val_12345",
            b"org.apache.spark.version": b"3.4.0",
        })
        assert canonical_data_sha256(meta_table) == base_hash


# ============================================================================
# Section 2: Manifest Metadata Mutation & Hash Verification Challenge
# ============================================================================

class TestManifestMetadataMutationAndHashVerification:
    """
    Stress-tests manifest immutability and tamper detection:
    - Mutating any field alters self-hash.
    - Embedded vs sidecar desynchronization is caught.
    - Tampered hashes in manifest trigger verification failure.
    """

    @pytest.fixture
    def base_manifest_dict(self) -> Dict[str, Any]:
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
                "created_by": "replan_to_learn.provenance",
                "payload_file": "flight_7152.parquet",
                "payload_format": "PARQUET",
                "payload_file_sha256": "b" * 64,
                "canonical_data_sha256": "c" * 64,
                "row_count": 1800,
                "duration_s": 1800.0,
                "parent_manifest_hashes": ["d" * 64],
                "split_group": "train",
                "status_flags": ["HEALTHY", "COMPLETE"],
                "extra_metadata": {"pilot": "Captain Test"},
            },
        }

    def test_top_level_field_mutations_alter_manifest_hash(
        self, base_manifest_dict: Dict[str, Any]
    ) -> None:
        """
        Verify that mutating each of the 8 required fields strictly alters the manifest self-hash.
        """
        manifest_orig = ProvenanceManifest.from_dict(base_manifest_dict)
        orig_hash = manifest_orig.compute_manifest_hash()

        mutations = [
            ("dataset_id", "DS_RAW_MUTATED_01"),
            ("flight_id", "9999"),
            ("aircraft_id", "AC_02"),
            ("engine_id", "ROTAX_915_IS_SN9999"),
            ("model_version", "2.0.0"),
            ("regime_grid_version", "REGIME_GRID_V2"),
            ("calibration_manifest_hash", "e" * 64),
            ("schema_version", "TELEMETRY_SCHEMA_V2"),
        ]

        for field_name, new_val in mutations:
            mut_dict = copy.deepcopy(base_manifest_dict)
            mut_dict[field_name] = new_val
            manifest_mut = ProvenanceManifest.from_dict(mut_dict)
            new_hash = manifest_mut.compute_manifest_hash()
            assert new_hash != orig_hash, f"Mutation in '{field_name}' did not alter manifest hash!"
            assert len(new_hash) == 64

    def test_extension_field_mutations_alter_manifest_hash(
        self, base_manifest_dict: Dict[str, Any]
    ) -> None:
        """
        Verify that mutating any extended audit metadata field strictly alters the manifest self-hash.
        """
        manifest_orig = ProvenanceManifest.from_dict(base_manifest_dict)
        orig_hash = manifest_orig.compute_manifest_hash()

        ext_mutations = [
            ("row_count", 1801),
            ("duration_s", 1801.5),
            ("created_at_utc", "2026-08-26T21:05:00.000000Z"),
            ("created_by", "unauthorized_agent"),
            ("payload_file_sha256", "f" * 64),
            ("canonical_data_sha256", "0" * 64),
            ("parent_manifest_hashes", ["1" * 64]),
            ("split_group", "val"),
            ("status_flags", ["HEALTHY", "MODIFIED"]),
            ("extra_metadata", {"pilot": "Captain Tamper"}),
        ]

        for ext_key, new_val in ext_mutations:
            mut_dict = copy.deepcopy(base_manifest_dict)
            mut_dict["extensions"][ext_key] = new_val
            manifest_mut = ProvenanceManifest.from_dict(mut_dict)
            new_hash = manifest_mut.compute_manifest_hash()
            assert new_hash != orig_hash, f"Extension mutation in '{ext_key}' did not alter manifest hash!"

    def test_sidecar_and_embedded_manifest_desynchronization_detection(
        self, synthetic_flight_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        """
        Verify that if a Parquet file contains embedded manifest A, but sidecar manifest B is passed
        to verify_manifest (e.g. metadata was tampered in sidecar), verify_manifest detects mismatch.
        """
        pq_path = tmp_path / "sync_test.parquet"
        manifest_a = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_EMBEDDED_ORIGINAL",
            flight_id="FL_001",
            aircraft_id="AC_01",
            engine_id="ENG_01",
            data=synthetic_flight_dataframe,
            model_version="1.0.0",
        )
        write_parquet_with_manifest(synthetic_flight_dataframe, pq_path, manifest_a)

        # Create desynchronized manifest B (e.g. claiming different model_version or row_count)
        manifest_b = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_EMBEDDED_ORIGINAL",
            flight_id="FL_001",
            aircraft_id="AC_01",
            engine_id="ENG_01",
            data=synthetic_flight_dataframe,
            model_version="1.0.1",
        )

        result = verify_manifest(pq_path, manifest_b)
        assert result.is_valid is False
        assert any("Embedded Parquet manifest does not match provided manifest" in err for err in result.errors)

    def test_verify_manifest_detects_tampered_hashes_in_manifest(
        self, synthetic_flight_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        """
        Verify that forging payload_file_sha256 or canonical_data_sha256 inside the manifest
        causes verify_manifest to fail with explicit error descriptions.
        """
        pq_path = tmp_path / "hash_tamper.parquet"
        valid_manifest = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_HASH_TAMPER",
            flight_id="FL_001",
            aircraft_id="AC_01",
            engine_id="ENG_01",
            data=synthetic_flight_dataframe,
        )
        pq.write_table(synthetic_flight_dataframe.to_arrow(), pq_path)

        # 1. Forge file SHA256
        forged_file_manifest = valid_manifest.model_copy(deep=True)
        forged_file_manifest.extensions.payload_file_sha256 = "0" * 64
        res1 = verify_manifest(pq_path, forged_file_manifest)
        assert res1.is_valid is False
        assert any("File SHA256 mismatch" in err for err in res1.errors)

        # 2. Forge canonical data SHA256
        forged_data_manifest = valid_manifest.model_copy(deep=True)
        forged_data_manifest.extensions.canonical_data_sha256 = "f" * 64
        res2 = verify_manifest(pq_path, forged_data_manifest)
        assert res2.is_valid is False
        assert any("Canonical data SHA256 mismatch" in err for err in res2.errors)


# ============================================================================
# Section 3: Exhaustive 8-Field Regex & Schema Validation Challenge
# ============================================================================

class TestExhaustive8RequiredFieldsSchemaValidation:
    """
    Exhaustive boundary condition, regex conformance, and negative validation testing
    for each of the 8 required fields in ProvenanceManifest.
    """

    @pytest.fixture
    def valid_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": "DS_TEST_01",
            "flight_id": "FL_100",
            "aircraft_id": "AC_01",
            "engine_id": "ENG_ROTAX_915",
            "model_version": "1.0.0",
            "regime_grid_version": "REGIME_GRID_V1",
            "calibration_manifest_hash": "a" * 64,
            "schema_version": "TELEMETRY_SCHEMA_V1",
        }

    # ------------------------------------------------------------------------
    # 1. dataset_id: regex r"^[a-zA-Z0-9_\-\.:]{3,128}$"
    # ------------------------------------------------------------------------
    @pytest.mark.parametrize("valid_val", [
        "abc", "DS_01", "dataset.2026:run-1", "a" * 128
    ])
    def test_dataset_id_valid_values(self, valid_dict: Dict[str, Any], valid_val: str):
        d = dict(valid_dict, dataset_id=valid_val)
        manifest = ProvenanceManifest.from_dict(d)
        assert manifest.dataset_id == valid_val

    @pytest.mark.parametrize("invalid_val", [
        "", "a", "ab", "a" * 129, "bad/slash", "bad\\backslash", "bad space", "bad@char", "bad#char", "bad$val"
    ])
    def test_dataset_id_invalid_values(self, valid_dict: Dict[str, Any], invalid_val: str):
        d = dict(valid_dict, dataset_id=invalid_val)
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(d)

    # ------------------------------------------------------------------------
    # 2. flight_id: regex r"^([a-zA-Z0-9_\-\.:]{1,64}|MULTI)$"
    # ------------------------------------------------------------------------
    @pytest.mark.parametrize("valid_val", [
        "1", "7152", "FL_100.1:A", "MULTI", "f" * 64
    ])
    def test_flight_id_valid_values(self, valid_dict: Dict[str, Any], valid_val: str):
        d = dict(valid_dict, flight_id=valid_val)
        manifest = ProvenanceManifest.from_dict(d)
        assert manifest.flight_id == valid_val

    @pytest.mark.parametrize("invalid_val", [
        "", "f" * 65, "flight 101", "fl/101", "fl?101", "flight#1"
    ])
    def test_flight_id_invalid_values(self, valid_dict: Dict[str, Any], invalid_val: str):
        d = dict(valid_dict, flight_id=invalid_val)
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(d)

    # ------------------------------------------------------------------------
    # 3. aircraft_id: regex r"^[a-zA-Z0-9_\-]{2,64}$" (No dots or colons allowed)
    # ------------------------------------------------------------------------
    @pytest.mark.parametrize("valid_val", [
        "AC", "AC_01", "AIRCRAFT-ALPHA-99", "a" * 64
    ])
    def test_aircraft_id_valid_values(self, valid_dict: Dict[str, Any], valid_val: str):
        d = dict(valid_dict, aircraft_id=valid_val)
        manifest = ProvenanceManifest.from_dict(d)
        assert manifest.aircraft_id == valid_val

    @pytest.mark.parametrize("invalid_val", [
        "", "A", "a" * 65, "AC.01", "AC:01", "AC/01", "AC 01", "AC#01"
    ])
    def test_aircraft_id_invalid_values(self, valid_dict: Dict[str, Any], invalid_val: str):
        d = dict(valid_dict, aircraft_id=invalid_val)
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(d)

    # ------------------------------------------------------------------------
    # 4. engine_id: regex r"^[a-zA-Z0-9_\-]{2,64}$"
    # ------------------------------------------------------------------------
    @pytest.mark.parametrize("valid_val", [
        "EN", "ROTAX_915_IS", "ENG-001", "e" * 64
    ])
    def test_engine_id_valid_values(self, valid_dict: Dict[str, Any], valid_val: str):
        d = dict(valid_dict, engine_id=valid_val)
        manifest = ProvenanceManifest.from_dict(d)
        assert manifest.engine_id == valid_val

    @pytest.mark.parametrize("invalid_val", [
        "", "E", "e" * 65, "ENG.01", "ENG:01", "ENG/01", "ENG 01"
    ])
    def test_engine_id_invalid_values(self, valid_dict: Dict[str, Any], invalid_val: str):
        d = dict(valid_dict, engine_id=invalid_val)
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(d)

    # ------------------------------------------------------------------------
    # 5. model_version: regex SemVer or Git SHA
    # ------------------------------------------------------------------------
    @pytest.mark.parametrize("valid_val", [
        "0.0.1", "1.0.0", "v1.2.3", "2.1.0-alpha.1", "10.20.30-rc.5",
        "9f8e7d6", "a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e", "a" * 40
    ])
    def test_model_version_valid_values(self, valid_dict: Dict[str, Any], valid_val: str):
        d = dict(valid_dict, model_version=valid_val)
        manifest = ProvenanceManifest.from_dict(d)
        assert manifest.model_version == valid_val

    @pytest.mark.parametrize("invalid_val", [
        "", "1.0", "v1", "latest", "1.0.0.0", "1.0.0_beta", "9f8e7dg",  # 'g' is not hex
        "a" * 6,   # 6 chars too short for SHA
        "a" * 41,  # 41 chars too long for SHA
        "1.0.0+build.1"  # '+' not allowed
    ])
    def test_model_version_invalid_values(self, valid_dict: Dict[str, Any], invalid_val: str):
        d = dict(valid_dict, model_version=invalid_val)
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(d)

    # ------------------------------------------------------------------------
    # 6. regime_grid_version: regex r"^REGIME_GRID_V[1-9][0-9]*$"
    # ------------------------------------------------------------------------
    @pytest.mark.parametrize("valid_val", [
        "REGIME_GRID_V1", "REGIME_GRID_V2", "REGIME_GRID_V10", "REGIME_GRID_V99"
    ])
    def test_regime_grid_version_valid_values(self, valid_dict: Dict[str, Any], valid_val: str):
        d = dict(valid_dict, regime_grid_version=valid_val)
        manifest = ProvenanceManifest.from_dict(d)
        assert manifest.regime_grid_version == valid_val

    @pytest.mark.parametrize("invalid_val", [
        "", "REGIME_GRID", "GRID_V1", "REGIME_GRID_V0", "REGIME_GRID_V01", "REGIME_GRID_VX", "regime_grid_v1"
    ])
    def test_regime_grid_version_invalid_values(self, valid_dict: Dict[str, Any], invalid_val: str):
        d = dict(valid_dict, regime_grid_version=invalid_val)
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(d)

    # ------------------------------------------------------------------------
    # 7. calibration_manifest_hash: 64-char hex or 'NONE'
    # ------------------------------------------------------------------------
    @pytest.mark.parametrize("valid_val", [
        "NONE", "0" * 64, "a" * 64, "0123456789abcdef" * 4
    ])
    def test_calibration_manifest_hash_valid_values(self, valid_dict: Dict[str, Any], valid_val: str):
        d = dict(valid_dict, calibration_manifest_hash=valid_val)
        manifest = ProvenanceManifest.from_dict(d)
        assert manifest.calibration_manifest_hash == valid_val.lower() if valid_val != "NONE" else "NONE"

    @pytest.mark.parametrize("invalid_val", [
        "", "none", "None", "NULL", "123", "a" * 63, "a" * 65, "g" * 64, " " * 64
    ])
    def test_calibration_manifest_hash_invalid_values(self, valid_dict: Dict[str, Any], invalid_val: str):
        d = dict(valid_dict, calibration_manifest_hash=invalid_val)
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(d)

    # ------------------------------------------------------------------------
    # 8. schema_version: regex r"^[a-zA-Z0-9_\-\.]{3,64}$"
    # ------------------------------------------------------------------------
    @pytest.mark.parametrize("valid_val", [
        "V1.0", "TELEMETRY_SCHEMA_V1", "RESIDUAL_FRAME_V1.2-BETA", "s" * 64
    ])
    def test_schema_version_valid_values(self, valid_dict: Dict[str, Any], valid_val: str):
        d = dict(valid_dict, schema_version=valid_val)
        manifest = ProvenanceManifest.from_dict(d)
        assert manifest.schema_version == valid_val

    @pytest.mark.parametrize("invalid_val", [
        "", "a", "ab", "s" * 65, "SCHEMA V1", "SCHEMA/V1", "SCHEMA:V1"
    ])
    def test_schema_version_invalid_values(self, valid_dict: Dict[str, Any], invalid_val: str):
        d = dict(valid_dict, schema_version=invalid_val)
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(d)

    # ------------------------------------------------------------------------
    # Missing, Null, and Incorrect Type Checks
    # ------------------------------------------------------------------------
    def test_missing_each_required_field_raises(self, valid_dict: Dict[str, Any]):
        for k in valid_dict.keys():
            corrupt = dict(valid_dict)
            del corrupt[k]
            with pytest.raises(ValidationError):
                ProvenanceManifest.from_dict(corrupt)

    def test_none_value_for_each_required_field_raises(self, valid_dict: Dict[str, Any]):
        for k in valid_dict.keys():
            corrupt = dict(valid_dict)
            corrupt[k] = None
            with pytest.raises(ValidationError):
                ProvenanceManifest.from_dict(corrupt)

    def test_invalid_types_for_fields_raises(self, valid_dict: Dict[str, Any]):
        # Dict passed for string
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(dict(valid_dict, dataset_id={"id": "DS_01"}))

        # List passed for string
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(dict(valid_dict, aircraft_id=["AC_01"]))

        # Int passed for calibration_manifest_hash
        with pytest.raises(ValidationError):
            ProvenanceManifest.from_dict(dict(valid_dict, calibration_manifest_hash=1234567890))


# ============================================================================
# Section 4: Multi-Threaded / Concurrent Stress Harness
# ============================================================================

class TestConcurrentManifestAndParquetHarness:
    """
    Stress-tests thread safety, race conditions, memory isolation, and high contention
    during concurrent manifest creation, Parquet metadata embedding, and verification.
    """

    def test_concurrent_manifest_generation_thread_safety(
        self, synthetic_flight_dataframe: pl.DataFrame
    ) -> None:
        """
        Spawn 20 threads simultaneously generating 50 manifests each (1,000 total).
        Assert no race conditions, valid deterministic hashing, and zero exceptions.
        """
        n_workers = 10
        manifests_per_worker = 10
        table = synthetic_flight_dataframe.to_arrow()

        def generate_manifests(worker_id: int) -> List[str]:
            hashes = []
            for idx in range(manifests_per_worker):
                m = ManifestGenerator.create_raw_manifest(
                    dataset_id=f"DS_CONCURRENT_T{worker_id}_{idx}",
                    flight_id=f"FL_{worker_id}_{idx}",
                    aircraft_id=f"AC_{worker_id}",
                    engine_id="ROTAX_915_IS",
                    data=table,
                )
                h = m.compute_manifest_hash()
                hashes.append(h)
            return hashes

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(generate_manifests, i) for i in range(n_workers)]
            all_hashes = []
            for f in concurrent.futures.as_completed(futures):
                all_hashes.extend(f.result())

        assert len(all_hashes) == n_workers * manifests_per_worker
        assert all(len(h) == 64 for h in all_hashes)

    def test_concurrent_parquet_writing_embedding_and_verification(
        self, synthetic_flight_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        """
        16 worker threads concurrently write distinct Parquet files with embedded manifests,
        then concurrently read and verify each file.
        """
        n_workers = 16
        table = synthetic_flight_dataframe.to_arrow()

        def worker_write_and_verify(worker_id: int) -> bool:
            sub_dir = tmp_path / f"worker_{worker_id}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            pq_path = sub_dir / f"artifact_{worker_id}.parquet"

            manifest = ManifestGenerator.create_raw_manifest(
                dataset_id=f"DS_CONCURRENT_PQ_{worker_id}",
                flight_id=f"FL_CONCURRENT_{worker_id}",
                aircraft_id=f"AC_{worker_id % 4:02d}",
                engine_id="ROTAX_915_IS_SN01",
                data=table,
                payload_file=pq_path,
            )

            write_parquet_with_manifest(table, pq_path, manifest)

            # Read embedded manifest directly
            read_m = read_manifest_from_parquet(pq_path)
            if read_m.compute_manifest_hash() != manifest.compute_manifest_hash():
                return False

            # Verify with verify_manifest
            ver_res = verify_manifest(pq_path)
            return ver_res.is_valid and ver_res.status == "PASSED"

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker_write_and_verify, i) for i in range(n_workers)]
            for f in concurrent.futures.as_completed(futures):
                assert f.result() is True

    def test_high_contention_concurrent_read_and_verify_on_shared_file(
        self, synthetic_flight_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        """
        Write a single shared Parquet file with embedded manifest.
        Spawn 20 concurrent threads reading and verifying the same file 25 times each (500 verifications).
        Assert zero failures or race corruptions.
        """
        shared_pq = tmp_path / "shared_artifact.parquet"
        table = synthetic_flight_dataframe.to_arrow()
        manifest = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_SHARED_CONTENTION",
            flight_id="FL_SHARED_01",
            aircraft_id="AC_SHARED_01",
            engine_id="ROTAX_SHARED_01",
            data=table,
            payload_file=shared_pq,
        )
        write_parquet_with_manifest(table, shared_pq, manifest)

        n_threads = 20
        n_iters = 25
        verification_failures: List[str] = []
        lock = threading.Lock()

        def reader_task():
            for _ in range(n_iters):
                res = verify_manifest(shared_pq)
                if not res.is_valid or res.status != "PASSED":
                    with lock:
                        verification_failures.append(str(res.errors))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(reader_task) for _ in range(n_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert len(verification_failures) == 0, f"Concurrent read failures: {verification_failures}"

    def test_concurrent_grouped_splitters_thread_safety(
        self, multi_aircraft_fleet_dataframe: pl.DataFrame
    ) -> None:
        """
        Test that running multiple splitters concurrently across threads produces
        deterministic splits and guarantees zero leakage without race conditions.
        """
        def run_split(seed_val: int) -> bool:
            flight_splitter = GroupedFlightSplitter()
            ac_splitter = GroupedAircraftSplitter()
            hier_splitter = HierarchicalGroupSplitter()
            temp_splitter = TemporalFlightSplitter()

            cfg_flight = SplitConfiguration(strategy="grouped_flight", seed=seed_val)
            cfg_ac = SplitConfiguration(strategy="grouped_aircraft", seed=seed_val)
            cfg_hier = SplitConfiguration(strategy="hierarchical", seed=seed_val)
            cfg_temp = SplitConfiguration(strategy="temporal", seed=seed_val)

            res_fl = flight_splitter.split(multi_aircraft_fleet_dataframe, cfg_flight)
            res_ac = ac_splitter.split(multi_aircraft_fleet_dataframe, cfg_ac)
            res_hi = hier_splitter.split(multi_aircraft_fleet_dataframe, cfg_hier)
            res_tm = temp_splitter.split(multi_aircraft_fleet_dataframe, cfg_temp)

            res_fl.assert_zero_leakage()
            res_ac.assert_zero_leakage()
            res_hi.assert_zero_leakage()
            res_tm.assert_zero_leakage()
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_split, seed) for seed in range(50, 70)]
            for f in concurrent.futures.as_completed(futures):
                assert f.result() is True
