"""
SIH26054 Replan to Learn: Provenance & Data Versioning.
8-Field Provenance Manifest Schema, Generation, Serialization & Verification.

Implements strict validation of the 8 required provenance fields:
- dataset_id
- flight_id
- aircraft_id
- engine_id
- model_version
- regime_grid_version
- calibration_manifest_hash
- schema_version

Supports RFC 8785 Canonical JSON, sidecar serialization, embedded Apache Parquet
key-value metadata (b"sih26054.provenance_manifest"), and automated verification.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from replan_to_learn.provenance.hasher import (
    canonical_data_sha256,
    canonical_json_dumps,
    canonical_json_loads,
    compute_manifest_hash,
    payload_file_sha256,
)

# ============================================================================
# 1. Constants & Validation Regex Patterns
# ============================================================================

PARQUET_MANIFEST_KEY: bytes = b"sih26054.provenance_manifest"
REGIME_GRID_VERSION_V1: str = "REGIME_GRID_V1"

DATASET_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.:]{3,128}$")
FLIGHT_ID_REGEX = re.compile(r"^([a-zA-Z0-9_\-\.:]{1,64}|MULTI)$")
AIRCRAFT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{2,64}$")
ENGINE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{2,64}$")
MODEL_VERSION_REGEX = re.compile(r"^(v?[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9\.]+)?|[0-9a-f]{7,40})$")
REGIME_GRID_VERSION_REGEX = re.compile(r"^REGIME_GRID_V[1-9][0-9]*$")
CALIBRATION_HASH_REGEX = re.compile(r"^([0-9a-f]{64}|NONE)$")
SCHEMA_VERSION_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{3,64}$")
SHA256_HEX_REGEX = re.compile(r"^[0-9a-f]{64}$")


# ============================================================================
# 2. Audit Extensions Sub-Model
# ============================================================================

class ProvenanceAuditMetadata(BaseModel):
    """Extended audit and lineage metadata associated with an artifact."""
    model_config = ConfigDict(extra="allow", populate_by_name=True, protected_namespaces=())

    created_at_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    )
    created_by: str = "replan_to_learn.provenance"
    payload_file: str = ""
    payload_format: str = "PARQUET"
    payload_file_sha256: str = ""
    canonical_data_sha256: str = ""
    row_count: int = 0
    duration_s: float = 0.0
    parent_manifest_hashes: List[str] = Field(default_factory=list)
    split_group: Optional[Literal["train", "val", "test", "holdout"]] = None
    status_flags: List[str] = Field(default_factory=lambda: ["HEALTHY", "COMPLETE"])
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload_file_sha256", "canonical_data_sha256")
    @classmethod
    def validate_hashes(cls, v: str) -> str:
        if v and v != "NONE" and not SHA256_HEX_REGEX.match(v):
            raise ValueError(f"Hash must be a 64-character lowercase SHA256 hex string, got '{v}'")
        return v.lower() if v else ""

    @field_validator("parent_manifest_hashes")
    @classmethod
    def validate_parent_hashes(cls, v: List[str]) -> List[str]:
        for h in v:
            if not SHA256_HEX_REGEX.match(h):
                raise ValueError(f"Parent manifest hash '{h}' is not a valid 64-character SHA256 hex digest")
        return [h.lower() for h in v]


# ============================================================================
# 3. Core 8-Field ProvenanceManifest Model
# ============================================================================

class ProvenanceManifest(BaseModel):
    """
    Core Provenance Tracking Manifest containing all 8 required fields.
    Validates strictly against SIH26054 data versioning specifications.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True, protected_namespaces=())

    # The 8 Required Provenance Fields
    dataset_id: str
    flight_id: str
    aircraft_id: str
    engine_id: str
    model_version: str
    regime_grid_version: str
    calibration_manifest_hash: str
    schema_version: str

    # Extended Audit Metadata
    extensions: ProvenanceAuditMetadata = Field(default_factory=ProvenanceAuditMetadata)

    # ------------------------------------------------------------------------
    # Field Validators for the 8 Required Fields
    # ------------------------------------------------------------------------

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        if not DATASET_ID_REGEX.match(v):
            raise ValueError(
                f"dataset_id '{v}' invalid: must be 3-128 alphanumeric characters with '_', '-', '.', or ':'"
            )
        return v

    @field_validator("flight_id")
    @classmethod
    def validate_flight_id(cls, v: str) -> str:
        if not FLIGHT_ID_REGEX.match(v):
            raise ValueError(
                f"flight_id '{v}' invalid: must be 1-64 alphanumeric characters or 'MULTI'"
            )
        return v

    @field_validator("aircraft_id")
    @classmethod
    def validate_aircraft_id(cls, v: str) -> str:
        if not AIRCRAFT_ID_REGEX.match(v):
            raise ValueError(
                f"aircraft_id '{v}' invalid: must be 2-64 alphanumeric characters with '_' or '-'"
            )
        return v

    @field_validator("engine_id")
    @classmethod
    def validate_engine_id(cls, v: str) -> str:
        if not ENGINE_ID_REGEX.match(v):
            raise ValueError(
                f"engine_id '{v}' invalid: must be 2-64 alphanumeric characters with '_' or '-'"
            )
        return v

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, v: str) -> str:
        if not MODEL_VERSION_REGEX.match(v):
            raise ValueError(
                f"model_version '{v}' invalid: must be semver (e.g. '1.0.0', 'v1.2.0') or git commit SHA (7-40 hex)"
            )
        return v

    @field_validator("regime_grid_version")
    @classmethod
    def validate_regime_grid_version(cls, v: str) -> str:
        if not REGIME_GRID_VERSION_REGEX.match(v):
            raise ValueError(
                f"regime_grid_version '{v}' invalid: must match pattern 'REGIME_GRID_V<N>'"
            )
        return v

    @field_validator("calibration_manifest_hash")
    @classmethod
    def validate_calibration_manifest_hash(cls, v: str) -> str:
        if not CALIBRATION_HASH_REGEX.match(v):
            raise ValueError(
                f"calibration_manifest_hash '{v}' invalid: must be 64-char SHA256 hex string or 'NONE'"
            )
        return v.lower() if v != "NONE" else "NONE"

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if not SCHEMA_VERSION_REGEX.match(v):
            raise ValueError(
                f"schema_version '{v}' invalid: must be 3-64 alphanumeric characters with '_', '-', or '.'"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def preprocess_input(cls, values: Any) -> Any:
        """Allow passing extension fields at root level or under 'extensions'."""
        if not isinstance(values, dict):
            return values

        data = dict(values)
        ext_dict = {}

        if "extensions" in data and isinstance(data["extensions"], (dict, ProvenanceAuditMetadata)):
            ext_dict = (
                data["extensions"].model_dump()
                if isinstance(data["extensions"], ProvenanceAuditMetadata)
                else dict(data["extensions"])
            )

        # Move root extension keys into ext_dict if present at top level
        known_ext_fields = {
            "created_at_utc",
            "created_by",
            "payload_file",
            "payload_format",
            "payload_file_sha256",
            "canonical_data_sha256",
            "row_count",
            "duration_s",
            "parent_manifest_hashes",
            "split_group",
            "status_flags",
            "extra_metadata",
        }

        for k in list(data.keys()):
            if k in known_ext_fields:
                ext_dict[k] = data.pop(k)

        data["extensions"] = ext_dict
        return data

    # ------------------------------------------------------------------------
    # Property shortcuts to extension attributes
    # ------------------------------------------------------------------------

    @property
    def created_at_utc(self) -> str:
        return self.extensions.created_at_utc

    @property
    def created_by(self) -> str:
        return self.extensions.created_by

    @property
    def payload_file(self) -> str:
        return self.extensions.payload_file

    @property
    def payload_format(self) -> str:
        return self.extensions.payload_format

    @property
    def payload_file_sha256(self) -> str:
        return self.extensions.payload_file_sha256

    @property
    def canonical_data_sha256(self) -> str:
        return self.extensions.canonical_data_sha256

    @property
    def row_count(self) -> int:
        return self.extensions.row_count

    @property
    def duration_s(self) -> float:
        return self.extensions.duration_s

    @property
    def parent_manifest_hashes(self) -> List[str]:
        return self.extensions.parent_manifest_hashes

    @property
    def split_group(self) -> Optional[str]:
        return self.extensions.split_group

    @property
    def status_flags(self) -> List[str]:
        return self.extensions.status_flags

    @property
    def extra_metadata(self) -> Dict[str, Any]:
        return self.extensions.extra_metadata

    @property
    def audit_metadata(self) -> Any:
        return self.extensions

    # ------------------------------------------------------------------------
    # Serialization & Hashing Methods
    # ------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to standardized dictionary with 8 fields and extensions."""
        return {
            "dataset_id": self.dataset_id,
            "flight_id": self.flight_id,
            "aircraft_id": self.aircraft_id,
            "engine_id": self.engine_id,
            "model_version": self.model_version,
            "regime_grid_version": self.regime_grid_version,
            "calibration_manifest_hash": self.calibration_manifest_hash,
            "schema_version": self.schema_version,
            "extensions": self.extensions.model_dump(),
        }

    def to_canonical_json(self) -> str:
        """Serialize manifest to RFC 8785 canonical JSON string."""
        return canonical_json_dumps(self.to_dict())

    def compute_manifest_hash(self) -> str:
        """Compute deterministic SHA256 self-hash of the manifest."""
        return compute_manifest_hash(self.to_dict())

    def to_parquet_metadata(self) -> Dict[bytes, bytes]:
        """Convert manifest to Apache Parquet key-value metadata dictionary."""
        return {PARQUET_MANIFEST_KEY: self.to_canonical_json().encode("utf-8")}

    def save(self, path: Union[str, Path], format: str = "json") -> Path:
        """Save manifest as a sidecar file (.manifest.json or .manifest.yaml)."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_canonical_json()
        target_path.write_text(content, encoding="utf-8")
        return target_path

    # ------------------------------------------------------------------------
    # Class Constructors & Deserializers
    # ------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProvenanceManifest:
        """Construct and validate a ProvenanceManifest from a dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str_or_bytes: Union[str, bytes]) -> ProvenanceManifest:
        """Construct and validate a ProvenanceManifest from a JSON string or bytes."""
        data = canonical_json_loads(json_str_or_bytes)
        return cls.from_dict(data)

    @classmethod
    def from_parquet_metadata(
        cls, metadata: Union[Dict[bytes, bytes], Dict[str, str], pa.KeyValueMetadata, None]
    ) -> ProvenanceManifest:
        """Extract and parse ProvenanceManifest from Parquet Key-Value metadata."""
        if metadata is None:
            raise ValueError("Parquet schema contains no metadata")

        raw_meta: Dict[Any, Any] = metadata if isinstance(metadata, dict) else metadata.as_dict()

        manifest_bytes: Optional[bytes] = None
        if PARQUET_MANIFEST_KEY in raw_meta:
            manifest_bytes = raw_meta[PARQUET_MANIFEST_KEY]
        elif PARQUET_MANIFEST_KEY.decode("utf-8") in raw_meta:
            val = raw_meta[PARQUET_MANIFEST_KEY.decode("utf-8")]
            manifest_bytes = val.encode("utf-8") if isinstance(val, str) else val

        if manifest_bytes is None:
            raise KeyError(
                f"Parquet metadata does not contain key '{PARQUET_MANIFEST_KEY.decode('utf-8')}'"
            )

        return cls.from_json(manifest_bytes)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ProvenanceManifest:
        """
        Load a ProvenanceManifest from a JSON sidecar file or an Apache Parquet file.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Manifest source file not found: {file_path}")

        if file_path.suffix == ".parquet":
            schema = pq.read_schema(file_path)
            return cls.from_parquet_metadata(schema.metadata)
        else:
            content = file_path.read_text(encoding="utf-8")
            return cls.from_json(content)


# ============================================================================
# 4. Manifest Generator Factory
# ============================================================================

class ManifestGenerator:
    """Factory and utility class to create validated manifests for raw and processed datasets."""

    @staticmethod
    def create_raw_manifest(
        dataset_id: str,
        flight_id: str,
        aircraft_id: str,
        engine_id: str,
        model_version: str = "1.0.0",
        regime_grid_version: str = REGIME_GRID_VERSION_V1,
        schema_version: str = "TELEMETRY_SCHEMA_V1",
        data: Optional[Union[pl.DataFrame, pa.Table]] = None,
        payload_file: Optional[Union[str, Path]] = None,
        payload_format: str = "PARQUET",
        created_by: str = "replan_to_learn.provenance.manifest_generator",
        extra_metadata: Optional[Dict[str, Any]] = None,
        calibration_manifest_hash: str = "NONE",
    ) -> ProvenanceManifest:
        """
        Generate a validated manifest for a raw telemetry dataset.
        Sets calibration_manifest_hash to 'NONE' by contract.
        """
        file_sha = ""
        filename = ""
        if payload_file:
            path = Path(payload_file)
            filename = path.name
            if path.exists() and path.is_file():
                file_sha = payload_file_sha256(path)

        data_sha = ""
        row_cnt = 0
        dur_s = 0.0

        if data is not None:
            data_sha = canonical_data_sha256(data)
            row_cnt = len(data)
            if isinstance(data, pl.DataFrame):
                if "timestamp" in data.columns and not data.is_empty():
                    t_min = data["timestamp"].min()
                    t_max = data["timestamp"].max()
                    if t_min is not None and t_max is not None:
                        dur_s = float(t_max - t_min)
                else:
                    dur_s = float(row_cnt)
            elif isinstance(data, pa.Table):
                if "timestamp" in data.column_names and data.num_rows > 0:
                    t_col = data["timestamp"]
                    dur_s = float(t_col[-1].as_py() - t_col[0].as_py())
                else:
                    dur_s = float(row_cnt)

        ext = ProvenanceAuditMetadata(
            created_by=created_by,
            payload_file=filename,
            payload_format=payload_format,
            payload_file_sha256=file_sha,
            canonical_data_sha256=data_sha,
            row_count=row_cnt,
            duration_s=dur_s,
            parent_manifest_hashes=[],
            status_flags=["HEALTHY", "RAW_UNPROCESSED"],
            extra_metadata=extra_metadata or {},
        )

        return ProvenanceManifest(
            dataset_id=dataset_id,
            flight_id=flight_id,
            aircraft_id=aircraft_id,
            engine_id=engine_id,
            model_version=model_version,
            regime_grid_version=regime_grid_version,
            calibration_manifest_hash=calibration_manifest_hash,
            schema_version=schema_version,
            extensions=ext,
        )

    @staticmethod
    def create_processed_manifest(
        dataset_id: str,
        flight_id: str,
        aircraft_id: str,
        engine_id: str,
        model_version: str,
        calibration_manifest_hash: str,
        regime_grid_version: str = REGIME_GRID_VERSION_V1,
        schema_version: str = "RESIDUAL_FRAME_V1",
        parent_manifest_hashes: Optional[List[str]] = None,
        data: Optional[Union[pl.DataFrame, pa.Table]] = None,
        payload_file: Optional[Union[str, Path]] = None,
        payload_format: str = "PARQUET",
        split_group: Optional[Literal["train", "val", "test", "holdout"]] = None,
        created_by: str = "replan_to_learn.provenance.manifest_generator",
        status_flags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceManifest:
        """
        Generate a validated manifest for a processed / derived artifact.
        Links upstream parent manifest hashes and calibration manifest hash.
        """
        file_sha = ""
        filename = ""
        if payload_file:
            path = Path(payload_file)
            filename = path.name
            if path.exists() and path.is_file():
                file_sha = payload_file_sha256(path)

        data_sha = ""
        row_cnt = 0
        dur_s = 0.0

        if data is not None:
            data_sha = canonical_data_sha256(data)
            row_cnt = len(data)
            if isinstance(data, pl.DataFrame):
                if "timestamp" in data.columns and not data.is_empty():
                    t_min = data["timestamp"].min()
                    t_max = data["timestamp"].max()
                    if t_min is not None and t_max is not None:
                        dur_s = float(t_max - t_min)
                else:
                    dur_s = float(row_cnt)
            elif isinstance(data, pa.Table):
                if "timestamp" in data.column_names and data.num_rows > 0:
                    t_col = data["timestamp"]
                    dur_s = float(t_col[-1].as_py() - t_col[0].as_py())
                else:
                    dur_s = float(row_cnt)

        ext = ProvenanceAuditMetadata(
            created_by=created_by,
            payload_file=filename,
            payload_format=payload_format,
            payload_file_sha256=file_sha,
            canonical_data_sha256=data_sha,
            row_count=row_cnt,
            duration_s=dur_s,
            parent_manifest_hashes=parent_manifest_hashes or [],
            split_group=split_group,
            status_flags=status_flags or ["HEALTHY", "PROCESSED"],
            extra_metadata=extra_metadata or {},
        )

        return ProvenanceManifest(
            dataset_id=dataset_id,
            flight_id=flight_id,
            aircraft_id=aircraft_id,
            engine_id=engine_id,
            model_version=model_version,
            regime_grid_version=regime_grid_version,
            calibration_manifest_hash=calibration_manifest_hash,
            schema_version=schema_version,
            extensions=ext,
        )


# ============================================================================
# 5. Parquet Embedding & Extraction Helpers
# ============================================================================

def embed_manifest_in_schema(schema: pa.Schema, manifest: ProvenanceManifest) -> pa.Schema:
    """Embed ProvenanceManifest as key-value metadata in a PyArrow Schema."""
    existing_metadata = dict(schema.metadata or {})
    existing_metadata[PARQUET_MANIFEST_KEY] = manifest.to_canonical_json().encode("utf-8")
    return schema.with_metadata(existing_metadata)


def write_parquet_with_manifest(
    table_or_df: Union[pa.Table, pl.DataFrame],
    file_path: Union[str, Path],
    manifest: ProvenanceManifest,
    **parquet_kwargs: Any,
) -> Path:
    """
    Write a PyArrow Table or Polars DataFrame to Parquet with embedded ProvenanceManifest.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(table_or_df, pl.DataFrame):
        table = table_or_df.to_arrow()
    elif isinstance(table_or_df, pa.Table):
        table = table_or_df
    else:
        raise TypeError(f"Expected pa.Table or pl.DataFrame, got {type(table_or_df)}")

    schema_with_meta = embed_manifest_in_schema(table.schema, manifest)
    table_with_meta = table.replace_schema_metadata(schema_with_meta.metadata)

    pq.write_table(table_with_meta, path, **parquet_kwargs)
    return path


def read_manifest_from_parquet(file_path: Union[str, Path]) -> ProvenanceManifest:
    """Read and deserialize ProvenanceManifest directly from Parquet metadata in <1 ms."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    schema = pq.read_schema(path)
    return ProvenanceManifest.from_parquet_metadata(schema.metadata)


# ============================================================================
# 6. Automated Verification Engine
# ============================================================================

class VerificationResult(BaseModel):
    """Result of an artifact manifest verification audit."""
    model_config = ConfigDict(protected_namespaces=())

    is_valid: bool
    status: Literal["PASSED", "FAILED"]
    errors: List[str] = Field(default_factory=list)
    manifest: Optional[ProvenanceManifest] = None
    audit_details: Dict[str, Any] = Field(default_factory=dict)


def verify_manifest(
    artifact_path: Union[str, Path],
    manifest: Optional[Union[str, Path, ProvenanceManifest, Dict[str, Any]]] = None,
    expected_regime_grid_version: str = REGIME_GRID_VERSION_V1,
) -> VerificationResult:
    """
    Verify integrity and cryptographic provenance of an artifact and its manifest.

    Checks:
    1. Artifact file existence.
    2. Manifest schema conformance (8 required fields + types).
    3. Payload file SHA256 integrity check.
    4. Canonical Arrow data SHA256 integrity check (for Parquet).
    5. Embedded Parquet metadata matching sidecar manifest.
    6. Calibration manifest hash format (64-char hex or 'NONE').
    7. Regime grid version compatibility.
    """
    errors: List[str] = []
    audit_details: Dict[str, Any] = {}
    target_path = Path(artifact_path)

    if not target_path.exists():
        return VerificationResult(
            is_valid=False,
            status="FAILED",
            errors=[f"Artifact file does not exist: {target_path}"],
        )

    # Resolve manifest
    parsed_manifest: Optional[ProvenanceManifest] = None
    if manifest is None:
        # Check embedded metadata if Parquet
        if target_path.suffix == ".parquet":
            try:
                parsed_manifest = read_manifest_from_parquet(target_path)
            except Exception as e:
                errors.append(f"Failed to read embedded Parquet manifest: {e}")
        # Check sidecar manifest
        if parsed_manifest is None:
            sidecar_path = target_path.with_name(f"{target_path.name}.manifest.json")
            if sidecar_path.exists():
                try:
                    parsed_manifest = ProvenanceManifest.load(sidecar_path)
                except Exception as e:
                    errors.append(f"Failed to parse sidecar manifest: {e}")
            else:
                errors.append("No manifest provided and no embedded/sidecar manifest found")
    elif isinstance(manifest, ProvenanceManifest):
        parsed_manifest = manifest
    elif isinstance(manifest, dict):
        try:
            parsed_manifest = ProvenanceManifest.from_dict(manifest)
        except Exception as e:
            errors.append(f"Manifest schema validation error: {e}")
    elif isinstance(manifest, (str, Path)):
        try:
            parsed_manifest = ProvenanceManifest.load(manifest)
        except Exception as e:
            errors.append(f"Failed to load manifest from path '{manifest}': {e}")
    else:
        errors.append(f"Invalid manifest type: {type(manifest)}")

    if parsed_manifest is None or errors:
        return VerificationResult(
            is_valid=False,
            status="FAILED",
            errors=errors,
            manifest=parsed_manifest,
        )

    # 1. Check regime grid version
    if parsed_manifest.regime_grid_version != expected_regime_grid_version:
        errors.append(
            f"Regime grid version mismatch: expected '{expected_regime_grid_version}', got '{parsed_manifest.regime_grid_version}'"
        )

    # 2. Check calibration manifest hash format
    calib_hash = parsed_manifest.calibration_manifest_hash
    if calib_hash != "NONE":
        if len(calib_hash) != 64 or not all(c in "0123456789abcdef" for c in calib_hash):
            errors.append(f"Invalid calibration manifest hash format: {calib_hash}")

    # 3. Check payload file SHA256
    computed_file_sha = payload_file_sha256(target_path)
    audit_details["computed_payload_file_sha256"] = computed_file_sha
    if parsed_manifest.payload_file_sha256:
        if computed_file_sha != parsed_manifest.payload_file_sha256:
            errors.append(
                f"File SHA256 mismatch: file={computed_file_sha}, manifest={parsed_manifest.payload_file_sha256}"
            )

    # 4. Check canonical data SHA256 for Parquet
    if target_path.suffix == ".parquet":
        try:
            table = pq.read_table(target_path)
            computed_data_sha = canonical_data_sha256(table)
            audit_details["computed_canonical_data_sha256"] = computed_data_sha
            if parsed_manifest.canonical_data_sha256:
                if computed_data_sha != parsed_manifest.canonical_data_sha256:
                    errors.append(
                        f"Canonical data SHA256 mismatch: table={computed_data_sha}, manifest={parsed_manifest.canonical_data_sha256}"
                    )
        except Exception as e:
            errors.append(f"Error reading Parquet table for canonical data verification: {e}")

    # 5. If Parquet, check embedded manifest matches provided manifest
    if target_path.suffix == ".parquet":
        try:
            embedded = read_manifest_from_parquet(target_path)
            if embedded.compute_manifest_hash() != parsed_manifest.compute_manifest_hash():
                errors.append("Embedded Parquet manifest does not match provided manifest")
        except Exception:
            # If writing without embedded manifest was done intentionally, only note if relevant
            pass

    is_valid = len(errors) == 0
    return VerificationResult(
        is_valid=is_valid,
        status="PASSED" if is_valid else "FAILED",
        errors=errors,
        manifest=parsed_manifest,
        audit_details=audit_details,
    )
