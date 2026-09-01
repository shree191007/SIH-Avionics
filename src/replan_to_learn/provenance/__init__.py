"""
SIH26054 Replan to Learn: Requirement 2 (Data Versioning & Provenance Metadata).
Exports manifests, deterministic hashing, Parquet metadata roundtrips, and zero-leakage splitters.
"""

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
    BaseSplitter,
    GroupedAircraftSplitter,
    GroupedFlightSplitter,
    HierarchicalGroupSplitter,
    SingleAircraftWarning,
    SplitConfiguration,
    SplitResult,
    TemporalFlightSplitter,
    ZeroLeakageViolationError,
)

__all__ = [
    # Hasher
    "canonical_data_sha256",
    "canonical_json_dumps",
    "canonical_json_loads",
    "compute_manifest_hash",
    "payload_file_sha256",
    # Manifest
    "PARQUET_MANIFEST_KEY",
    "REGIME_GRID_VERSION_V1",
    "ManifestGenerator",
    "ProvenanceAuditMetadata",
    "ProvenanceManifest",
    "VerificationResult",
    "embed_manifest_in_schema",
    "read_manifest_from_parquet",
    "verify_manifest",
    "write_parquet_with_manifest",
    # Splitters
    "BaseSplitter",
    "GroupedAircraftSplitter",
    "GroupedFlightSplitter",
    "HierarchicalGroupSplitter",
    "SingleAircraftWarning",
    "SplitConfiguration",
    "SplitResult",
    "TemporalFlightSplitter",
    "ZeroLeakageViolationError",
]
