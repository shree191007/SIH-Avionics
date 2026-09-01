"""
SIH26054 Replan to Learn: Manifest loaders (M3, F11).
Per PROJECT.md's M2<->M3 interface contract: uses manifest hashes and
embedded Parquet metadata to load and validate artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl

from replan_to_learn.provenance.hasher import compute_manifest_hash
from replan_to_learn.provenance.manifest import (
    ProvenanceManifest,
    VerificationResult,
    read_manifest_from_parquet,
    verify_manifest,
)


class ManifestLoader:
    """Loads and validates a single artifact against its manifest hash."""

    @staticmethod
    def load_manifest(path: Union[str, Path]) -> ProvenanceManifest:
        path = Path(path)
        sidecar = path.with_name(f"{path.name}.manifest.json")
        if sidecar.exists():
            return ProvenanceManifest.load(sidecar)
        return read_manifest_from_parquet(path)

    @staticmethod
    def manifest_hash(manifest: ProvenanceManifest) -> str:
        return compute_manifest_hash(manifest)

    @staticmethod
    def load_artifact(path: Union[str, Path]) -> pl.DataFrame:
        return pl.read_parquet(path)

    @staticmethod
    def load_and_verify(path: Union[str, Path]) -> tuple[pl.DataFrame, ProvenanceManifest, VerificationResult]:
        """Load an artifact's data + manifest, and run the full provenance integrity audit."""
        path = Path(path)
        manifest = ManifestLoader.load_manifest(path)
        result = verify_manifest(path, manifest=manifest)
        df = pl.read_parquet(path)
        return df, manifest, result
