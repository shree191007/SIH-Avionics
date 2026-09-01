"""
SIH26054 Replan to Learn: Artifact Registry (M3, F10-F11).
Implements 01_foundation_data_contracts.md Sec 3-4: every artifact stored
under artifacts/<category>/ is reproducible from its manifest, written
atomically, and loadable with zero-copy Polars/PyArrow.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import polars as pl
import pyarrow as pa

from replan_to_learn.provenance.manifest import (
    ProvenanceManifest,
    read_manifest_from_parquet,
    write_parquet_with_manifest,
)
from replan_to_learn.registry.atomic import atomic_write_via
from replan_to_learn.registry.paths import ArtifactPaths


class ArtifactRegistry:
    """
    Content-addressable-by-manifest artifact store.

    Every stored artifact is a Parquet file with its ProvenanceManifest
    embedded in the file's Arrow schema metadata (so a reader never needs a
    separate manifest lookup to get provenance) AND mirrored to a sidecar
    `<filename>.manifest.json` (so lineage traversal and manifest audits
    don't require opening the Parquet payload).
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self.paths = ArtifactPaths(Path(root))
        self.paths.ensure_hierarchy()

    def store(
        self,
        category: str,
        table_or_df: Union[pa.Table, pl.DataFrame],
        manifest: ProvenanceManifest,
        filename: Optional[str] = None,
        **parquet_kwargs,
    ) -> Path:
        """Atomically persist `table_or_df` + `manifest` under artifacts/<category>/."""
        if filename is None:
            filename = f"{manifest.dataset_id}.parquet"
        if not filename.endswith(".parquet"):
            filename = f"{filename}.parquet"

        target = self.paths.artifact_path(category, filename)

        def _write(tmp_path: Path) -> None:
            write_parquet_with_manifest(table_or_df, tmp_path, manifest, **parquet_kwargs)

        atomic_write_via(target, _write)

        sidecar = self.paths.manifest_sidecar_path(target)
        manifest.save(sidecar)

        return target

    def load(self, category: str, filename: str) -> Tuple[pl.DataFrame, ProvenanceManifest]:
        """Load an artifact's data (zero-copy via Polars) and its embedded manifest."""
        path = self.resolve_path(category, filename)
        manifest = read_manifest_from_parquet(path)
        df = pl.read_parquet(path)
        return df, manifest

    def load_manifest(self, category: str, filename: str) -> ProvenanceManifest:
        path = self.resolve_path(category, filename)
        sidecar = self.paths.manifest_sidecar_path(path)
        if sidecar.exists():
            return ProvenanceManifest.load(sidecar)
        return read_manifest_from_parquet(path)

    def resolve_path(self, category: str, filename: str) -> Path:
        if not filename.endswith(".parquet"):
            filename = f"{filename}.parquet"
        path = self.paths.artifact_path(category, filename)
        if not path.exists():
            raise FileNotFoundError(f"No artifact '{filename}' in category '{category}' under {self.paths.root}")
        return path

    def list_artifacts(self, category: str) -> List[Path]:
        return sorted(self.paths.category_dir(category).glob("*.parquet"))

    def list_all_manifests(self) -> List[ProvenanceManifest]:
        """Every manifest across every category -- used by the lineage DAG builder."""
        manifests: List[ProvenanceManifest] = []
        from replan_to_learn.registry.paths import ARTIFACT_CATEGORIES

        for category in ARTIFACT_CATEGORIES:
            for artifact_path in self.list_artifacts(category):
                sidecar = self.paths.manifest_sidecar_path(artifact_path)
                try:
                    if sidecar.exists():
                        manifests.append(ProvenanceManifest.load(sidecar))
                    else:
                        manifests.append(read_manifest_from_parquet(artifact_path))
                except Exception:
                    continue
        return manifests
