"""
SIH26054 Replan to Learn: Reproducibility Engine (M3, F12).
Orchestrates lineage DAG construction and chain verification for a given
artifact, per 01_foundation_data_contracts.md Sec 4 ("Each artifact must be
reproducible from its manifest").
"""

from __future__ import annotations

from typing import Any, Dict

from replan_to_learn.provenance.hasher import compute_manifest_hash
from replan_to_learn.reproducibility.lineage import LineageDAG
from replan_to_learn.reproducibility.verifier import ReproducibilityVerifier


class ReproducibilityEngine:
    """
    Ties an ArtifactRegistry to lineage resolution and integrity
    verification. Usage:

        engine = ReproducibilityEngine(registry)
        report = engine.verify_reproducibility("processed", "flight_123.parquet")
    """

    def __init__(self, registry) -> None:
        self.registry = registry

    def build_lineage_dag(self) -> LineageDAG:
        return LineageDAG.build_from_registry(self.registry)

    def verify_reproducibility(self, category: str, filename: str) -> Dict[str, Any]:
        _, manifest = self.registry.load(category, filename)
        node_hash = compute_manifest_hash(manifest)

        dag = self.build_lineage_dag()
        chain_report = ReproducibilityVerifier.verify_lineage_chain(dag, node_hash, self.registry)
        artifact_result = ReproducibilityVerifier.verify_artifact(
            self.registry.resolve_path(category, filename), manifest=manifest
        )

        return {
            "artifact": filename,
            "category": category,
            "manifest_hash": node_hash,
            "artifact_valid": artifact_result.is_valid,
            "artifact_errors": artifact_result.errors,
            "lineage": chain_report,
            "fully_reproducible": artifact_result.is_valid and chain_report["all_valid"],
        }

    def persist_lineage_index(self) -> None:
        dag = self.build_lineage_dag()
        dag.save(self.registry.paths.lineage_index_path())
