"""
SIH26054 Replan to Learn: Reproducibility verification (M3, F12).
Deterministic re-execution dry-runs and lineage-chain integrity checks, per
01_foundation_data_contracts.md Sec 4.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from replan_to_learn.provenance.hasher import canonical_json_dumps
from replan_to_learn.provenance.manifest import VerificationResult, verify_manifest
from replan_to_learn.reproducibility.lineage import LineageDAG


class ReproducibilityVerifier:
    """Verifies a single artifact's provenance integrity, or an entire lineage chain."""

    @staticmethod
    def verify_artifact(path: Union[str, Path], manifest=None) -> VerificationResult:
        return verify_manifest(path, manifest=manifest)

    @staticmethod
    def verify_lineage_chain(dag: LineageDAG, node_hash: str, registry) -> Dict[str, Any]:
        """
        Walk every ancestor of `node_hash` and verify each one whose artifact
        can still be resolved in `registry` (an ArtifactRegistry). Returns a
        report: per-node VerificationResult plus an overall pass/fail.
        """
        if node_hash not in dag.graph:
            return {
                "root": node_hash,
                "chain_length": 0,
                "all_valid": False,
                "results": {},
                "error": f"Node {node_hash} not present in lineage DAG",
            }

        chain = sorted(dag.ancestors(node_hash) | {node_hash})
        results: Dict[str, bool] = {}
        errors: Dict[str, list] = {}
        resolved = 0

        from replan_to_learn.registry.paths import ARTIFACT_CATEGORIES

        for node in chain:
            manifest = dag.manifest_for(node)
            if manifest is None:
                continue
            artifact_path = None
            for category in ARTIFACT_CATEGORIES:
                candidate = registry.paths.artifact_path(category, f"{manifest.dataset_id}.parquet")
                if candidate.exists():
                    artifact_path = candidate
                    break
            if artifact_path is None:
                continue
            resolved += 1
            result = verify_manifest(artifact_path, manifest=manifest)
            results[node] = result.is_valid
            if not result.is_valid:
                errors[node] = result.errors

        return {
            "root": node_hash,
            "chain_length": len(chain),
            "resolved_count": resolved,
            "all_valid": all(results.values()) if results else False,
            "results": results,
            "errors": errors,
        }

    @staticmethod
    def deterministic_replay_check(
        compute_fn: Callable[[], Any],
        n_runs: int = 2,
        hash_fn: Optional[Callable[[Any], str]] = None,
    ) -> Dict[str, Any]:
        """
        Run `compute_fn()` `n_runs` times and check every run produces a
        bit-identical result -- the "deterministic re-execution dry-run"
        requirement from Sec 4. Implements the same bit-identical-replay
        contract as G1.5, generalized to any artifact-producing function
        (not just PhysicsTwin.step), since reproducibility is a
        registry-wide requirement, not just Stage 2's.

        `hash_fn` lets the caller supply a custom serializer for outputs
        that aren't JSON-serializable (e.g. hash the .tobytes() of a numpy
        array); the default hashes the canonical JSON representation.
        """
        if hash_fn is None:
            def hash_fn(result: Any) -> str:
                try:
                    payload = canonical_json_dumps(result).encode("utf-8")
                except TypeError:
                    payload = repr(result).encode("utf-8")
                return hashlib.sha256(payload).hexdigest()

        hashes = []
        for _ in range(n_runs):
            result = compute_fn()
            hashes.append(hash_fn(result))

        return {
            "n_runs": n_runs,
            "hashes": hashes,
            "is_deterministic": len(set(hashes)) == 1,
        }
