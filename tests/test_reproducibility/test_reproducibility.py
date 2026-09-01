"""
Tests for the reproducibility engine (M3, F12):
Merkle lineage DAG traversal, deterministic replay verification, and
end-to-end "reproducible from its manifest" checks.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from replan_to_learn.provenance.hasher import compute_manifest_hash
from replan_to_learn.provenance.manifest import ManifestGenerator
from replan_to_learn.registry import ArtifactRegistry
from replan_to_learn.reproducibility import LineageDAG, ReproducibilityEngine, ReproducibilityVerifier


@pytest.fixture
def registry(tmp_path: Path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "artifacts")


@pytest.fixture
def raw_manifest(synthetic_flight_dataframe: pl.DataFrame):
    return ManifestGenerator.create_raw_manifest(
        dataset_id="flight_7152_raw",
        flight_id="7152",
        aircraft_id="AC_01",
        engine_id="ROTAX_915_IS_SN1042",
        data=synthetic_flight_dataframe,
    )


def _processed_manifest(parent_hash: str, synthetic_flight_dataframe: pl.DataFrame):
    return ManifestGenerator.create_processed_manifest(
        dataset_id="flight_7152_processed",
        flight_id="7152",
        aircraft_id="AC_01",
        engine_id="ROTAX_915_IS_SN1042",
        model_version="1.0.0",
        calibration_manifest_hash="NONE",
        parent_manifest_hashes=[parent_hash],
        data=synthetic_flight_dataframe,
    )


class TestLineageDAG:
    def test_add_manifest_creates_node(self, raw_manifest):
        dag = LineageDAG()
        node_hash = dag.add_manifest(raw_manifest)
        assert node_hash in dag.graph
        assert node_hash == compute_manifest_hash(raw_manifest)

    def test_parent_child_edge_from_parent_manifest_hashes(self, raw_manifest, synthetic_flight_dataframe):
        dag = LineageDAG()
        parent_hash = dag.add_manifest(raw_manifest)
        processed = _processed_manifest(parent_hash, synthetic_flight_dataframe)
        child_hash = dag.add_manifest(processed)

        assert dag.graph.has_edge(parent_hash, child_hash)
        assert parent_hash in dag.ancestors(child_hash)
        assert child_hash in dag.descendants(parent_hash)

    def test_dag_is_acyclic(self, raw_manifest, synthetic_flight_dataframe):
        dag = LineageDAG()
        parent_hash = dag.add_manifest(raw_manifest)
        dag.add_manifest(_processed_manifest(parent_hash, synthetic_flight_dataframe))
        assert dag.is_acyclic()

    def test_topological_order_respects_dependency(self, raw_manifest, synthetic_flight_dataframe):
        dag = LineageDAG()
        parent_hash = dag.add_manifest(raw_manifest)
        child_hash = dag.add_manifest(_processed_manifest(parent_hash, synthetic_flight_dataframe))
        order = dag.topological_order()
        assert order.index(parent_hash) < order.index(child_hash)

    def test_build_from_registry_reconstructs_full_graph(self, registry, raw_manifest, synthetic_flight_dataframe):
        registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        parent_hash = compute_manifest_hash(raw_manifest)
        processed = _processed_manifest(parent_hash, synthetic_flight_dataframe)
        registry.store("processed", synthetic_flight_dataframe, processed)

        dag = LineageDAG.build_from_registry(registry)
        child_hash = compute_manifest_hash(processed)
        assert parent_hash in dag.graph
        assert child_hash in dag.graph
        assert dag.graph.has_edge(parent_hash, child_hash)

    def test_save_and_from_index_roundtrip(self, tmp_path, raw_manifest, synthetic_flight_dataframe):
        dag = LineageDAG()
        parent_hash = dag.add_manifest(raw_manifest)
        child_hash = dag.add_manifest(_processed_manifest(parent_hash, synthetic_flight_dataframe))

        index_path = tmp_path / "lineage_index.json"
        dag.save(index_path)
        assert index_path.exists()

        restored = LineageDAG.from_index(index_path)
        assert restored.graph.has_edge(parent_hash, child_hash)
        assert set(restored.graph.nodes()) == {parent_hash, child_hash}


class TestReproducibilityVerifier:
    def test_verify_artifact_valid(self, registry, raw_manifest, synthetic_flight_dataframe):
        path = registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        result = ReproducibilityVerifier.verify_artifact(path, manifest=raw_manifest)
        assert result.is_valid

    def test_verify_lineage_chain_all_valid(self, registry, raw_manifest, synthetic_flight_dataframe):
        registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        parent_hash = compute_manifest_hash(raw_manifest)
        processed = _processed_manifest(parent_hash, synthetic_flight_dataframe)
        registry.store("processed", synthetic_flight_dataframe, processed)
        child_hash = compute_manifest_hash(processed)

        dag = LineageDAG.build_from_registry(registry)
        report = ReproducibilityVerifier.verify_lineage_chain(dag, child_hash, registry)

        assert report["chain_length"] == 2
        assert report["all_valid"] is True

    def test_verify_lineage_chain_missing_node_reports_invalid(self, registry):
        dag = LineageDAG()
        report = ReproducibilityVerifier.verify_lineage_chain(dag, "nonexistent_hash", registry)
        assert report["all_valid"] is False
        assert report["chain_length"] == 0

    def test_deterministic_replay_check_detects_determinism(self):
        def compute_fn():
            return {"z": [0.1, 0.2, 0.3], "regime": 5}

        result = ReproducibilityVerifier.deterministic_replay_check(compute_fn, n_runs=3)
        assert result["is_deterministic"] is True
        assert result["n_runs"] == 3

    def test_deterministic_replay_check_detects_nondeterminism(self):
        state = {"call_count": 0}

        def compute_fn():
            state["call_count"] += 1
            return {"z": [0.1, 0.2, state["call_count"]]}

        result = ReproducibilityVerifier.deterministic_replay_check(compute_fn, n_runs=3)
        assert result["is_deterministic"] is False

    def test_deterministic_replay_check_on_physics_twin_step_g15(self):
        """
        Same bit-identical contract as physics_twin's G1.5, exercised through
        the registry-wide deterministic_replay_check API instead of a
        one-off PhysicsTwin-specific assertion.
        """
        import numpy as np

        def compute_fn():
            rng_state = np.random.default_rng(seed=7)
            return rng_state.standard_normal(5).tobytes()

        def hash_fn(payload: bytes) -> str:
            import hashlib
            return hashlib.sha256(payload).hexdigest()

        result = ReproducibilityVerifier.deterministic_replay_check(compute_fn, n_runs=4, hash_fn=hash_fn)
        assert result["is_deterministic"] is True


class TestReproducibilityEngine:
    def test_verify_reproducibility_full_report(self, registry, raw_manifest, synthetic_flight_dataframe):
        registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        parent_hash = compute_manifest_hash(raw_manifest)
        processed = _processed_manifest(parent_hash, synthetic_flight_dataframe)
        registry.store("processed", synthetic_flight_dataframe, processed)

        engine = ReproducibilityEngine(registry)
        report = engine.verify_reproducibility("processed", "flight_7152_processed.parquet")

        assert report["artifact_valid"] is True
        assert report["lineage"]["all_valid"] is True
        assert report["fully_reproducible"] is True

    def test_persist_lineage_index_writes_file(self, registry, raw_manifest, synthetic_flight_dataframe):
        registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        engine = ReproducibilityEngine(registry)
        engine.persist_lineage_index()
        assert registry.paths.lineage_index_path().exists()

    def test_corrupted_artifact_is_not_reproducible(self, registry, raw_manifest, synthetic_flight_dataframe):
        path = registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        # Simulate corruption: truncate the file after it was manifested.
        path.write_bytes(b"not a parquet file")

        engine = ReproducibilityEngine(registry)
        with pytest.raises(Exception):
            engine.verify_reproducibility("raw", "flight_7152_raw.parquet")
