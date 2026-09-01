"""
Tests for the artifact registry (M3, F10-F11):
7-directory hierarchy, atomic multi-process persistence, manifest loaders.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from replan_to_learn.provenance.hasher import compute_manifest_hash
from replan_to_learn.provenance.manifest import ManifestGenerator
from replan_to_learn.registry import ARTIFACT_CATEGORIES, ArtifactPaths, ArtifactRegistry, ManifestLoader
from replan_to_learn.registry.atomic import atomic_write_json


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


class TestArtifactPaths:
    def test_ensure_hierarchy_creates_all_seven_categories(self, tmp_path):
        paths = ArtifactPaths(tmp_path / "artifacts")
        paths.ensure_hierarchy()
        for category in ARTIFACT_CATEGORIES:
            assert (tmp_path / "artifacts" / category).is_dir()

    def test_category_dir_rejects_unknown_category(self, tmp_path):
        paths = ArtifactPaths(tmp_path / "artifacts")
        with pytest.raises(ValueError):
            paths.category_dir("not_a_real_category")

    def test_seven_categories_match_spec(self):
        assert set(ARTIFACT_CATEGORIES) == {
            "raw", "processed", "calibration", "models", "fingerprints", "replay", "evaluation",
        }


class TestArtifactRegistryStoreLoad:
    def test_store_creates_hierarchy_on_init(self, tmp_path):
        ArtifactRegistry(tmp_path / "artifacts")
        for category in ARTIFACT_CATEGORIES:
            assert (tmp_path / "artifacts" / category).is_dir()

    def test_store_and_load_roundtrip(self, registry, synthetic_flight_dataframe, raw_manifest):
        path = registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        assert path.exists()
        assert path.parent.name == "raw"

        df, manifest = registry.load("raw", "flight_7152_raw.parquet")
        assert df.shape == synthetic_flight_dataframe.shape
        assert manifest.dataset_id == "flight_7152_raw"
        assert manifest.flight_id == "7152"

    def test_store_writes_sidecar_manifest(self, registry, synthetic_flight_dataframe, raw_manifest):
        path = registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        sidecar = path.with_name(f"{path.name}.manifest.json")
        assert sidecar.exists()

    def test_load_manifest_matches_embedded_manifest(self, registry, synthetic_flight_dataframe, raw_manifest):
        registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        manifest = registry.load_manifest("raw", "flight_7152_raw.parquet")
        assert manifest.dataset_id == raw_manifest.dataset_id
        assert compute_manifest_hash(manifest) == compute_manifest_hash(raw_manifest)

    def test_resolve_path_raises_for_missing_artifact(self, registry):
        with pytest.raises(FileNotFoundError):
            registry.resolve_path("raw", "does_not_exist")

    def test_list_artifacts_returns_stored_files(self, registry, synthetic_flight_dataframe, raw_manifest):
        assert registry.list_artifacts("raw") == []
        registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        artifacts = registry.list_artifacts("raw")
        assert len(artifacts) == 1
        assert artifacts[0].name == "flight_7152_raw.parquet"

    def test_store_across_all_seven_categories(self, registry, synthetic_flight_dataframe):
        for i, category in enumerate(ARTIFACT_CATEGORIES):
            manifest = ManifestGenerator.create_raw_manifest(
                dataset_id=f"ds_{category}_{i}",
                flight_id="7152",
                aircraft_id="AC_01",
                engine_id="ROTAX_915_IS_SN1042",
                data=synthetic_flight_dataframe,
            )
            path = registry.store(category, synthetic_flight_dataframe, manifest)
            assert path.parent.name == category

        for category in ARTIFACT_CATEGORIES:
            assert len(registry.list_artifacts(category)) == 1

    def test_list_all_manifests_across_categories(self, registry, synthetic_flight_dataframe):
        for i, category in enumerate(ARTIFACT_CATEGORIES):
            manifest = ManifestGenerator.create_raw_manifest(
                dataset_id=f"ds_{category}_{i}",
                flight_id="7152",
                aircraft_id="AC_01",
                engine_id="ROTAX_915_IS_SN1042",
                data=synthetic_flight_dataframe,
            )
            registry.store(category, synthetic_flight_dataframe, manifest)

        all_manifests = registry.list_all_manifests()
        assert len(all_manifests) == len(ARTIFACT_CATEGORIES)


class TestAtomicPersistence:
    def test_atomic_write_json_no_partial_file_on_success(self, tmp_path):
        target = tmp_path / "manifest.json"
        atomic_write_json(target, {"a": 1, "b": 2})
        assert target.exists()
        assert not target.with_suffix(".tmp").exists()

    def test_atomic_write_produces_no_leftover_tmp_files(self, tmp_path):
        target = tmp_path / "artifact.json"
        for i in range(5):
            atomic_write_json(target, {"iteration": i})
        leftovers = list(tmp_path.glob(".*"))
        assert leftovers == []

    def test_reproducible_registry_write_is_content_stable(self, registry, synthetic_flight_dataframe, raw_manifest):
        """Writing the same manifest twice (idempotent re-run) must not corrupt the artifact."""
        registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        df, manifest = registry.load("raw", "flight_7152_raw.parquet")
        assert df.shape == synthetic_flight_dataframe.shape


class TestManifestLoader:
    def test_load_and_verify_valid_artifact(self, registry, synthetic_flight_dataframe, raw_manifest):
        path = registry.store("raw", synthetic_flight_dataframe, raw_manifest)
        df, manifest, result = ManifestLoader.load_and_verify(path)
        assert result.is_valid
        assert result.status == "PASSED"
        assert df.shape == synthetic_flight_dataframe.shape

    def test_manifest_hash_is_deterministic(self, raw_manifest):
        h1 = ManifestLoader.manifest_hash(raw_manifest)
        h2 = ManifestLoader.manifest_hash(raw_manifest)
        assert h1 == h2
        assert len(h1) == 64
