"""
SIH26054 Phase 1: Unit Tests for R2 Grouped Anti-Leakage Splitters (splitters.py).
"""

from __future__ import annotations

import random
from typing import Dict, List

import polars as pl
import pyarrow as pa
import pytest

from replan_to_learn.provenance.manifest import (
    REGIME_GRID_VERSION_V1,
    ProvenanceManifest,
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


# ============================================================================
# Test Fixtures & Synthetic Generators
# ============================================================================

@pytest.fixture
def multi_flight_df() -> pl.DataFrame:
    """Generate multi-flight synthetic telemetry DataFrame with 10 flights across 2 aircraft."""
    rows = []
    flight_durations = {
        "flight_01": 200, "flight_02": 300, "flight_03": 150, "flight_04": 400, "flight_05": 250,
        "flight_06": 350, "flight_07": 180, "flight_08": 220, "flight_09": 500, "flight_10": 100,
    }
    t_global = 0.0
    for fl_idx, (fl_id, n_sec) in enumerate(flight_durations.items()):
        ac_id = "AC_01" if fl_idx < 5 else "AC_02"
        for s in range(n_sec):
            rows.append({
                "timestamp": t_global + s,
                "flight_id": fl_id,
                "aircraft_id": ac_id,
                "engine_id": "ROTAX_915_IS_001" if ac_id == "AC_01" else "ROTAX_915_IS_002",
                "egt_1": 1050.0 + random.uniform(-5, 5),
                "cht": 450.0 + random.uniform(-2, 2),
                "oil_pressure": 350000.0,
                "oil_temperature": 360.0,
                "engine_speed_rpm": 5000.0,
                "fuel_flow_kg_s": 0.0055,
            })
        t_global += n_sec + 100.0

    return pl.DataFrame(rows)


@pytest.fixture
def single_aircraft_df() -> pl.DataFrame:
    """Generate multi-flight DataFrame for a single aircraft (AC_SOLO)."""
    rows = []
    t_global = 0.0
    for fl_idx in range(1, 6):
        fl_id = f"flight_solo_{fl_idx:02d}"
        for s in range(100):
            rows.append({
                "timestamp": t_global + s,
                "flight_id": fl_id,
                "aircraft_id": "AC_SOLO",
                "engine_id": "ROTAX_915_IS_SOLO",
                "egt_1": 1050.0,
                "cht": 450.0,
                "oil_pressure": 350000.0,
                "oil_temperature": 360.0,
                "engine_speed_rpm": 5000.0,
                "fuel_flow_kg_s": 0.0055,
            })
        t_global += 200.0
    return pl.DataFrame(rows)


# ============================================================================
# 1. SplitConfiguration Validation Tests
# ============================================================================

def test_split_configuration_valid_ratios():
    """Verify SplitConfiguration instantiates cleanly when ratios sum to 1.0."""
    cfg = SplitConfiguration(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    assert cfg.train_ratio == 0.70
    assert cfg.val_ratio == 0.15
    assert cfg.test_ratio == 0.15


def test_split_configuration_invalid_ratios_raise():
    """Verify ValueError when ratios do not sum to 1.0."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        SplitConfiguration(train_ratio=0.80, val_ratio=0.20, test_ratio=0.20)

    with pytest.raises(ValueError, match="Split ratios must be non-negative"):
        SplitConfiguration(train_ratio=1.2, val_ratio=-0.2, test_ratio=0.0)


# ============================================================================
# 2. GroupedFlightSplitter Tests
# ============================================================================

def test_grouped_flight_splitter_zero_flight_leakage(multi_flight_df: pl.DataFrame):
    """
    Verify GroupedFlightSplitter guarantees:
    1. flights(Train) ∩ flights(Val) ∩ flights(Test) == ∅
    2. indices(Train) ∩ indices(Val) ∩ indices(Test) == ∅
    3. Sum of split heights == total rows
    """
    splitter = GroupedFlightSplitter()
    cfg = SplitConfiguration(strategy="grouped_flight", train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=42)
    res = splitter.split(multi_flight_df, cfg)

    train_fl = set(res.train_groups)
    val_fl = set(res.val_groups)
    test_fl = set(res.test_groups)

    assert train_fl.isdisjoint(val_fl)
    assert train_fl.isdisjoint(test_fl)
    assert val_fl.isdisjoint(test_fl)
    assert len(train_fl) + len(val_fl) + len(test_fl) == 10

    # Verify index disjointness
    train_idx = set(res.train_indices)
    val_idx = set(res.val_indices)
    test_idx = set(res.test_indices)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert len(train_idx) + len(val_idx) + len(test_idx) == multi_flight_df.height


def test_grouped_flight_splitter_preserves_intra_flight_temporal_order(multi_flight_df: pl.DataFrame):
    """Verify row ordering and timestamp monotonicity within each flight are preserved."""
    splitter = GroupedFlightSplitter()
    res = splitter.split(multi_flight_df)

    for fold_df in [res.train_df, res.val_df, res.test_df]:
        if fold_df.height > 0:
            for (fl_id,), sub_df in fold_df.partition_by("flight_id", as_dict=True).items():
                timestamps = sub_df["timestamp"].to_list()
                assert timestamps == sorted(timestamps), f"Timestamps out of order in flight {fl_id}"


def test_grouped_flight_splitter_arrow_table_support(multi_flight_df: pl.DataFrame):
    """Verify GroupedFlightSplitter seamlessly accepts PyArrow Table."""
    arrow_table = multi_flight_df.to_arrow()
    splitter = GroupedFlightSplitter()
    res = splitter.split(arrow_table)
    assert isinstance(res.train_df, pl.DataFrame)
    assert res.train_df.height > 0


def test_grouped_flight_splitter_manifest_generation(multi_flight_df: pl.DataFrame):
    """Verify split manifests are created with correct provenance and parent lineage."""
    parent_manifest = ProvenanceManifest(
        dataset_id="DS_ORIGINAL",
        flight_id="MULTI",
        aircraft_id="AC_FLEET",
        engine_id="ROTAX_915_IS",
        model_version="1.0.0",
        regime_grid_version=REGIME_GRID_VERSION_V1,
        calibration_manifest_hash="NONE",
        schema_version="TELEMETRY_SCHEMA_V1",
    )
    splitter = GroupedFlightSplitter()
    res = splitter.split(multi_flight_df, parent_manifest=parent_manifest)

    assert "train" in res.split_manifests
    assert "val" in res.split_manifests
    assert "test" in res.split_manifests

    train_m = res.split_manifests["train"]
    assert train_m.dataset_id == "DS_ORIGINAL_TRAIN"
    assert train_m.split_group == "train"
    assert train_m.parent_manifest_hashes == [parent_manifest.compute_manifest_hash()]


# ============================================================================
# 3. GroupedAircraftSplitter Tests
# ============================================================================

def test_grouped_aircraft_splitter_zero_aircraft_leakage(multi_flight_df: pl.DataFrame):
    """
    Verify GroupedAircraftSplitter guarantees:
    aircraft(Train) ∩ aircraft(Test) == ∅
    """
    splitter = GroupedAircraftSplitter()
    cfg = SplitConfiguration(strategy="grouped_aircraft", train_ratio=0.50, val_ratio=0.0, test_ratio=0.50, seed=42)
    res = splitter.split(multi_flight_df, cfg)

    train_ac = set(res.train_groups)
    test_ac = set(res.test_groups)

    assert train_ac.isdisjoint(test_ac)
    assert "AC_01" in train_ac or "AC_01" in test_ac
    assert "AC_02" in train_ac or "AC_02" in test_ac


def test_grouped_aircraft_splitter_single_aircraft_fallback(single_aircraft_df: pl.DataFrame):
    """
    Verify single-aircraft dataset triggers SingleAircraftWarning and graceful fallback
    to flight-based grouping when allow_single_aircraft_fallback=True.
    """
    splitter = GroupedAircraftSplitter()
    cfg = SplitConfiguration(strategy="grouped_aircraft", allow_single_aircraft_fallback=True)

    with pytest.warns(SingleAircraftWarning, match="falling back to GroupedFlightSplitter"):
        res = splitter.split(single_aircraft_df, cfg)

    assert res.audit_summary["fallback_applied"] is True
    assert res.audit_summary["fleet_generalization_evaluable"] is False
    assert len(res.train_groups) > 0


def test_grouped_aircraft_splitter_single_aircraft_error_when_fallback_disabled(single_aircraft_df: pl.DataFrame):
    """Verify ValueError is raised on single-aircraft dataset when fallback is disabled."""
    splitter = GroupedAircraftSplitter()
    cfg = SplitConfiguration(strategy="grouped_aircraft", allow_single_aircraft_fallback=False)

    with pytest.raises(ValueError, match="Cannot perform grouped aircraft split"):
        splitter.split(single_aircraft_df, cfg)


# ============================================================================
# 4. HierarchicalGroupSplitter Tests
# ============================================================================

def test_hierarchical_group_splitter_two_tier_partitioning(multi_flight_df: pl.DataFrame):
    """
    Verify two-tier hierarchical partitioning:
    - Train: Seen Aircraft, Seen Flights
    - Val: Seen Aircraft, Unseen Validation Flights
    - Test: Unseen Aircraft, Unseen Flights
    """
    splitter = HierarchicalGroupSplitter()
    cfg = SplitConfiguration(strategy="hierarchical", train_ratio=0.50, val_ratio=0.25, test_ratio=0.25, seed=42)
    res = splitter.split(multi_flight_df, cfg)

    train_ac = set(res.audit_summary["train_aircraft"])
    test_ac = set(res.audit_summary["test_aircraft"])

    # Level 1 disjointness: Aircraft in Train/Val vs Test
    assert train_ac.isdisjoint(test_ac)

    # Level 2 disjointness: Flights across all 3 partitions
    train_fl = set(res.train_groups)
    val_fl = set(res.val_groups)
    test_fl = set(res.test_groups)

    assert train_fl.isdisjoint(val_fl)
    assert train_fl.isdisjoint(test_fl)
    assert val_fl.isdisjoint(test_fl)

    # Zero sample leakage
    res.assert_zero_leakage(group_type="flight")


def test_hierarchical_group_splitter_single_aircraft_fallback(single_aircraft_df: pl.DataFrame):
    """Verify HierarchicalGroupSplitter warns and falls back to flight grouping on N_ac == 1."""
    splitter = HierarchicalGroupSplitter()
    cfg = SplitConfiguration(strategy="hierarchical", allow_single_aircraft_fallback=True)

    with pytest.warns(SingleAircraftWarning, match="Falling back to 3-way GroupedFlightSplitter"):
        res = splitter.split(single_aircraft_df, cfg)

    assert res.audit_summary["fallback_applied"] is True
    assert res.audit_summary["fleet_generalization_evaluable"] is False


# ============================================================================
# 5. TemporalFlightSplitter Tests
# ============================================================================

def test_temporal_flight_splitter_chronological_ordering(multi_flight_df: pl.DataFrame):
    """
    Verify TemporalFlightSplitter guarantees zero lookahead leakage:
    max(timestamp(Train)) < min(timestamp(Val)) and max(timestamp(Val)) < min(timestamp(Test))
    """
    splitter = TemporalFlightSplitter()
    cfg = SplitConfiguration(strategy="temporal", train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)
    res = splitter.split(multi_flight_df, cfg)

    assert res.train_df.height > 0
    assert res.val_df.height > 0
    assert res.test_df.height > 0

    t_train_max = float(res.train_df["timestamp"].max())
    t_val_min = float(res.val_df["timestamp"].min())
    t_val_max = float(res.val_df["timestamp"].max())
    t_test_min = float(res.test_df["timestamp"].min())

    assert t_train_max < t_val_min, f"Temporal lookahead leak: train_max={t_train_max} >= val_min={t_val_min}"
    assert t_val_max < t_test_min, f"Temporal lookahead leak: val_max={t_val_max} >= test_min={t_test_min}"


# ============================================================================
# 6. ZeroLeakageViolationError Assertion Tests
# ============================================================================

def test_zero_leakage_assertion_detects_row_overlap(multi_flight_df: pl.DataFrame):
    """Verify assert_zero_leakage raises ZeroLeakageViolationError if indices overlap."""
    splitter = GroupedFlightSplitter()
    res = splitter.split(multi_flight_df)

    # Intentionally corrupt test_indices to overlap with train
    res.test_indices.append(res.train_indices[0])

    with pytest.raises(ZeroLeakageViolationError, match="Sample index leakage detected"):
        res.assert_zero_leakage()
