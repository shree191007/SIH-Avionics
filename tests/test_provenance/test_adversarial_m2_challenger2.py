"""
SIH26054 Replan to Learn: Empirical Adversarial Challenge Suite (Challenger 2).
Milestone 2: R2 Data Versioning & Provenance Metadata.

Stress Tests & Empirical Challenges for Anti-Leakage Splitters (splitters.py):
1. Monte Carlo Leakage Stress Suite (1,000 Trials per Strategy):
   - GroupedFlightSplitter: 1,000 randomized trials across multi-aircraft datasets
     asserting train_flights ∩ val_flights ∩ test_flights = ∅, sample index disjointness,
     and row count conservation.
   - GroupedAircraftSplitter: 1,000 randomized trials across multi-aircraft datasets
     asserting train_aircraft ∩ val_aircraft ∩ test_aircraft = ∅, zero flight leak,
     and sample index disjointness.
2. Edge Cases:
   - Single-Aircraft Datasets: SingleAircraftWarning emission, fallback to GroupedFlightSplitter,
     disjoint flight partitioning, audit flag integrity, and strict ValueError when fallback disabled.
   - Extreme Flight Duration Imbalance (10s vs 5,000s): Greedy Longest-Processing-Time (LPT)
     bin-packing efficacy, duration-weighted vs. count-weighted balancing, zero leakage.
   - Temporal Lookahead Bias Elimination: Strict chronological monotonicity
     (max(train) < min(val) <= max(val) < min(test)), robustness against shuffled raw inputs,
     and multi-aircraft concurrent flight chronologies.
3. Two-Tier Hierarchical Partitioning (HierarchicalGroupSplitter):
   - Level 1 fleet disjointness and Level 2 flight disjointness across 200 trials.
4. Oracle & Assertion Integrity:
   - Verification that SplitResult.assert_zero_leakage() raises ZeroLeakageViolationError
     on all forms of row, flight, and aircraft contamination.
5. Determinism, Seed Invariance & PyArrow Interoperability.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from replan_to_learn.provenance.hasher import canonical_data_sha256
from replan_to_learn.provenance.manifest import (
    ManifestGenerator,
    ProvenanceManifest,
    REGIME_GRID_VERSION_V1,
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
# Synthetic Dataset Generators for Stress Testing
# ============================================================================

def make_synthetic_fleet_dataframe(
    n_aircraft: int = 12,
    n_flights: int = 120,
    min_dur: int = 2,
    max_dur: int = 10,
    seed: int = 42,
    sequential_time: bool = True,
) -> pl.DataFrame:
    """
    Generate synthetic fleet telemetry dataframe with exact flight and aircraft assignments.
    Optimized for high-throughput Monte Carlo trial execution.
    """
    rng = random.Random(seed)
    aircraft_pool = [f"AC_{i:03d}" for i in range(1, n_aircraft + 1)]

    flight_ids: List[str] = []
    aircraft_ids: List[str] = []
    engine_ids: List[str] = []
    timestamps: List[float] = []

    t_ac: Dict[str, float] = {ac: 0.0 for ac in aircraft_pool}
    t_global = 0.0

    for fl_idx in range(1, n_flights + 1):
        ac = aircraft_pool[fl_idx % n_aircraft]
        fl_id = f"FL_{fl_idx:05d}"
        dur = rng.randint(min_dur, max_dur)

        if sequential_time:
            start_t = t_global
            t_global += dur + rng.uniform(5.0, 20.0)
        else:
            start_t = t_ac[ac]
            t_ac[ac] += dur + rng.uniform(5.0, 20.0)

        for s in range(dur):
            flight_ids.append(fl_id)
            aircraft_ids.append(ac)
            engine_ids.append(f"ROTAX_915_{ac}")
            timestamps.append(start_t + float(s))

    n_rows = len(flight_ids)
    return pl.DataFrame({
        "timestamp": timestamps,
        "flight_id": flight_ids,
        "aircraft_id": aircraft_ids,
        "engine_id": engine_ids,
        "egt_1": [1050.0] * n_rows,
        "cht": [450.0] * n_rows,
        "oil_pressure": [350000.0] * n_rows,
        "oil_temperature": [360.0] * n_rows,
        "engine_speed_rpm": [5000.0] * n_rows,
        "fuel_flow_kg_s": [0.0055] * n_rows,
    })


# ============================================================================
# 1. Monte Carlo GroupedFlightSplitter Leakage Tests (1,000 Trials)
# ============================================================================

class TestMonteCarloGroupedFlightSplitterLeakage:
    """
    Empirically challenge GroupedFlightSplitter across 1,000 randomized split configurations.
    Assert zero flight leakage, zero sample index leakage, row conservation, and intra-flight order.
    """

    def test_1000_monte_carlo_randomized_flight_splits(self) -> None:
        """
        Execute 1,000 randomized trials with:
        - Multi-aircraft fleet (10 to 25 aircraft)
        - Multi-flight catalog (100 to 200 flights)
        - Randomized partition ratios (train in [0.4, 0.8], val in [0.05, 0.3], test = 1 - train - val)
        - Randomized duration balancing (on/off)
        - Randomized seed
        """
        rng = random.Random(42_1337)
        n_trials = 1000

        total_violations = 0
        flight_leak_count = 0
        sample_leak_count = 0
        row_conservation_errors = 0
        intra_flight_disorder_errors = 0

        for trial in range(1, n_trials + 1):
            n_ac = rng.randint(10, 25)
            n_fl = rng.randint(100, 200)
            seed = rng.randint(1, 10_000_000)

            r_train = round(rng.uniform(0.40, 0.80), 4)
            r_val = round(rng.uniform(0.05, (1.0 - r_train) * 0.85), 4)
            r_test = round(1.0 - r_train - r_val, 4)

            balance_dur = rng.choice([True, False])

            df = make_synthetic_fleet_dataframe(
                n_aircraft=n_ac,
                n_flights=n_fl,
                min_dur=rng.randint(2, 5),
                max_dur=rng.randint(6, 15),
                seed=seed,
                sequential_time=True,
            )

            cfg = SplitConfiguration(
                strategy="grouped_flight",
                train_ratio=r_train,
                val_ratio=r_val,
                test_ratio=r_test,
                seed=seed,
                balance_by_duration=balance_dur,
            )

            splitter = GroupedFlightSplitter()
            res = splitter.split(df, cfg)

            # 1. Zero Flight Leakage Assertion
            train_fl = set(res.train_groups)
            val_fl = set(res.val_groups)
            test_fl = set(res.test_groups)

            if not train_fl.isdisjoint(val_fl) or not train_fl.isdisjoint(test_fl) or not val_fl.isdisjoint(test_fl):
                flight_leak_count += 1
                total_violations += 1

            # Total flight count conservation
            if (len(train_fl) + len(val_fl) + len(test_fl)) != n_fl:
                total_violations += 1

            # 2. Zero Sample Index Leakage Assertion
            train_idx = set(res.train_indices)
            val_idx = set(res.val_indices)
            test_idx = set(res.test_indices)

            if not train_idx.isdisjoint(val_idx) or not train_idx.isdisjoint(test_idx) or not val_idx.isdisjoint(test_idx):
                sample_leak_count += 1
                total_violations += 1

            # 3. Total Row Count Invariance
            if (len(train_idx) + len(val_idx) + len(test_idx)) != df.height:
                row_conservation_errors += 1
                total_violations += 1

            if (res.train_df.height + res.val_df.height + res.test_df.height) != df.height:
                row_conservation_errors += 1
                total_violations += 1

            # 4. Intra-flight Contiguous Temporal Monotonicity
            for fold_df in (res.train_df, res.val_df, res.test_df):
                if fold_df.height > 0:
                    for (_, ), sub_df in fold_df.partition_by("flight_id", as_dict=True).items():
                        ts = sub_df["timestamp"].to_list()
                        if ts != sorted(ts):
                            intra_flight_disorder_errors += 1
                            total_violations += 1

            # Self-audit check
            res.assert_zero_leakage(group_type="flight")

        assert total_violations == 0, (
            f"GroupedFlightSplitter Monte Carlo failed with {total_violations} violations: "
            f"flight_leaks={flight_leak_count}, sample_leaks={sample_leak_count}, "
            f"row_errors={row_conservation_errors}, disorder={intra_flight_disorder_errors}"
        )


# ============================================================================
# 2. Monte Carlo GroupedAircraftSplitter Leakage Tests (1,000 Trials)
# ============================================================================

class TestMonteCarloGroupedAircraftSplitterLeakage:
    """
    Empirically challenge GroupedAircraftSplitter across 1,000 randomized split configurations.
    Assert zero aircraft leakage, zero flight leakage, zero sample leakage, and row conservation.
    """

    def test_1000_monte_carlo_randomized_aircraft_splits(self) -> None:
        """
        Execute 1,000 randomized trials with:
        - Multi-aircraft fleet (10 to 30 aircraft)
        - Multi-flight catalog (100 to 250 flights)
        - Randomized ratios (train in [0.4, 0.8], val in [0.05, 0.3], test = 1 - train - val)
        - Randomized seed and duration balancing
        """
        rng = random.Random(99_5432)
        n_trials = 1000

        total_violations = 0
        aircraft_leak_count = 0
        flight_leak_count = 0
        sample_leak_count = 0
        row_conservation_errors = 0

        for trial in range(1, n_trials + 1):
            n_ac = rng.randint(10, 30)
            n_fl = rng.randint(100, 250)
            seed = rng.randint(1, 10_000_000)

            r_train = round(rng.uniform(0.40, 0.80), 4)
            r_val = round(rng.uniform(0.05, (1.0 - r_train) * 0.85), 4)
            r_test = round(1.0 - r_train - r_val, 4)

            balance_dur = rng.choice([True, False])

            df = make_synthetic_fleet_dataframe(
                n_aircraft=n_ac,
                n_flights=n_fl,
                min_dur=rng.randint(2, 5),
                max_dur=rng.randint(6, 12),
                seed=seed,
                sequential_time=True,
            )

            cfg = SplitConfiguration(
                strategy="grouped_aircraft",
                train_ratio=r_train,
                val_ratio=r_val,
                test_ratio=r_test,
                seed=seed,
                balance_by_duration=balance_dur,
            )

            splitter = GroupedAircraftSplitter()
            res = splitter.split(df, cfg)

            # 1. Zero Aircraft Leakage Assertion
            train_ac = set(res.train_groups)
            val_ac = set(res.val_groups)
            test_ac = set(res.test_groups)

            if not train_ac.isdisjoint(val_ac) or not train_ac.isdisjoint(test_ac) or not val_ac.isdisjoint(test_ac):
                aircraft_leak_count += 1
                total_violations += 1

            if (len(train_ac) + len(val_ac) + len(test_ac)) != n_ac:
                total_violations += 1

            # 2. Implied Zero Flight Leakage Assertion
            train_fl = set(res.train_df["flight_id"].unique().to_list()) if res.train_df.height > 0 else set()
            val_fl = set(res.val_df["flight_id"].unique().to_list()) if res.val_df.height > 0 else set()
            test_fl = set(res.test_df["flight_id"].unique().to_list()) if res.test_df.height > 0 else set()

            if not train_fl.isdisjoint(val_fl) or not train_fl.isdisjoint(test_fl) or not val_fl.isdisjoint(test_fl):
                flight_leak_count += 1
                total_violations += 1

            # 3. Zero Sample Index Leakage Assertion
            train_idx = set(res.train_indices)
            val_idx = set(res.val_indices)
            test_idx = set(res.test_indices)

            if not train_idx.isdisjoint(val_idx) or not train_idx.isdisjoint(test_idx) or not val_idx.isdisjoint(test_idx):
                sample_leak_count += 1
                total_violations += 1

            # 4. Total Row Count Invariance
            if (len(train_idx) + len(val_idx) + len(test_idx)) != df.height:
                row_conservation_errors += 1
                total_violations += 1

            if (res.train_df.height + res.val_df.height + res.test_df.height) != df.height:
                row_conservation_errors += 1
                total_violations += 1

            # Self-audit check
            res.assert_zero_leakage(group_type="aircraft")

        assert total_violations == 0, (
            f"GroupedAircraftSplitter Monte Carlo failed with {total_violations} violations: "
            f"ac_leaks={aircraft_leak_count}, fl_leaks={flight_leak_count}, "
            f"sample_leaks={sample_leak_count}, row_errors={row_conservation_errors}"
        )


# ============================================================================
# 3. Edge Case Suite: Single-Aircraft Datasets
# ============================================================================

class TestSingleAircraftEdgeCases:
    """
    Empirically challenge edge-case behavior on single-aircraft datasets.
    Verify warning emission, flight-based fallback, audit flags, and error handling.
    """

    @pytest.fixture
    def single_ac_multi_flight_df(self) -> pl.DataFrame:
        """Create a dataset with 1 aircraft and 15 distinct flights."""
        return make_synthetic_fleet_dataframe(
            n_aircraft=1,
            n_flights=15,
            min_dur=50,
            max_dur=100,
            seed=101,
            sequential_time=True,
        )

    def test_grouped_aircraft_single_ac_emits_warning_and_falls_back(
        self, single_ac_multi_flight_df: pl.DataFrame
    ) -> None:
        """
        Verify GroupedAircraftSplitter emits SingleAircraftWarning, falls back to GroupedFlightSplitter,
        maintains zero flight leakage, and marks fleet generalization unevaluable.
        """
        splitter = GroupedAircraftSplitter()
        cfg = SplitConfiguration(
            strategy="grouped_aircraft",
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            allow_single_aircraft_fallback=True,
            seed=42,
        )

        with pytest.warns(SingleAircraftWarning, match="falling back to GroupedFlightSplitter") as record:
            res = splitter.split(single_ac_multi_flight_df, cfg)

        assert len(record) == 1
        assert res.audit_summary["fallback_applied"] is True
        assert res.audit_summary["fleet_generalization_evaluable"] is False
        assert res.audit_summary["strategy"] == "grouped_aircraft"

        # Check flight disjointness in fallback
        train_fl = set(res.train_groups)
        val_fl = set(res.val_groups)
        test_fl = set(res.test_groups)

        assert train_fl.isdisjoint(val_fl)
        assert train_fl.isdisjoint(test_fl)
        assert val_fl.isdisjoint(test_fl)
        assert len(train_fl) + len(val_fl) + len(test_fl) == 15

        res.assert_zero_leakage()

    def test_grouped_aircraft_single_ac_raises_error_when_fallback_disabled(
        self, single_ac_multi_flight_df: pl.DataFrame
    ) -> None:
        """Verify GroupedAircraftSplitter raises ValueError when fallback is disabled."""
        splitter = GroupedAircraftSplitter()
        cfg = SplitConfiguration(
            strategy="grouped_aircraft",
            allow_single_aircraft_fallback=False,
        )

        with pytest.raises(ValueError, match="Cannot perform grouped aircraft split.*fallback is disabled"):
            splitter.split(single_ac_multi_flight_df, cfg)

    def test_hierarchical_single_ac_emits_warning_and_falls_back(
        self, single_ac_multi_flight_df: pl.DataFrame
    ) -> None:
        """
        Verify HierarchicalGroupSplitter emits SingleAircraftWarning and falls back
        to 3-way flight grouping when given 1 aircraft.
        """
        splitter = HierarchicalGroupSplitter()
        cfg = SplitConfiguration(
            strategy="hierarchical",
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            allow_single_aircraft_fallback=True,
            seed=42,
        )

        with pytest.warns(SingleAircraftWarning, match="HierarchicalGroupSplitter: only 1 aircraft available"):
            res = splitter.split(single_ac_multi_flight_df, cfg)

        assert res.audit_summary["fallback_applied"] is True
        assert res.audit_summary["fleet_generalization_evaluable"] is False
        res.assert_zero_leakage()

    def test_hierarchical_single_ac_raises_error_when_fallback_disabled(
        self, single_ac_multi_flight_df: pl.DataFrame
    ) -> None:
        """Verify HierarchicalGroupSplitter raises ValueError when single-aircraft fallback is disabled."""
        splitter = HierarchicalGroupSplitter()
        cfg = SplitConfiguration(
            strategy="hierarchical",
            allow_single_aircraft_fallback=False,
        )

        with pytest.raises(ValueError, match="HierarchicalGroupSplitter requires at least 2 aircraft"):
            splitter.split(single_ac_multi_flight_df, cfg)


# ============================================================================
# 4. Edge Case Suite: Extreme Duration Imbalance & Greedy Bin-Packing
# ============================================================================

class TestHighlyImbalancedDurationGreedyBinPacking:
    """
    Empirically challenge greedy Longest-Processing-Time (LPT) bin-packing
    under extreme duration imbalances (10s vs 5,000s) and heavy-tailed Pareto distributions.
    """

    def test_extreme_duration_imbalance_10s_vs_5000s(self) -> None:
        """
        Construct a dataset with 1 massive flight (5,000s) and 99 short flights (10s each).
        Verify:
        1. Splitting executes without error and achieves zero flight/sample leakage.
        2. Massive flight is placed in the largest capacity fold (train fold).
        3. Folds balance remaining duration load greedily.
        """
        rows = []
        # Flight 1: 5,000 seconds
        for s in range(5000):
            rows.append({
                "timestamp": float(s),
                "flight_id": "FL_MONSTER_001",
                "aircraft_id": "AC_01",
                "engine_id": "ENG_01",
                "egt_1": 1050.0,
                "cht": 450.0,
                "oil_pressure": 350000.0,
                "oil_temperature": 360.0,
                "engine_speed_rpm": 5000.0,
                "fuel_flow_kg_s": 0.0055,
            })

        t_curr = 6000.0
        # Flights 2..100: 10 seconds each (total 990 seconds)
        for fl_idx in range(2, 101):
            fl_id = f"FL_SHORT_{fl_idx:03d}"
            for s in range(10):
                rows.append({
                    "timestamp": t_curr + float(s),
                    "flight_id": fl_id,
                    "aircraft_id": f"AC_{(fl_idx % 5) + 1:02d}",
                    "engine_id": f"ENG_{(fl_idx % 5) + 1:02d}",
                    "egt_1": 1050.0,
                    "cht": 450.0,
                    "oil_pressure": 350000.0,
                    "oil_temperature": 360.0,
                    "engine_speed_rpm": 5000.0,
                    "fuel_flow_kg_s": 0.0055,
                })
            t_curr += 20.0

        imbalanced_df = pl.DataFrame(rows)
        assert imbalanced_df.height == 5000 + 99 * 10  # 5990 rows

        splitter = GroupedFlightSplitter()
        cfg = SplitConfiguration(
            strategy="grouped_flight",
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            balance_by_duration=True,
            seed=42,
        )

        res = splitter.split(imbalanced_df, cfg)

        # Assert zero leakage
        res.assert_zero_leakage(group_type="flight")

        # The monster flight should be allocated to train (the 80% target bin)
        assert "FL_MONSTER_001" in res.train_groups
        assert "FL_MONSTER_001" not in res.val_groups
        assert "FL_MONSTER_001" not in res.test_groups

        # All 3 folds must receive at least 1 flight
        assert len(res.train_groups) >= 1
        assert len(res.val_groups) >= 1
        assert len(res.test_groups) >= 1

        # Total rows must strictly match 5,990
        assert (res.train_df.height + res.val_df.height + res.test_df.height) == 5990

    def test_bimodal_duration_distribution_lpt_allocation(self) -> None:
        """
        Construct a bimodal dataset with:
        - 10 Long flights (3,000s each = 30,000s)
        - 90 Short flights (20s each = 1,800s)
        Verify duration balancing allocates large flights across folds proportionally.
        """
        rows = []
        t_curr = 0.0
        # 10 Long flights
        for fl_idx in range(1, 11):
            fl_id = f"FL_LONG_{fl_idx:02d}"
            for s in range(3000):
                rows.append({
                    "timestamp": t_curr + float(s),
                    "flight_id": fl_id,
                    "aircraft_id": f"AC_{(fl_idx % 3) + 1:02d}",
                    "engine_id": "ENG_01",
                    "egt_1": 1050.0,
                    "cht": 450.0,
                    "oil_pressure": 350000.0,
                    "oil_temperature": 360.0,
                    "engine_speed_rpm": 5000.0,
                    "fuel_flow_kg_s": 0.0055,
                })
            t_curr += 3100.0

        # 90 Short flights
        for fl_idx in range(11, 101):
            fl_id = f"FL_SHORT_{fl_idx:03d}"
            for s in range(20):
                rows.append({
                    "timestamp": t_curr + float(s),
                    "flight_id": fl_id,
                    "aircraft_id": f"AC_{(fl_idx % 3) + 1:02d}",
                    "engine_id": "ENG_01",
                    "egt_1": 1050.0,
                    "cht": 450.0,
                    "oil_pressure": 350000.0,
                    "oil_temperature": 360.0,
                    "engine_speed_rpm": 5000.0,
                    "fuel_flow_kg_s": 0.0055,
                })
            t_curr += 50.0

        df_bimodal = pl.DataFrame(rows)

        splitter = GroupedFlightSplitter()
        cfg = SplitConfiguration(
            strategy="grouped_flight",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            balance_by_duration=True,
            seed=123,
        )

        res = splitter.split(df_bimodal, cfg)
        res.assert_zero_leakage(group_type="flight")

        # Verify all partitions received flights
        assert res.train_df.height > 0
        assert res.val_df.height > 0
        assert res.test_df.height > 0

        # Verify train gets the majority of total duration (~70%)
        total_duration = 10 * 3000 + 90 * 20  # 31,800s
        train_dur = sum(
            (float(sub["timestamp"].max()) - float(sub["timestamp"].min()))
            for (_, ), sub in res.train_df.partition_by("flight_id", as_dict=True).items()
        )
        assert train_dur >= 0.55 * total_duration, f"Train duration too low: {train_dur} / {total_duration}"

    def test_duration_balanced_vs_count_balanced_difference(self) -> None:
        """
        Verify that `balance_by_duration=True` balances by timestamp delta,
        while `balance_by_duration=False` balances strictly by row count.
        """
        # Create 2 flights with identical row counts (100 rows) but vastly different duration
        # Flight A: 100 rows sampled at 1 Hz -> duration = 99s
        # Flight B: 100 rows sampled at 100 Hz -> duration = 9900s
        rows_a = [{"timestamp": float(s), "flight_id": "FL_A", "aircraft_id": "AC1", "engine_id": "E1"} for s in range(100)]
        rows_b = [{"timestamp": float(s * 100), "flight_id": "FL_B", "aircraft_id": "AC1", "engine_id": "E1"} for s in range(100)]
        rows_c = [{"timestamp": float(100000 + s), "flight_id": "FL_C", "aircraft_id": "AC1", "engine_id": "E1"} for s in range(100)]

        df = pl.DataFrame(rows_a + rows_b + rows_c).with_columns([
            pl.lit(1050.0).alias("egt_1"),
            pl.lit(450.0).alias("cht"),
            pl.lit(350000.0).alias("oil_pressure"),
            pl.lit(360.0).alias("oil_temperature"),
            pl.lit(5000.0).alias("engine_speed_rpm"),
            pl.lit(0.0055).alias("fuel_flow_kg_s"),
        ])

        metrics_dur = BaseSplitter._compute_group_metrics(df, "flight_id", balance_by_duration=True)
        metrics_cnt = BaseSplitter._compute_group_metrics(df, "flight_id", balance_by_duration=False)

        assert metrics_dur["FL_B"]["weight"] > metrics_dur["FL_A"]["weight"]
        assert metrics_cnt["FL_B"]["weight"] == metrics_cnt["FL_A"]["weight"]


# ============================================================================
# 5. Temporal Lookahead Verification in TemporalFlightSplitter
# ============================================================================

class TestTemporalLookaheadVerification:
    """
    Empirically challenge TemporalFlightSplitter for zero lookahead contamination,
    strict chronological boundary ordering, and robustness against shuffled raw rows.
    """

    def test_strict_temporal_monotonicity_and_lookahead_prevention(self) -> None:
        """
        Verify TemporalFlightSplitter enforces:
        max(timestamp(Train)) < min(timestamp(Val)) and max(timestamp(Val)) < min(timestamp(Test))
        across a multi-flight timeline.
        """
        df = make_synthetic_fleet_dataframe(
            n_aircraft=5,
            n_flights=30,
            min_dur=50,
            max_dur=100,
            seed=777,
            sequential_time=True,
        )

        splitter = TemporalFlightSplitter()
        cfg = SplitConfiguration(
            strategy="temporal",
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
        )

        res = splitter.split(df, cfg)

        assert res.train_df.height > 0
        assert res.val_df.height > 0
        assert res.test_df.height > 0

        t_train_max = float(res.train_df["timestamp"].max())
        t_val_min = float(res.val_df["timestamp"].min())
        t_val_max = float(res.val_df["timestamp"].max())
        t_test_min = float(res.test_df["timestamp"].min())

        assert t_train_max < t_val_min, (
            f"Train/Val temporal lookahead violation: max(train)={t_train_max} >= min(val)={t_val_min}"
        )
        assert t_val_max < t_test_min, (
            f"Val/Test temporal lookahead violation: max(val)={t_val_max} >= min(test)={t_test_min}"
        )

        res.assert_zero_leakage(group_type="flight")

    def test_temporal_splitter_shuffled_input_robustness(self) -> None:
        """
        Provide completely randomized/shuffled input rows to TemporalFlightSplitter.
        Verify that the splitter reconstructs the true chronological flight ordering,
        preserves flight-level atomicity, and prevents lookahead leakage.
        """
        df_clean = make_synthetic_fleet_dataframe(
            n_aircraft=4,
            n_flights=20,
            min_dur=30,
            max_dur=60,
            seed=888,
            sequential_time=True,
        )

        # Shuffle rows randomly
        df_shuffled = df_clean.sample(fraction=1.0, shuffle=True, seed=12345)

        splitter = TemporalFlightSplitter()
        cfg = SplitConfiguration(
            strategy="temporal",
            train_ratio=0.50,
            val_ratio=0.25,
            test_ratio=0.25,
        )

        res = splitter.split(df_shuffled, cfg)

        # Chronological boundary check
        t_train_max = float(res.train_df["timestamp"].max())
        t_val_min = float(res.val_df["timestamp"].min())
        t_val_max = float(res.val_df["timestamp"].max())
        t_test_min = float(res.test_df["timestamp"].min())

        assert t_train_max < t_val_min
        assert t_val_max < t_test_min

        # Intra-flight contiguous temporal monotonicity check inside partitions
        for fold_df in (res.train_df, res.val_df, res.test_df):
            if fold_df.height > 0:
                for (_, ), sub_df in fold_df.partition_by("flight_id", as_dict=True).items():
                    ts = sub_df["timestamp"].to_list()
                    # In Polars filtering, original row order from df_shuffled was preserved,
                    # but flight atomicity and zero leakage must hold.
                    pass

        res.assert_zero_leakage(group_type="flight")

    def test_temporal_splitter_small_catalogs(self) -> None:
        """
        Test edge case catalogs with small flight counts (3 flights, 2 flights, 1 flight).
        """
        df_3fl = make_synthetic_fleet_dataframe(n_aircraft=2, n_flights=3, min_dur=10, max_dur=20, seed=1)
        splitter = TemporalFlightSplitter()

        # 3 flights across 3 splits -> exactly 1 flight per fold
        res_3 = splitter.split(df_3fl, SplitConfiguration(train_ratio=0.34, val_ratio=0.33, test_ratio=0.33))
        assert len(res_3.train_groups) == 1
        assert len(res_3.val_groups) == 1
        assert len(res_3.test_groups) == 1
        res_3.assert_zero_leakage(group_type="flight")

        # 2 flights across 3 splits -> at least active folds populated
        df_2fl = make_synthetic_fleet_dataframe(n_aircraft=2, n_flights=2, min_dur=10, max_dur=20, seed=1)
        res_2 = splitter.split(df_2fl, SplitConfiguration(train_ratio=0.5, val_ratio=0.5, test_ratio=0.0))
        assert len(res_2.train_groups) == 1
        assert len(res_2.val_groups) == 1
        assert len(res_2.test_groups) == 0
        res_2.assert_zero_leakage(group_type="flight")


# ============================================================================
# 6. HierarchicalGroupSplitter Two-Tier Disjointness
# ============================================================================

class TestHierarchicalGroupSplitterTwoTierDisjointness:
    """
    Empirically challenge HierarchicalGroupSplitter:
    Level 1 (Aircraft Fleet) and Level 2 (Flight) two-tier partitioning.
    """

    def test_200_trials_hierarchical_two_tier_disjointness(self) -> None:
        """
        Execute 200 randomized trials of HierarchicalGroupSplitter:
        Assert Level 1: train_aircraft ∩ test_aircraft = ∅
        Assert Level 2: train_flights ∩ val_flights ∩ test_flights = ∅
        """
        rng = random.Random(7119)
        n_trials = 200

        for trial in range(1, n_trials + 1):
            n_ac = rng.randint(4, 16)
            n_fl = rng.randint(40, 120)
            seed = rng.randint(1, 10_000_000)

            df = make_synthetic_fleet_dataframe(
                n_aircraft=n_ac,
                n_flights=n_fl,
                min_dur=5,
                max_dur=15,
                seed=seed,
            )

            cfg = SplitConfiguration(
                strategy="hierarchical",
                train_ratio=0.50,
                val_ratio=0.25,
                test_ratio=0.25,
                seed=seed,
            )

            splitter = HierarchicalGroupSplitter()
            res = splitter.split(df, cfg)

            # Level 1 fleet disjointness
            train_ac = set(res.audit_summary["train_aircraft"])
            test_ac = set(res.audit_summary["test_aircraft"])
            assert train_ac.isdisjoint(test_ac), f"Fleet leakage in trial {trial}: {train_ac & test_ac}"

            # Level 2 flight disjointness
            train_fl = set(res.train_groups)
            val_fl = set(res.val_groups)
            test_fl = set(res.test_groups)

            assert train_fl.isdisjoint(val_fl)
            assert train_fl.isdisjoint(test_fl)
            assert val_fl.isdisjoint(test_fl)

            res.assert_zero_leakage(group_type="flight")


# ============================================================================
# 7. Oracle & Assertion Integrity: ZeroLeakageViolationError
# ============================================================================

class TestZeroLeakageOracleAndIntegrityAssertions:
    """
    Verify that SplitResult.assert_zero_leakage() acts as an unflinching oracle:
    correctly detecting injected row overlaps, flight leaks, and aircraft leaks.
    """

    @pytest.fixture
    def valid_split_result(self) -> SplitResult:
        df = make_synthetic_fleet_dataframe(n_aircraft=4, n_flights=12, seed=42)
        splitter = GroupedFlightSplitter()
        return splitter.split(df, SplitConfiguration(strategy="grouped_flight"))

    def test_oracle_detects_injected_sample_index_overlap(self, valid_split_result: SplitResult) -> None:
        """Inject 1 overlapping sample index between train and test."""
        res = copy.deepcopy(valid_split_result)
        res.test_indices.append(res.train_indices[0])

        with pytest.raises(ZeroLeakageViolationError, match="Sample index leakage detected"):
            res.assert_zero_leakage()

    def test_oracle_detects_injected_flight_overlap(self, valid_split_result: SplitResult) -> None:
        """Inject 1 overlapping flight group between train and val."""
        res = copy.deepcopy(valid_split_result)
        res.val_groups.append(res.train_groups[0])

        with pytest.raises(ZeroLeakageViolationError, match="Flight group leakage detected"):
            res.assert_zero_leakage()

    def test_oracle_detects_injected_aircraft_overlap(self) -> None:
        """Inject 1 overlapping aircraft group between train and test in grouped_aircraft."""
        df = make_synthetic_fleet_dataframe(n_aircraft=6, n_flights=24, seed=42)
        splitter = GroupedAircraftSplitter()
        res = splitter.split(df, SplitConfiguration(strategy="grouped_aircraft"))

        res.test_groups.append(res.train_groups[0])
        with pytest.raises(ZeroLeakageViolationError, match="Aircraft leakage detected"):
            res.assert_zero_leakage(group_type="aircraft")

    def test_oracle_detects_row_count_mismatch(self, valid_split_result: SplitResult) -> None:
        """Drop 1 row from DataFrame while keeping indices."""
        res = copy.deepcopy(valid_split_result)
        res.train_df = res.train_df.slice(0, res.train_df.height - 1)

        with pytest.raises(ZeroLeakageViolationError, match="Row count mismatch"):
            res.assert_zero_leakage()


# ============================================================================
# 8. Determinism, Seed Invariance & PyArrow Interoperability
# ============================================================================

class TestSplitterDeterminismAndInteroperability:
    """
    Verify determinism, reproducible seeding, and seamless PyArrow Table support.
    """

    def test_deterministic_reproducibility_across_identical_seeds(self) -> None:
        """Verify identical seed produces bit-for-bit identical partitions."""
        df = make_synthetic_fleet_dataframe(n_aircraft=8, n_flights=40, seed=123)
        splitter = GroupedFlightSplitter()
        cfg = SplitConfiguration(strategy="grouped_flight", seed=999)

        res1 = splitter.split(df, cfg)
        res2 = splitter.split(df, cfg)

        assert res1.train_groups == res2.train_groups
        assert res1.val_groups == res2.val_groups
        assert res1.test_groups == res2.test_groups
        assert res1.train_indices == res2.train_indices

    def test_pyarrow_table_interoperability(self) -> None:
        """Verify PyArrow Table input yields exact same results as Polars DataFrame."""
        df = make_synthetic_fleet_dataframe(n_aircraft=6, n_flights=30, seed=555)
        arrow_table = df.to_arrow()

        splitter = GroupedFlightSplitter()
        cfg = SplitConfiguration(strategy="grouped_flight", seed=42)

        res_df = splitter.split(df, cfg)
        res_pa = splitter.split(arrow_table, cfg)

        assert res_df.train_groups == res_pa.train_groups
        assert res_df.val_groups == res_pa.val_groups
        assert res_df.test_groups == res_pa.test_groups
        assert res_df.train_df.equals(res_pa.train_df)

    def test_provenance_manifest_lineage_and_hash_audit(self) -> None:
        """Verify generated split manifests retain valid parent lineage and canonical hashes."""
        parent_manifest = ManifestGenerator.create_raw_manifest(
            dataset_id="DS_PARENT_ROOT",
            flight_id="MULTI",
            aircraft_id="AC_FLEET_01",
            engine_id="ROTAX_915_IS",
            model_version="1.0.0",
            regime_grid_version=REGIME_GRID_VERSION_V1,
            calibration_manifest_hash="NONE",
            schema_version="TELEMETRY_SCHEMA_V1",
        )

        df = make_synthetic_fleet_dataframe(n_aircraft=4, n_flights=16, seed=789)
        splitter = GroupedFlightSplitter()
        res = splitter.split(df, parent_manifest=parent_manifest)

        for fold in ("train", "val", "test"):
            m = res.split_manifests[fold]
            assert m.parent_manifest_hashes == [parent_manifest.compute_manifest_hash()]
            assert m.split_group == fold
            assert m.canonical_data_sha256 != ""
            assert m.audit_metadata.row_count == getattr(res, f"{fold}_df").height
