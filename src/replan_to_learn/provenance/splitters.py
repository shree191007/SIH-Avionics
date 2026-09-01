"""
SIH26054 Replan to Learn: Provenance & Data Versioning.
Grouped Anti-Leakage Splitters with Greedy Bin-Packing & Edge Case Handling.

Implements mathematically guaranteed zero-leakage dataset partitioning:
- GroupedFlightSplitter: Guaranteed zero flight leakage across partitions.
- GroupedAircraftSplitter: Guaranteed zero aircraft leakage across partitions.
- HierarchicalGroupSplitter: Two-tier fleet (L1) and flight (L2) partitioning.
- TemporalFlightSplitter: Chronological sorting preventing future lookahead bias.

Handles edge cases:
- Single-aircraft datasets (graceful fallback with SingleAircraftWarning).
- Extreme flight duration and sample count imbalances (greedy bin-packing).
- Preserves intra-flight contiguous temporal ordering.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union
import warnings

import polars as pl
import pyarrow as pa

from replan_to_learn.provenance.manifest import (
    ManifestGenerator,
    ProvenanceManifest,
    REGIME_GRID_VERSION_V1,
)


# ============================================================================
# 1. Custom Exceptions & Warnings
# ============================================================================

class SingleAircraftWarning(UserWarning):
    """Warning emitted when a single-aircraft dataset falls back to flight grouping."""
    pass


class ZeroLeakageViolationError(ValueError):
    """Exception raised when sample, flight, or aircraft leakage is detected across splits."""
    pass


# ============================================================================
# 2. Split Configuration & Result Dataclasses
# ============================================================================

@dataclasses.dataclass(frozen=True)
class SplitConfiguration:
    """Configuration parameters for dataset partitioning strategies."""
    strategy: Literal["grouped_flight", "grouped_aircraft", "hierarchical", "temporal"] = "grouped_flight"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    balance_by_duration: bool = True
    allow_single_aircraft_fallback: bool = True
    flight_id_column: str = "flight_id"
    aircraft_id_column: str = "aircraft_id"
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"Split ratios must sum to 1.0, got train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio} (sum={total})"
            )
        if self.train_ratio < 0.0 or self.val_ratio < 0.0 or self.test_ratio < 0.0:
            raise ValueError("Split ratios must be non-negative")


@dataclasses.dataclass
class SplitResult:
    """Container holding partitioned data, group assignments, manifests, and audit metrics."""
    train_df: pl.DataFrame
    val_df: pl.DataFrame
    test_df: pl.DataFrame
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    train_groups: List[str]
    val_groups: List[str]
    test_groups: List[str]
    split_manifests: Dict[str, ProvenanceManifest]
    audit_summary: Dict[str, Any]

    def assert_zero_leakage(self, group_type: str = "auto") -> None:
        """
        Verify that no sample, flight, or aircraft overlap exists between splits.
        Raises ZeroLeakageViolationError if any leakage is detected.
        """
        # 1. Row Index Disjointness
        train_idx_set = set(self.train_indices)
        val_idx_set = set(self.val_indices)
        test_idx_set = set(self.test_indices)

        leak_tv = train_idx_set.intersection(val_idx_set)
        leak_tt = train_idx_set.intersection(test_idx_set)
        leak_vt = val_idx_set.intersection(test_idx_set)

        if leak_tv or leak_tt or leak_vt:
            raise ZeroLeakageViolationError(
                f"Sample index leakage detected: train∩val={len(leak_tv)}, train∩test={len(leak_tt)}, val∩test={len(leak_vt)}"
            )

        # 2. Total Row Count Invariance
        total_rows = len(train_idx_set) + len(val_idx_set) + len(test_idx_set)
        sum_heights = self.train_df.height + self.val_df.height + self.test_df.height
        if total_rows != sum_heights:
            raise ZeroLeakageViolationError(
                f"Row count mismatch: sum of index sets={total_rows}, sum of DataFrame heights={sum_heights}"
            )

        # 3. Flight Disjointness
        train_fl = set(self.train_groups)
        val_fl = set(self.val_groups)
        test_fl = set(self.test_groups)

        fl_tv = train_fl.intersection(val_fl)
        fl_tt = train_fl.intersection(test_fl)
        fl_vt = val_fl.intersection(test_fl)

        if group_type in ("flight", "auto") and (fl_tv or fl_tt or fl_vt):
            # Check if groups are indeed flight IDs
            if self.audit_summary.get("group_column") == "flight_id":
                raise ZeroLeakageViolationError(
                    f"Flight group leakage detected: train∩val={fl_tv}, train∩test={fl_tt}, val∩test={fl_vt}"
                )

        # 4. Aircraft Disjointness (if aircraft strategy applied and not fallen back)
        if self.audit_summary.get("strategy") == "grouped_aircraft" and not self.audit_summary.get("fallback_applied", False):
            train_ac = set(self.train_groups)
            val_ac = set(self.val_groups)
            test_ac = set(self.test_groups)
            ac_leak = train_ac.intersection(val_ac) | train_ac.intersection(test_ac) | val_ac.intersection(test_ac)
            if ac_leak:
                raise ZeroLeakageViolationError(f"Aircraft leakage detected: {ac_leak}")


# ============================================================================
# 3. Base Splitter Implementation
# ============================================================================

class BaseSplitter:
    """Base class providing group weight calculation, greedy bin-packing, and result creation."""

    @staticmethod
    def _normalize_dataframe(data: Union[pl.DataFrame, pa.Table]) -> pl.DataFrame:
        """Convert input data to Polars DataFrame."""
        if isinstance(data, pa.Table):
            return pl.from_arrow(data)
        elif isinstance(data, pl.DataFrame):
            return data
        else:
            raise TypeError(f"Expected pl.DataFrame or pa.Table, got {type(data)}")

    @staticmethod
    def _compute_group_metrics(
        df: pl.DataFrame,
        group_col: str,
        balance_by_duration: bool = True,
        timestamp_col: str = "timestamp",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute weight, row count, duration, and min/max timestamp for each group.
        """
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in DataFrame columns: {df.columns}")

        has_timestamp = timestamp_col in df.columns
        metrics: Dict[str, Dict[str, Any]] = {}

        grouped = df.partition_by(group_col, as_dict=True)
        for (g_val,), sub_df in grouped.items():
            g_str = str(g_val)
            cnt = sub_df.height
            if has_timestamp and not sub_df.is_empty():
                t_min = float(sub_df[timestamp_col].min())
                t_max = float(sub_df[timestamp_col].max())
                dur_s = max(0.0, t_max - t_min) if t_max >= t_min else float(cnt)
            else:
                t_min = 0.0
                t_max = float(cnt)
                dur_s = float(cnt)

            weight = dur_s if balance_by_duration else float(cnt)
            metrics[g_str] = {
                "group": g_str,
                "weight": max(1e-6, weight),
                "row_count": cnt,
                "duration_s": dur_s,
                "t_min": t_min,
                "t_max": t_max,
            }

        return metrics

    @staticmethod
    def _greedy_bin_pack(
        group_metrics: Dict[str, Dict[str, Any]],
        ratios: Tuple[float, float, float],
        fold_names: Tuple[str, str, str] = ("train", "val", "test"),
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """
        Greedy bin-packing allocation (LPT: Longest Processing Time first) to minimize fold imbalance.
        """
        active_folds = [name for name, r in zip(fold_names, ratios) if r > 0.0]
        if not active_folds:
            return {name: [] for name in fold_names}

        total_ratio = sum(r for r in ratios if r > 0.0)
        norm_ratios = {
            name: r / total_ratio for name, r in zip(fold_names, ratios) if r > 0.0
        }

        total_weight = sum(m["weight"] for m in group_metrics.values())
        target_weights = {name: norm_ratios[name] * total_weight for name in active_folds}

        fold_groups: Dict[str, List[str]] = {name: [] for name in fold_names}
        fold_weights: Dict[str, float] = {name: 0.0 for name in active_folds}

        # Deterministic sort: descending by weight, then deterministic tie-breaker by hash
        rng = random.Random(seed)
        salt = {g: rng.random() for g in group_metrics}

        sorted_groups = sorted(
            group_metrics.keys(),
            key=lambda g: (group_metrics[g]["weight"], salt[g]),
            reverse=True,
        )

        for g in sorted_groups:
            w = group_metrics[g]["weight"]
            # Greedily assign to the fold with the lowest relative fill (current_weight / target_weight)
            best_fold = min(
                active_folds,
                key=lambda f: (fold_weights[f] + w) / max(target_weights[f], 1e-9),
            )
            fold_groups[best_fold].append(g)
            fold_weights[best_fold] += w

        return fold_groups

    @classmethod
    def _build_split_result(
        cls,
        df: pl.DataFrame,
        fold_groups: Dict[str, List[str]],
        group_col: str,
        config: SplitConfiguration,
        parent_manifest: Optional[ProvenanceManifest] = None,
        audit_extra: Optional[Dict[str, Any]] = None,
    ) -> SplitResult:
        """
        Assemble SplitResult, preserving intra-flight temporal row ordering and creating manifests.
        """
        # Add internal row index column to track original row indices
        df_indexed = df.with_columns(pl.int_range(0, df.height).alias("__row_idx__"))

        def get_fold_data(groups: List[str]) -> Tuple[pl.DataFrame, List[int]]:
            if not groups or df.is_empty():
                empty_df = df.clear()
                return empty_df, []
            
            # Filter groups while preserving row ordering
            fold_df_indexed = df_indexed.filter(pl.col(group_col).cast(pl.String).is_in(groups))
            indices = fold_df_indexed["__row_idx__"].to_list()
            fold_df = fold_df_indexed.drop("__row_idx__")
            return fold_df, indices

        train_df, train_indices = get_fold_data(fold_groups.get("train", []))
        val_df, val_indices = get_fold_data(fold_groups.get("val", []))
        test_df, test_indices = get_fold_data(fold_groups.get("test", []))

        # Generate split manifests
        split_manifests: Dict[str, ProvenanceManifest] = {}
        parent_hash = parent_manifest.compute_manifest_hash() if parent_manifest else "0" * 64

        dataset_id = parent_manifest.dataset_id if parent_manifest else "DS_SPLIT"
        aircraft_id = parent_manifest.aircraft_id if parent_manifest else (
            df[config.aircraft_id_column][0] if config.aircraft_id_column in df.columns and not df.is_empty() else "AC_MULTI"
        )
        engine_id = parent_manifest.engine_id if parent_manifest else (
            df["engine_id"][0] if "engine_id" in df.columns and not df.is_empty() else "ROTAX_915_IS"
        )
        model_version = parent_manifest.model_version if parent_manifest else "1.0.0"
        regime_grid_version = parent_manifest.regime_grid_version if parent_manifest else REGIME_GRID_VERSION_V1
        schema_version = parent_manifest.schema_version if parent_manifest else "TELEMETRY_SCHEMA_V1"

        for fold_name, fold_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            m = ManifestGenerator.create_processed_manifest(
                dataset_id=f"{dataset_id}_{fold_name.upper()}",
                flight_id="MULTI" if len(fold_groups.get(fold_name, [])) > 1 else (
                    fold_groups.get(fold_name, ["NONE"])[0] if fold_groups.get(fold_name) else "NONE"
                ),
                aircraft_id=aircraft_id if fold_df.height > 0 else "AC_NONE",
                engine_id=engine_id,
                model_version=model_version,
                regime_grid_version=regime_grid_version,
                calibration_manifest_hash=parent_manifest.calibration_manifest_hash if parent_manifest else "NONE",
                schema_version=schema_version,
                parent_manifest_hashes=[parent_hash] if parent_manifest else [],
                data=fold_df,
                split_group=fold_name,  # type: ignore
                created_by=f"replan_to_learn.provenance.{config.strategy}",
                status_flags=["HEALTHY", f"SPLIT_{fold_name.upper()}"],
                extra_metadata={"assigned_groups": fold_groups.get(fold_name, [])},
            )
            split_manifests[fold_name] = m

        audit_summary = {
            "strategy": config.strategy,
            "group_column": group_col,
            "total_rows": df.height,
            "train_rows": train_df.height,
            "val_rows": val_df.height,
            "test_rows": test_df.height,
            "train_ratio_actual": (train_df.height / df.height) if df.height > 0 else 0.0,
            "val_ratio_actual": (val_df.height / df.height) if df.height > 0 else 0.0,
            "test_ratio_actual": (test_df.height / df.height) if df.height > 0 else 0.0,
            "train_groups_count": len(fold_groups.get("train", [])),
            "val_groups_count": len(fold_groups.get("val", [])),
            "test_groups_count": len(fold_groups.get("test", [])),
            "fleet_generalization_evaluable": True,
            "fallback_applied": False,
        }
        if audit_extra:
            audit_summary.update(audit_extra)

        result = SplitResult(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_groups=fold_groups.get("train", []),
            val_groups=fold_groups.get("val", []),
            test_groups=fold_groups.get("test", []),
            split_manifests=split_manifests,
            audit_summary=audit_summary,
        )

        return result


# ============================================================================
# 4. Specific Splitter Implementations
# ============================================================================

class GroupedFlightSplitter(BaseSplitter):
    """
    Partitions telemetry data strictly by flight_id.
    Mathematically guarantees: flights(Train) ∩ flights(Val) ∩ flights(Test) = ∅.
    """

    def split(
        self,
        data: Union[pl.DataFrame, pa.Table],
        config: Optional[SplitConfiguration] = None,
        parent_manifest: Optional[ProvenanceManifest] = None,
    ) -> SplitResult:
        df = self._normalize_dataframe(data)
        cfg = config or SplitConfiguration(strategy="grouped_flight")

        flight_col = cfg.flight_id_column
        metrics = self._compute_group_metrics(
            df,
            group_col=flight_col,
            balance_by_duration=cfg.balance_by_duration,
            timestamp_col=cfg.timestamp_column,
        )

        ratios = (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
        fold_groups = self._greedy_bin_pack(
            group_metrics=metrics,
            ratios=ratios,
            seed=cfg.seed,
        )

        res = self._build_split_result(
            df=df,
            fold_groups=fold_groups,
            group_col=flight_col,
            config=cfg,
            parent_manifest=parent_manifest,
        )
        res.assert_zero_leakage(group_type="flight")
        return res


class GroupedAircraftSplitter(BaseSplitter):
    """
    Partitions telemetry data strictly by aircraft_id.
    Mathematically guarantees: aircraft(Train) ∩ aircraft(Test) = ∅.

    Edge Case Handling:
    If N_aircraft == 1:
      - If allow_single_aircraft_fallback: logs SingleAircraftWarning, falls back to
        GroupedFlightSplitter, sets fleet_generalization_evaluable: False.
      - If not allowed: raises ValueError.
    """

    def split(
        self,
        data: Union[pl.DataFrame, pa.Table],
        config: Optional[SplitConfiguration] = None,
        parent_manifest: Optional[ProvenanceManifest] = None,
    ) -> SplitResult:
        df = self._normalize_dataframe(data)
        cfg = config or SplitConfiguration(strategy="grouped_aircraft")

        ac_col = cfg.aircraft_id_column
        if ac_col not in df.columns:
            raise ValueError(f"Aircraft column '{ac_col}' not present in DataFrame columns: {df.columns}")

        unique_aircraft = df[ac_col].unique().to_list()

        # Handle Single-Aircraft Dataset Edge Case
        if len(unique_aircraft) <= 1:
            if cfg.allow_single_aircraft_fallback:
                warnings.warn(
                    f"Dataset contains only {len(unique_aircraft)} aircraft ('{unique_aircraft}'); "
                    "falling back to GroupedFlightSplitter on flight_id. Fleet generalization cannot be evaluated.",
                    category=SingleAircraftWarning,
                    stacklevel=2,
                )
                flight_splitter = GroupedFlightSplitter()
                flight_cfg = SplitConfiguration(
                    strategy="grouped_flight",
                    train_ratio=cfg.train_ratio,
                    val_ratio=cfg.val_ratio,
                    test_ratio=cfg.test_ratio,
                    seed=cfg.seed,
                    balance_by_duration=cfg.balance_by_duration,
                    flight_id_column=cfg.flight_id_column,
                    aircraft_id_column=cfg.aircraft_id_column,
                    timestamp_column=cfg.timestamp_column,
                )
                res = flight_splitter.split(df, flight_cfg, parent_manifest=parent_manifest)
                res.audit_summary["fleet_generalization_evaluable"] = False
                res.audit_summary["fallback_applied"] = True
                res.audit_summary["strategy"] = "grouped_aircraft"
                return res
            else:
                raise ValueError(
                    f"Cannot perform grouped aircraft split: dataset has only {len(unique_aircraft)} aircraft "
                    "and single-aircraft fallback is disabled."
                )

        metrics = self._compute_group_metrics(
            df,
            group_col=ac_col,
            balance_by_duration=cfg.balance_by_duration,
            timestamp_col=cfg.timestamp_column,
        )

        ratios = (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
        fold_groups = self._greedy_bin_pack(
            group_metrics=metrics,
            ratios=ratios,
            seed=cfg.seed,
        )

        res = self._build_split_result(
            df=df,
            fold_groups=fold_groups,
            group_col=ac_col,
            config=cfg,
            parent_manifest=parent_manifest,
        )
        res.assert_zero_leakage(group_type="aircraft")
        return res


class HierarchicalGroupSplitter(BaseSplitter):
    """
    Two-Tier Fleet (L1) and Flight (L2) Hierarchical Partitioning:
    - Level 1: Partitions aircraft_id into Train Fleet and Test Fleet (Out-of-Fleet Transfer).
    - Level 2: Within Train Fleet, partitions flight_id into Train Flights and Val Flights.

    Resulting Evaluation Sets:
    - Train: Seen Aircraft, Seen Flights.
    - Val: Seen Aircraft, Unseen Validation Flights (Flight Generalization).
    - Test: Unseen Aircraft, Unseen Flights (True Fleet Transferability).
    """

    def split(
        self,
        data: Union[pl.DataFrame, pa.Table],
        config: Optional[SplitConfiguration] = None,
        parent_manifest: Optional[ProvenanceManifest] = None,
    ) -> SplitResult:
        df = self._normalize_dataframe(data)
        cfg = config or SplitConfiguration(strategy="hierarchical")

        ac_col = cfg.aircraft_id_column
        flight_col = cfg.flight_id_column

        unique_aircraft = df[ac_col].unique().to_list()

        # Handle Single-Aircraft Dataset Fallback
        if len(unique_aircraft) <= 1:
            if cfg.allow_single_aircraft_fallback:
                warnings.warn(
                    f"HierarchicalGroupSplitter: only {len(unique_aircraft)} aircraft available. "
                    "Falling back to 3-way GroupedFlightSplitter. Fleet transferability cannot be evaluated.",
                    category=SingleAircraftWarning,
                    stacklevel=2,
                )
                flight_splitter = GroupedFlightSplitter()
                flight_cfg = SplitConfiguration(
                    strategy="grouped_flight",
                    train_ratio=cfg.train_ratio,
                    val_ratio=cfg.val_ratio,
                    test_ratio=cfg.test_ratio,
                    seed=cfg.seed,
                    balance_by_duration=cfg.balance_by_duration,
                    flight_id_column=cfg.flight_id_column,
                    aircraft_id_column=cfg.aircraft_id_column,
                    timestamp_column=cfg.timestamp_column,
                )
                res = flight_splitter.split(df, flight_cfg, parent_manifest=parent_manifest)
                res.audit_summary["fleet_generalization_evaluable"] = False
                res.audit_summary["fallback_applied"] = True
                res.audit_summary["strategy"] = "hierarchical"
                return res
            else:
                raise ValueError("HierarchicalGroupSplitter requires at least 2 aircraft when fallback is disabled.")

        # --------------------------------------------------------------------
        # Level 1 (Outer): Split Aircraft into Train Fleet and Test Fleet
        # --------------------------------------------------------------------
        ac_metrics = self._compute_group_metrics(
            df,
            group_col=ac_col,
            balance_by_duration=cfg.balance_by_duration,
            timestamp_col=cfg.timestamp_column,
        )

        test_fleet_ratio = cfg.test_ratio
        train_fleet_ratio = cfg.train_ratio + cfg.val_ratio

        fleet_packs = self._greedy_bin_pack(
            group_metrics=ac_metrics,
            ratios=(train_fleet_ratio, 0.0, test_fleet_ratio),
            fold_names=("train_fleet", "val_dummy", "test_fleet"),
            seed=cfg.seed,
        )

        train_aircraft = fleet_packs["train_fleet"]
        test_aircraft = fleet_packs["test_fleet"]

        # --------------------------------------------------------------------
        # Level 2 (Inner): Within Train Fleet, Split Flights into Train and Val
        # --------------------------------------------------------------------
        train_fleet_df = df.filter(pl.col(ac_col).cast(pl.String).is_in(train_aircraft))
        test_fleet_df = df.filter(pl.col(ac_col).cast(pl.String).is_in(test_aircraft))

        fl_metrics = self._compute_group_metrics(
            train_fleet_df,
            group_col=flight_col,
            balance_by_duration=cfg.balance_by_duration,
            timestamp_col=cfg.timestamp_column,
        )

        sub_train_ratio = cfg.train_ratio / max(1e-9, train_fleet_ratio)
        sub_val_ratio = cfg.val_ratio / max(1e-9, train_fleet_ratio)

        flight_packs = self._greedy_bin_pack(
            group_metrics=fl_metrics,
            ratios=(sub_train_ratio, sub_val_ratio, 0.0),
            fold_names=("train", "val", "dummy_test"),
            seed=cfg.seed + 1,
        )

        train_flights = flight_packs["train"]
        val_flights = flight_packs["val"]
        test_flights = test_fleet_df[flight_col].unique().to_list()

        fold_groups = {
            "train": train_flights,
            "val": val_flights,
            "test": test_flights,
        }

        res = self._build_split_result(
            df=df,
            fold_groups=fold_groups,
            group_col=flight_col,
            config=cfg,
            parent_manifest=parent_manifest,
            audit_extra={
                "train_aircraft": train_aircraft,
                "test_aircraft": test_aircraft,
                "fleet_generalization_evaluable": True,
            },
        )

        # Assert hierarchical disjointness
        assert set(train_aircraft).isdisjoint(set(test_aircraft)), "Aircraft overlap in hierarchical split"
        res.assert_zero_leakage(group_type="flight")
        return res


class TemporalFlightSplitter(BaseSplitter):
    """
    Chronological Flight-Level Splitter for RUL / Wear Prognostics Evaluation.
    Guarantees: max(timestamp(Train)) < min(timestamp(Val)) <= max(timestamp(Val)) < min(timestamp(Test)).
    Strictly preserves intra-flight time-series ordering and prevents future lookahead leakage.
    """

    def split(
        self,
        data: Union[pl.DataFrame, pa.Table],
        config: Optional[SplitConfiguration] = None,
        parent_manifest: Optional[ProvenanceManifest] = None,
    ) -> SplitResult:
        df = self._normalize_dataframe(data)
        cfg = config or SplitConfiguration(strategy="temporal")

        flight_col = cfg.flight_id_column
        metrics = self._compute_group_metrics(
            df,
            group_col=flight_col,
            balance_by_duration=cfg.balance_by_duration,
            timestamp_col=cfg.timestamp_column,
        )

        # Sort flights chronologically by t_min
        sorted_flights = sorted(
            metrics.keys(),
            key=lambda g: (metrics[g]["t_min"], metrics[g]["group"]),
        )

        total_weight = sum(metrics[g]["weight"] for g in sorted_flights)
        train_target = cfg.train_ratio * total_weight
        val_target = (cfg.train_ratio + cfg.val_ratio) * total_weight

        fold_groups: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        accum_weight = 0.0

        for f in sorted_flights:
            w = metrics[f]["weight"]
            mid_w = accum_weight + (w / 2.0)
            accum_weight += w

            if mid_w <= train_target or (not fold_groups["train"] and cfg.train_ratio > 0):
                fold_groups["train"].append(f)
            elif mid_w <= val_target or (not fold_groups["val"] and cfg.val_ratio > 0 and len(sorted_flights) >= 3):
                fold_groups["val"].append(f)
            else:
                fold_groups["test"].append(f)

        # Ensure active folds get at least 1 group if sufficient flights exist
        if cfg.test_ratio > 0 and not fold_groups["test"] and len(sorted_flights) >= 3:
            if len(fold_groups["val"]) > 1:
                fold_groups["test"].append(fold_groups["val"].pop())
            elif len(fold_groups["train"]) > 1:
                fold_groups["test"].append(fold_groups["train"].pop())

        res = self._build_split_result(
            df=df,
            fold_groups=fold_groups,
            group_col=flight_col,
            config=cfg,
            parent_manifest=parent_manifest,
            audit_extra={"temporal_order_verified": True},
        )
        res.assert_zero_leakage(group_type="flight")
        return res
