"""
L6 physics-derived RUL failure threshold solver (04_ml_rul_mission_probe.md
Sec 2.4). Locks in real values against the calibrated model and checks the
solver's physical sanity properties -- not just that it runs.

Before this solver existed, stage3.py hardcoded round-number thresholds
(0.80, 0.70, 0.75, 1.30) with no derivation, directly contradicting the
spec's explicit requirement that theta_j^fail be computed from the twin.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from replan_to_learn.physics_twin.physics_twin import PhysicsTwin
from replan_to_learn.rul.failure_threshold_solver import (
    load_or_solve_failure_threshold,
    solve_failure_threshold,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
CACHE_PATH = MODEL_DIR / "rul_failure_thresholds.json"

THETA_NAMES = [
    "theta_vol", "theta_comb", "theta_cool",
    "theta_inj1", "theta_inj2", "theta_inj3", "theta_inj4",
    "theta_oilp", "theta_fric",
]


@pytest.fixture(scope="module")
def twin():
    return PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)


class TestCachedThresholdsExistAndAreSane:
    """The precomputed cache (scripts/solve_rul_failure_thresholds.py) is checked in and physically sane."""

    def test_cache_file_exists(self):
        assert CACHE_PATH.exists(), (
            "model/rul_failure_thresholds.json is missing -- run "
            "scripts/solve_rul_failure_thresholds.py to regenerate it"
        )

    def test_all_nine_parameters_present(self):
        cached = json.loads(CACHE_PATH.read_text())
        for i in range(9):
            assert str(i) in cached["thresholds"], f"missing cached threshold for theta_index={i}"

    @pytest.mark.parametrize("idx,name,lo,hi", [
        (0, "theta_vol", 0.5, 0.95),    # degraded but not implausibly so before losing required climb power
        (1, "theta_comb", 0.5, 0.95),   # degraded combustion efficiency before EGT limit
        (2, "theta_cool", 0.1, 0.9),    # real cooling-loss margin before CHT limit, not near-total loss
        (3, "theta_inj1", 0.5, 0.95),
        (4, "theta_inj2", 0.5, 0.95),
        (5, "theta_inj3", 0.5, 0.95),
        (6, "theta_inj4", 0.5, 0.95),
        (7, "theta_oilp", 0.1, 0.9),
        (8, "theta_fric", 1.05, 2.5),   # friction fault direction is ABOVE nominal
    ])
    def test_threshold_in_plausible_range(self, idx, name, lo, hi):
        cached = json.loads(CACHE_PATH.read_text())
        val = cached["thresholds"][str(idx)]["threshold_value"]
        assert lo <= val <= hi, f"{name}: threshold {val} outside plausible range [{lo}, {hi}]"

    def test_injector_thresholds_are_identical_by_symmetry(self):
        """All 4 injectors share the same EGT-limit stress point and physics, so their solved thresholds should match exactly."""
        cached = json.loads(CACHE_PATH.read_text())
        vals = [cached["thresholds"][str(i)]["threshold_value"] for i in [3, 4, 5, 6]]
        assert max(vals) - min(vals) < 1e-6, f"injector thresholds should be identical by symmetry, got {vals}"

    def test_fric_threshold_is_above_nominal_others_below(self):
        """theta_fric is the only parameter whose fault direction is ABOVE 1.0 (more friction = worse); all others are efficiency losses (below 1.0)."""
        cached = json.loads(CACHE_PATH.read_text())
        assert cached["thresholds"]["8"]["threshold_value"] > 1.0, "theta_fric threshold should be > 1.0 (excess friction)"
        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            assert cached["thresholds"][str(i)]["threshold_value"] < 1.0, (
                f"theta_index={i} threshold should be < 1.0 (efficiency loss)"
            )


class TestSolverLiveConsistencyWithCache:
    """A fresh live solve should reproduce the cached values (both use the same calibrated model)."""

    @pytest.mark.parametrize("idx,name", [(2, "theta_cool"), (7, "theta_oilp")])
    def test_live_solve_matches_cache(self, twin, idx, name):
        cached = json.loads(CACHE_PATH.read_text())
        cached_val = cached["thresholds"][str(idx)]["threshold_value"]
        live = solve_failure_threshold(twin, idx, name)
        assert abs(live.threshold_value - cached_val) < 0.01, (
            f"{name}: live solve {live.threshold_value} disagrees with cached {cached_val} "
            "-- cache may be stale, rerun scripts/solve_rul_failure_thresholds.py"
        )


class TestLoadOrSolveFallback:
    def test_loads_from_cache_when_present(self, twin):
        ft = load_or_solve_failure_threshold(twin, 2, "theta_cool", cache_path=CACHE_PATH)
        assert ft.theta_index == 2
        assert ft.threshold_value > 0.0

    def test_falls_back_to_live_solve_when_cache_missing(self, twin, tmp_path):
        missing_cache = tmp_path / "nonexistent.json"
        ft = load_or_solve_failure_threshold(twin, 7, "theta_oilp", cache_path=missing_cache)
        assert ft.theta_index == 7
        assert 0.0 < ft.threshold_value < 1.0
