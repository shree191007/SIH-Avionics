"""
Stage 3 End-to-End Integration Tests.
Wires ML + RUL + Planner + GCS together and validates the full pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from replan_to_learn.contracts.residuals import ResidualFrame
from replan_to_learn.contracts.regimes import REGIME_GRID_VERSION
from replan_to_learn.gate.identifiability import IdentifiabilityGate
from replan_to_learn.gate.datatypes import Verdict
from replan_to_learn.stage3 import Stage3Orchestrator, Stage3Config


class MockPhysicsTwin:
    def __init__(self) -> None:
        self.health_params = np.zeros(15)
        self.health_params[0:9] = 1.0

    def jacobian(self, u, theta):
        # Must match PhysicsTwin.jacobian's real contract: a sequence of N
        # frames returns (N, 9, n_theta), not the same (9, n_theta) a
        # single frame returns -- IdentifiabilityGate._precompute_regime_
        # jacobian passes a multi-frame sequence and indexes per-step rows
        # (J[-1]/J[early_idx]) out of the result.
        n_frames = len(u) if hasattr(u, '__len__') else 1
        if n_frames == 1:
            return np.eye(9, len(theta))
        return np.tile(np.eye(9, len(theta)), (n_frames, 1, 1))

    def predict(self, frames, theta):
        n = len(frames) if hasattr(frames, '__len__') else 1
        return np.zeros((n, 9))

    def _compute_predictions(self, telemetry):
        return {"egt": np.zeros(4), "cht": 0.0, "p_oil": 0.0, "t_oil": 0.0, "mdot_f": 0.0, "p_brake": 0.0}


class TestStage3Orchestrator:
    def test_process_residual_frame_named(self) -> None:
        twin = MockPhysicsTwin()
        config = Stage3Config()
        orchestrator = Stage3Orchestrator(physics_twin=twin, config=config)

        rf = ResidualFrame(
            t=0.0,
            z=(0.1, 0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.2),
            regime=4,
            regime_stable_s=25.0,
            spatial=(0.1, 0.2, 0.05, 1),
            sigma=(1.0,) * 9,
            x_hat=(0.0,) * 9,
            status=0,
            model_version="1.0.0",
            regime_grid_version=REGIME_GRID_VERSION,
            flight_id="FLT001",
        )
        panel, metadata = orchestrator.process_residual_frame(rf)
        assert panel is not None
        assert "ENGINE HEALTH" in orchestrator.gcs_builder.render_text(panel)

    def test_process_residual_frame_invalid(self) -> None:
        twin = MockPhysicsTwin()
        config = Stage3Config()
        orchestrator = Stage3Orchestrator(physics_twin=twin, config=config)

        rf = ResidualFrame(
            t=0.0,
            z=(float('nan'),) * 9,
            regime=14,
            regime_stable_s=0.0,
            spatial=(0.0, 0.0, 0.0, 0),
            sigma=(1.0,) * 9,
            x_hat=(0.0,) * 9,
            status=3,
            model_version="1.0.0",
            regime_grid_version=REGIME_GRID_VERSION,
            flight_id="FLT001",
        )
        panel, metadata = orchestrator.process_residual_frame(rf)
        assert panel is not None

    def test_rul_filters_initialized(self) -> None:
        twin = MockPhysicsTwin()
        config = Stage3Config()
        orchestrator = Stage3Orchestrator(physics_twin=twin, config=config)
        assert len(orchestrator.rul_filters) > 0

    def test_gate_update_initializes(self) -> None:
        twin = MockPhysicsTwin()
        config = Stage3Config()
        orchestrator = Stage3Orchestrator(physics_twin=twin, config=config)
        rf = ResidualFrame(
            t=0.0,
            z=(0.1, 0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.2),
            regime=4,
            regime_stable_s=25.0,
            spatial=(0.1, 0.2, 0.05, 1),
            sigma=(1.0,) * 9,
            x_hat=(0.0,) * 9,
            status=0,
            model_version="1.0.0",
            regime_grid_version=REGIME_GRID_VERSION,
            flight_id="FLT001",
        )
        orchestrator.gate.update(rf)
        assert orchestrator.gate.theta_hat is not None

    def test_probe_selector_generated(self) -> None:
        twin = MockPhysicsTwin()
        config = Stage3Config()
        orchestrator = Stage3Orchestrator(physics_twin=twin, config=config)
        candidates = orchestrator.probe_selector.generate_candidates([], 50.0)
        assert len(candidates) > 0

    def test_mission_simulator_returns_result(self) -> None:
        twin = MockPhysicsTwin()
        config = Stage3Config()
        orchestrator = Stage3Orchestrator(physics_twin=twin, config=config)
        from replan_to_learn.planner.mission_simulator import MissionState
        profile = [MissionState(altitude_m=1500.0, ias_kt=120.0, rpm=2800.0, fuel_kg=50.0, elapsed_s=0.0, regime=4)]
        result = orchestrator.mission_simulator.simulate(np.ones(15), profile, 50.0)
        assert result.risk in ("LOW", "MEDIUM", "HIGH")
