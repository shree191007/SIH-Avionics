"""
SIH26054 Replan to Learn: Stage 5 API session management.

Holds one PhysicsTwin + Stage3Orchestrator per engine_id -- the API layer's
only state. All scientific logic (physics, estimation, gate, RUL,
mission, probe) lives in the stage modules and is called through
Stage3Orchestrator.process_residual_frame(), never reimplemented here
(05_gcs_integration_validation_deployment.md Sec 3: "The API contains
orchestration only.").
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional

from replan_to_learn.contracts.telemetry import TelemetryFrame
from replan_to_learn.fleet.fleet_node import FleetNode
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin
from replan_to_learn.stage3 import Stage3Config, Stage3Orchestrator

MODEL_DIR = Path(__file__).resolve().parents[3] / "model"


class EngineSession:
    def __init__(self, engine_id: str, fleet_node: Optional[FleetNode], model_dir: Path) -> None:
        self.engine_id = engine_id
        self.physics_twin = PhysicsTwin(model_dir=model_dir, dt=0.1)
        self.orchestrator = Stage3Orchestrator(
            physics_twin=self.physics_twin,
            config=Stage3Config(),
            fleet_node=fleet_node,
        )
        self.frames_processed = 0
        self.last_t: Optional[float] = None
        self.last_panel = None
        self.last_metadata = None
        self._reset_pending = True

    def process(self, frame: TelemetryFrame):
        if self._reset_pending:
            self.physics_twin.reset(frame)
            self._reset_pending = False
        rf = self.physics_twin.step(frame)
        panel, metadata = self.orchestrator.process_residual_frame(rf)
        self.frames_processed += 1
        self.last_t = frame.t
        self.last_panel = panel
        self.last_metadata = metadata
        return panel, metadata

    def reset(self) -> None:
        self._reset_pending = True
        self.frames_processed = 0
        self.last_t = None
        self.last_panel = None
        self.last_metadata = None


class SessionManager:
    """
    Thread-safe registry of EngineSession, one per engine_id. A single
    FleetNode is shared across all engines on this GCS instance (fleet
    federation is inherently cross-engine, per 03_state_identifiability_
    fleet.md); each engine gets its own isolated PhysicsTwin + gate state.
    """

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.fleet_node = FleetNode()
        self._sessions: Dict[str, EngineSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, engine_id: str) -> EngineSession:
        with self._lock:
            if engine_id not in self._sessions:
                self._sessions[engine_id] = EngineSession(engine_id, self.fleet_node, self.model_dir)
            return self._sessions[engine_id]

    def get(self, engine_id: str) -> Optional[EngineSession]:
        return self._sessions.get(engine_id)

    def all_engine_ids(self):
        return list(self._sessions.keys())
