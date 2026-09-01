"""
SIH26054 Replan to Learn: Stage 5 deterministic end-to-end replay.
Implements 05_gcs_integration_validation_deployment.md Sec 4:

    dataset -> Stage 2 -> Stage 3 -> Stage 4 -> GCS -> evaluation report

One command replays a real flight through the full pipeline
(PhysicsTwin -> IdentifiabilityGate -> RUL/mission/probe -> GCSPanel) and
records the provenance the spec requires: dataset hash, model versions,
configuration, random seeds, software version, and result artifacts.
Determinism is verified, not just claimed -- the CLI can run the same
replay twice and diff the two result artifacts bit-for-bit.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "real_data"))

from real_data_loader import RealFlightLoader  # noqa: E402
from replan_to_learn.contracts.telemetry import TelemetryFrame  # noqa: E402
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402
from replan_to_learn.stage3 import Stage3Config, Stage3Orchestrator  # noqa: E402

MODEL_DIR = ROOT / "model"
DATA_ROOT = ROOT / "data"
REPLAY_OUT_DIR = ROOT / "replay_results"
SOFTWARE_VERSION = "1.0.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_versions(model_dir: Path) -> Dict[str, str]:
    """Hash of every model/*.json artifact -- the real calibration state this replay ran against, not just a version string."""
    versions = {}
    for p in sorted(model_dir.glob("*.json")):
        versions[p.name] = _sha256_file(p)
    return versions


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    if isinstance(obj, np.integer):
        return obj.item()
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


@dataclass
class ReplayFrameResult:
    t: float
    regime: int
    verdict: str
    named: Optional[int]
    health_percent: int
    status: str
    risk: str


@dataclass
class ReplayReport:
    # Provenance (spec Sec 4's required record)
    dataset_flight_id: str
    dataset_hash: str
    model_versions: Dict[str, str]
    configuration: Dict[str, Any]
    random_seed: int
    software_version: str
    generated_at_utc: str
    python_version: str
    platform: str
    # Result
    n_frames_replayed: int
    frame_results: List[Dict[str, Any]]
    final_panel: Dict[str, Any]
    final_metadata: Dict[str, Any]
    report_hash: Optional[str] = None


class ReplayEngine:
    """Deterministic end-to-end replay: dataset -> Stage2 -> Stage3 -> Stage4 -> GCS -> report."""

    def __init__(self, model_dir: Path = MODEL_DIR, data_root: Path = DATA_ROOT, seed: int = 42) -> None:
        self.model_dir = model_dir
        self.data_root = data_root
        self.seed = seed

    def replay_flight(self, flight_id: str, max_frames: Optional[int] = None) -> ReplayReport:
        np.random.seed(self.seed)

        loader = RealFlightLoader(self.data_root)
        df = loader.load_flight(flight_id)
        if df is None:
            raise FileNotFoundError(f"No processed flight parquet for flight_id={flight_id!r} under {self.data_root}")
        frames = loader.to_telemetry_frames(df, flight_id)
        if max_frames is not None:
            frames = frames[:max_frames]
        if not frames:
            raise ValueError(f"flight_id={flight_id!r} produced zero telemetry frames")

        source_path = self.data_root / "processed" / f"flight_{flight_id}.parquet"
        dataset_hash = _sha256_file(source_path)

        twin = PhysicsTwin(model_dir=self.model_dir, dt=0.1)
        config = Stage3Config()
        orchestrator = Stage3Orchestrator(physics_twin=twin, config=config)

        twin.reset(frames[0])
        frame_results: List[ReplayFrameResult] = []
        last_panel = None
        last_metadata = None
        for frame in frames:
            try:
                rf = twin.step(frame)
            except Exception:
                continue
            if rf.status != 0:
                continue
            panel, metadata = orchestrator.process_residual_frame(rf)
            last_panel, last_metadata = panel, metadata
            frame_results.append(ReplayFrameResult(
                t=frame.t, regime=int(rf.regime), verdict=metadata["gate_attribution"]["verdict"],
                named=metadata["gate_attribution"]["named"], health_percent=panel.engine_health.health_percent,
                status=panel.engine_health.status, risk=panel.mission.risk,
            ))

        if last_panel is None:
            raise RuntimeError(f"flight_id={flight_id!r}: no frame produced a valid ResidualFrame (all status!=0)")

        report = ReplayReport(
            dataset_flight_id=flight_id,
            dataset_hash=dataset_hash,
            model_versions=_model_versions(self.model_dir),
            configuration=_to_jsonable(config),
            random_seed=self.seed,
            software_version=SOFTWARE_VERSION,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version,
            platform=platform.platform(),
            n_frames_replayed=len(frame_results),
            frame_results=[_to_jsonable(fr) for fr in frame_results],
            final_panel=_to_jsonable(last_panel),
            final_metadata=_to_jsonable(last_metadata),
        )
        # report_hash covers everything EXCEPT itself and the wall-clock
        # generated_at_utc timestamp -- two replays of the same inputs
        # must hash identically to prove determinism, even though they
        # run at different real times.
        hashable = _to_jsonable(report)
        hashable.pop("report_hash", None)
        hashable.pop("generated_at_utc", None)
        report.report_hash = hashlib.sha256(json.dumps(hashable, sort_keys=True).encode()).hexdigest()
        return report

    def save_report(self, report: ReplayReport, out_dir: Path = REPLAY_OUT_DIR) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"replay_{report.dataset_flight_id}_{report.report_hash[:12]}.json"
        out_path.write_text(json.dumps(_to_jsonable(report), indent=2, sort_keys=True))
        return out_path
