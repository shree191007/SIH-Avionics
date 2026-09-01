"""
SIH26054 Replan to Learn: Stage 5 API.
Implements 05_gcs_integration_validation_deployment.md Sec 3's endpoint
contract. Orchestration only -- every response is built from a real
Stage3Orchestrator (physics twin + L4 gate + L6 RUL + L7 planner + GCS
panel builder) call, no logic duplicated here.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from replan_to_learn.api.schemas import (
    DiagnosisOut,
    EngineHealthOut,
    EngineStateOut,
    EvidenceOut,
    FleetFingerprintOut,
    MissionSimulateIn,
    MissionSimulateOut,
    ProbeSelectIn,
    ProbeSelectOut,
    ProcessResultOut,
    ReplayIn,
    RulOut,
    TelemetryIn,
)
from replan_to_learn.api.session import SessionManager
from replan_to_learn.contracts.telemetry import TelemetryFrame
from replan_to_learn.gate.datatypes import Verdict
from replan_to_learn.ingestion import TelemetryCipher, TelemetryIngestionError, TelemetryIngestor
from replan_to_learn.planner.mission_simulator import MissionState

app = FastAPI(title="Replan to Learn API", version="1.0.0")
# Local-dev only: the gcs-console Vite frontend runs on a different port
# (5173) than this API (8000), so the browser's fetch() calls need CORS
# allowed. Not a deployment/production CORS policy -- narrow this before
# any real network-exposed deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# TELEMETRY_ENCRYPTION_KEY: optional Fernet key (see
# ingestion.crypto.TelemetryCipher.generate_key()) enabling POST
# /telemetry/encrypted for payload-level confidentiality independent of
# TLS. Unset by default -- most deployments only need TLS (below); this
# is for defense-in-depth or an untrusted relay between source and API.
_cipher_key = os.environ.get("TELEMETRY_ENCRYPTION_KEY")
_cipher = TelemetryCipher(_cipher_key) if _cipher_key else None

_sessions = SessionManager()
_ingestor = TelemetryIngestor(_sessions, cipher=_cipher)


def _finite_or_none(x: float) -> Optional[float]:
    """inf/-inf/nan are not valid JSON tokens under a strict encoder -- see RulOut's docstring for why RULReport.p05/p95 legitimately produce them."""
    if x is None:
        return None
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _telemetry_frame(t_in: TelemetryIn) -> TelemetryFrame:
    return TelemetryFrame(
        t=t_in.t, egt=t_in.egt, cht=t_in.cht, p_oil=t_in.p_oil, t_oil=t_in.t_oil,
        n_rpm=t_in.n_rpm, mdot_f=t_in.mdot_f, map_pa=t_in.map_pa, tps=t_in.tps,
        t_im=t_in.t_im, p_amb=t_in.p_amb, t_amb=t_in.t_amb, v_tas=t_in.v_tas, h_p=t_in.h_p,
        valid_mask=t_in.valid_mask, flight_id=t_in.flight_id,
        aircraft_id=t_in.aircraft_id, engine_id=t_in.engine_id,
    )


def _require_session(engine_id: str):
    session = _sessions.get(engine_id)
    if session is None or session.last_panel is None:
        raise HTTPException(status_code=404, detail=f"No processed state for engine_id={engine_id!r}. POST /telemetry first.")
    return session


def _build_process_result(session) -> ProcessResultOut:
    panel = session.last_panel
    metadata = session.last_metadata
    ga = metadata["gate_attribution"]
    theta_names = session.orchestrator.gcs_builder.theta_names
    named = ga["named"]
    ambiguous_set = tuple(ga["ambiguous_set"])

    diagnosis = DiagnosisOut(
        verdict=panel.diagnosis.verdict,
        named=named,
        named_name=theta_names[named] if named is not None and named < len(theta_names) else None,
        ambiguous_set=ambiguous_set,
        ambiguous_names=[theta_names[i] for i in ambiguous_set if i < len(theta_names)],
        confidence=panel.diagnosis.confidence,
        attribution_basis=panel.diagnosis.attribution_basis,
    )
    evidence = EvidenceOut(
        cos_matrix=ga.get("cos_matrix"),
        crlb=list(ga.get("crlb", [])),
        ambiguous_set=ambiguous_set,
        tree_shap_top5=list(panel.diagnosis.tree_shap_top5),
    )
    rul_reports = metadata.get("rul_reports", {})
    rul_out = [
        RulOut(
            theta_index=int(idx), theta_name=theta_names[int(idx)] if int(idx) < len(theta_names) else str(idx),
            p05=_finite_or_none(r["p05"]), p50=_finite_or_none(r["p50"]), p95=_finite_or_none(r["p95"]),
            p_fail_before_mission_end=r["p_fail_before_mission_end"],
            mission_end_time_s=r["mission_end_time_s"],
        )
        for idx, r in rul_reports.items()
    ]
    health = EngineHealthOut(
        health_percent=panel.engine_health.health_percent, status=panel.engine_health.status,
        primary_issue=panel.engine_health.primary_issue, egt=panel.engine_health.egt,
        cht=panel.engine_health.cht, p_oil=panel.engine_health.p_oil, t_oil=panel.engine_health.t_oil,
        n_rpm=panel.engine_health.n_rpm, mdot_f=panel.engine_health.mdot_f,
    )
    mission = MissionSimulateOut(
        risk=panel.mission.risk, risk_probability=panel.mission.risk_probability,
        recommended_action=panel.mission.recommendation, alternatives=panel.mission.alternatives,
        margin_warnings=tuple(),
    )
    fleet = FleetFingerprintOut(
        theta_index=named if named is not None else -1,
        fleet_epoch=panel.fleet.fleet_epoch, contributor_count=panel.fleet.contributor_count,
        r2=panel.fleet.r2, transfer_status=panel.fleet.transfer_status,
        borrowed_regimes=panel.fleet.borrowed_regimes,
    )
    return ProcessResultOut(engine_health=health, diagnosis=diagnosis, evidence=evidence, rul=rul_out, mission=mission, fleet=fleet)


@app.post("/telemetry", response_model=ProcessResultOut)
def post_telemetry(t_in: TelemetryIn) -> ProcessResultOut:
    try:
        session = _ingestor.ingest(t_in.model_dump())
    except TelemetryIngestionError as e:
        status = 400 if "engine_id is required" in str(e) else 422
        raise HTTPException(status_code=status, detail=str(e))
    return _build_process_result(session)


@app.post("/telemetry/encrypted", response_model=ProcessResultOut)
async def post_telemetry_encrypted(request: Request) -> ProcessResultOut:
    """Body is a raw Fernet token (see TelemetryCipher), not JSON -- for a
    telemetry source that encrypts the payload itself, independent of TLS.
    404s if TELEMETRY_ENCRYPTION_KEY isn't configured on this server."""
    if _cipher is None:
        raise HTTPException(status_code=404, detail="Payload encryption is not configured on this server")
    token = await request.body()
    try:
        session = _ingestor.ingest_encrypted(token)
    except TelemetryIngestionError as e:
        status = 400 if "engine_id is required" in str(e) else 422
        raise HTTPException(status_code=status, detail=str(e))
    return _build_process_result(session)


@app.post("/replay", response_model=ProcessResultOut)
def post_replay(body: ReplayIn) -> ProcessResultOut:
    """Replays a sequence of frames for one engine through the full pipeline; returns the FINAL processed state."""
    if not body.frames:
        raise HTTPException(status_code=400, detail="frames must be non-empty")
    session = _sessions.get_or_create(body.engine_id)
    if body.reset_state:
        session.reset()
    for t_in in body.frames:
        frame = _telemetry_frame(TelemetryIn(**{**t_in.model_dump(), "engine_id": body.engine_id}))
        try:
            session.process(frame)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Replay failed at t={t_in.t}: {e}")
    return _build_process_result(session)


@app.get("/engine/{engine_id}/state", response_model=EngineStateOut)
def get_state(engine_id: str) -> EngineStateOut:
    session = _require_session(engine_id)
    ga = session.last_metadata["gate_attribution"]
    theta_hat = session.orchestrator.gate.theta_hat
    return EngineStateOut(
        engine_id=engine_id, frames_processed=session.frames_processed, last_t=session.last_t,
        theta_hat=theta_hat.tolist() if theta_hat is not None else None,
        last_verdict=ga["verdict"],
    )


@app.get("/engine/{engine_id}/health", response_model=EngineHealthOut)
def get_health(engine_id: str) -> EngineHealthOut:
    session = _require_session(engine_id)
    return _build_process_result(session).engine_health


@app.get("/engine/{engine_id}/evidence", response_model=EvidenceOut)
def get_evidence(engine_id: str) -> EvidenceOut:
    session = _require_session(engine_id)
    return _build_process_result(session).evidence


@app.get("/engine/{engine_id}/diagnosis", response_model=DiagnosisOut)
def get_diagnosis(engine_id: str) -> DiagnosisOut:
    session = _require_session(engine_id)
    return _build_process_result(session).diagnosis


@app.get("/engine/{engine_id}/rul", response_model=List[RulOut])
def get_rul(engine_id: str) -> List[RulOut]:
    session = _require_session(engine_id)
    return _build_process_result(session).rul


@app.post("/mission/simulate", response_model=MissionSimulateOut)
def post_mission_simulate(body: MissionSimulateIn) -> MissionSimulateOut:
    session = _require_session(body.engine_id)
    theta_hat = session.orchestrator.gate.theta_hat
    theta = theta_hat if theta_hat is not None else np.ones(15)
    profile = (
        [MissionState(**s.model_dump()) for s in body.mission_profile]
        if body.mission_profile
        else session.orchestrator._base_mission_profile
    )
    fuel = body.current_fuel_kg if body.current_fuel_kg is not None else session.orchestrator.config.current_fuel_kg
    result = session.orchestrator.mission_simulator.simulate(theta=theta, mission_profile=profile, current_fuel_kg=fuel)
    return MissionSimulateOut(
        risk=result.risk, risk_probability=result.risk_probability,
        recommended_action=result.recommended_action, alternatives=result.alternatives,
        margin_warnings=result.margin_warnings,
    )


@app.post("/probe/select", response_model=ProbeSelectOut)
def post_probe_select(body: ProbeSelectIn) -> ProbeSelectOut:
    session = _require_session(body.engine_id)
    ga = session.last_metadata["gate_attribution"]
    if ga["verdict"] != str(Verdict.AMBIGUOUS):
        return ProbeSelectOut(selected=False, manoeuvre_type=None, parameters=None, separability_gain=None, cost=None, reason="Gate verdict is not AMBIGUOUS -- no probe needed.")

    theta_hat = session.orchestrator.gate.theta_hat
    theta = theta_hat if theta_hat is not None else np.ones(15)
    crlb = np.array(ga.get("crlb", np.ones(15)))
    fim_det = float(np.prod(np.maximum(1e-6, 1.0 / np.maximum(crlb, 1e-6) ** 2)))
    result = session.orchestrator.probe_selector.select_probe(
        ambiguous_set=tuple(ga["ambiguous_set"]), theta=theta,
        current_fim_det=fim_det * 0.8, post_probe_fim_det=fim_det,
        current_fuel_kg=session.orchestrator.config.current_fuel_kg,
        current_altitude_m=session.orchestrator.config.current_altitude_m,
        return_margin_s=session.orchestrator.config.return_margin_s,
        p_misattribution=body.p_misattribution, cost_wrong_action=body.cost_wrong_action,
    )
    if result.selected_probe is None:
        return ProbeSelectOut(selected=False, manoeuvre_type=None, parameters=None, separability_gain=None, cost=None, reason=result.reason)
    sp = result.selected_probe
    return ProbeSelectOut(
        selected=True, manoeuvre_type=sp.manoeuvre.manoeuvre_type.value, parameters=dict(sp.manoeuvre.parameters),
        separability_gain=sp.separability_gain, cost=sp.cost, reason=result.reason,
    )


@app.get("/fleet/{engine_id}/fingerprint", response_model=FleetFingerprintOut)
def get_fleet_fingerprint(engine_id: str) -> FleetFingerprintOut:
    session = _require_session(engine_id)
    return _build_process_result(session).fleet


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "engines": ",".join(_sessions.all_engine_ids())}
