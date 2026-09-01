"""
SIH26054 Replan to Learn: Stage 5 API request/response schemas.
Pydantic models mirroring the dataclass contracts in contracts/, gcs/,
rul/, and planner/ -- the API layer is orchestration only (05_gcs_
integration_validation_deployment.md Sec 3: "The API contains
orchestration only. Scientific logic remains in the stage modules."),
so these schemas are thin serialization wrappers, not a second copy of
any physics or estimation logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class TelemetryIn(BaseModel):
    """Mirrors contracts/telemetry.py:TelemetryFrame exactly (SI units, immutable-per-frame contract)."""
    t: float
    egt: Tuple[float, float, float, float]
    cht: float
    p_oil: float
    t_oil: float
    n_rpm: float
    mdot_f: float
    map_pa: float
    tps: float
    t_im: float
    p_amb: float
    t_amb: float
    v_tas: float
    h_p: float
    valid_mask: int = 0xFFFFFFFF
    flight_id: str = ""
    aircraft_id: str = ""
    engine_id: str = ""


class ReplayIn(BaseModel):
    """POST /replay: replay a sequence of telemetry frames for one engine through the full pipeline."""
    engine_id: str
    frames: List[TelemetryIn]
    reset_state: bool = True


class MissionStateIn(BaseModel):
    altitude_m: float
    ias_kt: float
    rpm: float
    fuel_kg: float
    elapsed_s: float
    regime: int = 4


class MissionSimulateIn(BaseModel):
    engine_id: str
    mission_profile: Optional[List[MissionStateIn]] = None
    current_fuel_kg: Optional[float] = None


class ProbeSelectIn(BaseModel):
    engine_id: str
    p_misattribution: float = 0.3
    cost_wrong_action: float = 1.0


class EngineStateOut(BaseModel):
    engine_id: str
    frames_processed: int
    last_t: Optional[float]
    theta_hat: Optional[List[float]]
    last_verdict: Optional[str]


class EngineHealthOut(BaseModel):
    health_percent: int
    status: str
    primary_issue: Optional[str]
    egt: Tuple[float, float, float, float]
    cht: float
    p_oil: float
    t_oil: float
    n_rpm: float
    mdot_f: float


class EvidenceOut(BaseModel):
    """GET /engine/{id}/evidence -- the explainability chain: cosine matrix, CRLB, TreeSHAP top-5."""
    cos_matrix: Optional[List[List[float]]]
    crlb: List[float]
    ambiguous_set: Tuple[int, ...]
    tree_shap_top5: List[Tuple[str, float]]


class DiagnosisOut(BaseModel):
    verdict: str
    named: Optional[int]
    named_name: Optional[str]
    ambiguous_set: Tuple[int, ...]
    ambiguous_names: List[str]
    confidence: float
    attribution_basis: str


class RulOut(BaseModel):
    """
    p05/p50/p95 are Optional: the particle filter's first-passage-time
    search returns math.inf for "no particle crossed the failure
    threshold within the search horizon" (e.g. a healthy/near-nominal
    parameter with no real degradation trend) -- a real, meaningful
    result ("this parameter isn't going to fail any time we can see"),
    not an error. `inf`/`nan` are not valid JSON tokens under a strict
    encoder, so these are normalized to null before serialization (see
    app.py's _sanitize_floats) and must be treated as "not applicable /
    no failure predicted" by any client, never coerced to 0 or a huge
    number.
    """
    theta_index: int
    theta_name: str
    p05: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    p_fail_before_mission_end: float
    mission_end_time_s: float


class MissionSimulateOut(BaseModel):
    risk: str
    risk_probability: float
    recommended_action: str
    alternatives: List[Dict[str, Any]]
    margin_warnings: Tuple[str, ...]


class ProbeSelectOut(BaseModel):
    selected: bool
    manoeuvre_type: Optional[str]
    parameters: Optional[Dict[str, float]]
    separability_gain: Optional[float]
    cost: Optional[float]
    reason: str


class FleetFingerprintOut(BaseModel):
    theta_index: int
    fleet_epoch: Optional[int]
    contributor_count: int
    r2: Optional[float]
    transfer_status: str
    borrowed_regimes: Tuple[int, ...]


class ProcessResultOut(BaseModel):
    """Full processed frame: everything GCS needs in one response, mirroring gcs/datatypes.py:GCSPanel."""
    engine_health: EngineHealthOut
    diagnosis: DiagnosisOut
    evidence: EvidenceOut
    rul: List[RulOut]
    mission: MissionSimulateOut
    fleet: FleetFingerprintOut
