"""
Transport-agnostic telemetry ingestion layer.

Separates "how a telemetry frame arrives" (an HTTP POST today, an MQTT
subscriber if/when that's built) from "what happens once it's a
TelemetryFrame" (SessionManager.process(), unchanged). A new transport
constructs one TelemetryIngestor around the shared SessionManager and
calls ingest() with a raw dict payload -- no change needed to
Stage3Orchestrator or any processing logic to add a transport.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from replan_to_learn.api.schemas import TelemetryIn
from replan_to_learn.api.session import EngineSession, SessionManager
from replan_to_learn.contracts.telemetry import TelemetryFrame
from replan_to_learn.ingestion.crypto import TelemetryCipher, TelemetryDecryptionError


class TelemetryIngestionError(Exception):
    """Raised when a raw telemetry payload fails validation or pipeline processing."""


class TelemetryIngestor:
    """Transport-agnostic entry point for telemetry frames.

    Construct one instance around a SessionManager and call ingest() from
    any transport handler (HTTP route, MQTT on_message callback, ...).

    If `cipher` is provided, ingest_encrypted() decrypts a Fernet token
    (see crypto.TelemetryCipher) into a raw dict before ingesting -- for a
    transport where the payload itself, not just the wire, needs to stay
    confidential (e.g. relayed through a broker that isn't fully trusted).
    """

    def __init__(self, sessions: SessionManager, cipher: Optional[TelemetryCipher] = None) -> None:
        self._sessions = sessions
        self._cipher = cipher

    def ingest_encrypted(self, token: bytes | str) -> EngineSession:
        if self._cipher is None:
            raise TelemetryIngestionError("No TelemetryCipher configured for this ingestor")
        try:
            raw = self._cipher.decrypt(token)
        except TelemetryDecryptionError as e:
            raise TelemetryIngestionError(str(e)) from e
        return self.ingest(raw)

    def ingest(self, raw: Dict[str, Any]) -> EngineSession:
        t_in = TelemetryIn(**raw)
        if not t_in.engine_id:
            raise TelemetryIngestionError("engine_id is required")
        session = self._sessions.get_or_create(t_in.engine_id)
        frame = self._frame_from(t_in)
        try:
            session.process(frame)
        except Exception as e:
            raise TelemetryIngestionError(f"Failed to process telemetry: {e}") from e
        return session

    @staticmethod
    def _frame_from(t_in: TelemetryIn) -> TelemetryFrame:
        return TelemetryFrame(
            t=t_in.t, egt=t_in.egt, cht=t_in.cht, p_oil=t_in.p_oil, t_oil=t_in.t_oil,
            n_rpm=t_in.n_rpm, mdot_f=t_in.mdot_f, map_pa=t_in.map_pa, tps=t_in.tps,
            t_im=t_in.t_im, p_amb=t_in.p_amb, t_amb=t_in.t_amb, v_tas=t_in.v_tas, h_p=t_in.h_p,
            valid_mask=t_in.valid_mask, flight_id=t_in.flight_id,
            aircraft_id=t_in.aircraft_id, engine_id=t_in.engine_id,
        )
