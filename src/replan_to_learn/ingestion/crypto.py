"""
Payload-level encryption for telemetry frames, independent of transport.

TLS (see scripts/run_api.sh's --ssl-keyfile/--ssl-certfile) protects the
wire between a telemetry source and this API, but only for that one hop.
TelemetryCipher protects the payload itself -- relevant once a transport
sits between the source and this backend that isn't fully trusted end-to-
end (e.g. a relayed MQTT broker), or for defense-in-depth if TLS is ever
terminated somewhere upstream of the API process.

Uses Fernet (AES-128-CBC + HMAC-SHA256, authenticated symmetric
encryption) from the `cryptography` package -- deliberately not a
hand-rolled scheme. A tampered or wrong-key payload raises
TelemetryDecryptionError rather than silently decrypting to garbage.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from cryptography.fernet import Fernet, InvalidToken


class TelemetryDecryptionError(Exception):
    """Raised when a telemetry payload fails to decrypt or authenticate."""


class TelemetryCipher:
    """Symmetric encrypt/decrypt for telemetry JSON payloads.

    The key is a Fernet key (32 url-safe base64-encoded bytes) -- generate
    one with TelemetryCipher.generate_key(). Treat it as a secret: store it
    in an env var or secrets manager, never commit it.
    """

    def __init__(self, key: bytes | str) -> None:
        if isinstance(key, str):
            key = key.encode("ascii")
        self._fernet = Fernet(key)

    @staticmethod
    def generate_key() -> str:
        """Returns a new Fernet key as a str, suitable for an env var."""
        return Fernet.generate_key().decode("ascii")

    def encrypt(self, payload: Dict[str, Any]) -> bytes:
        raw = json.dumps(payload).encode("utf-8")
        return self._fernet.encrypt(raw)

    def decrypt(self, token: bytes | str) -> Dict[str, Any]:
        if isinstance(token, str):
            token = token.encode("ascii")
        try:
            raw = self._fernet.decrypt(token)
        except InvalidToken as e:
            raise TelemetryDecryptionError(
                "Telemetry payload failed to decrypt/authenticate -- wrong key or tampered data."
            ) from e
        return json.loads(raw.decode("utf-8"))
