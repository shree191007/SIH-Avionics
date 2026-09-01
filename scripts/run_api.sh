#!/bin/bash
# Starts the FastAPI backend (src/replan_to_learn/api/app.py) on :8000.
# Portable: uses the script's own location to find the repo root and
# whatever "python3" is first on PATH (activate your venv first).
#
# Plain HTTP by default (zero-friction for the console's LIVE button /
# a live demo). Set USE_TLS=1 to serve over HTTPS with a self-signed dev
# cert instead (auto-generated via gen_dev_cert.sh) -- if you do, the
# console needs VITE_API_BASE=https://localhost:8000 (see
# gcs-console/src/api/client.js) and a real browser will show a one-time
# cert-warning click-through on https://localhost:8000 first (no way to
# skip that from code -- see README's Telemetry security section).
# Set TELEMETRY_ENCRYPTION_KEY (see
# ingestion.crypto.TelemetryCipher.generate_key()) to additionally enable
# POST /telemetry/encrypted for payload-level encryption, independent of
# USE_TLS.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"

SSL_ARGS=()
if [ "${USE_TLS:-0}" = "1" ]; then
    "$ROOT/scripts/gen_dev_cert.sh"
    SSL_ARGS=(--ssl-keyfile "$ROOT/certs/dev_key.pem" --ssl-certfile "$ROOT/certs/dev_cert.pem")
fi

exec python3 -m uvicorn replan_to_learn.api.app:app --host 0.0.0.0 --port 8000 --reload \
    --app-dir "$ROOT/src" --reload-dir "$ROOT/src" "${SSL_ARGS[@]}"
