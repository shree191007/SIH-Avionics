#!/bin/bash
# Starts the FastAPI backend (src/replan_to_learn/api/app.py) on :8000.
# Portable: uses the script's own location to find the repo root and
# whatever "python3" is first on PATH (activate your venv first).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"
exec python3 -m uvicorn replan_to_learn.api.app:app --host 0.0.0.0 --port 8000 --reload --app-dir "$ROOT/src" --reload-dir "$ROOT/src"
