#!/bin/bash
# Starts the FastAPI backend (:8000) and the gcs-console dev server
# (:5173) together, and stops both cleanly on Ctrl-C.
# Run scripts/setup_mac.sh once first.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

cleanup() {
  echo
  echo "stopping..."
  kill "$API_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "== starting API on http://localhost:8000 =="
bash scripts/run_api.sh &
API_PID=$!

echo "== starting gcs-console on http://localhost:5173 =="
npm --prefix gcs-console run dev -- --host &
FRONTEND_PID=$!

echo
echo "Both servers running. Open http://localhost:5173 -- press Ctrl-C to stop both."
wait
