#!/bin/bash
# One-time setup for macOS/Linux: creates a Python venv, installs the
# backend package + dependencies, and installs the frontend's npm
# dependencies. Run this once before scripts/run_mac.sh.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== checking prerequisites =="
command -v python3 >/dev/null || { echo "python3 not found -- install Python 3.10+ first."; exit 1; }
command -v npm >/dev/null || { echo "npm not found -- install Node.js 18+ first."; exit 1; }

echo "== creating virtual environment (.venv) =="
python3 -m venv .venv
source .venv/bin/activate

echo "== installing Python dependencies =="
pip install --upgrade pip
pip install -e ".[dev]"

echo "== installing frontend dependencies =="
npm --prefix gcs-console install

echo
echo "Setup complete. Next: ./scripts/run_mac.sh"
echo "(model/ ships pre-calibrated; data/ -- real flight data -- is not"
echo " included, see README.md's Data section if you need it.)"
