#!/bin/bash
set -e
ROOT="/Users/vatxn1907__/Desktop/V1"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"
exec /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m uvicorn replan_to_learn.api.app:app --host 0.0.0.0 --port 8000 --reload --app-dir "$ROOT/src" --reload-dir "$ROOT/src"
