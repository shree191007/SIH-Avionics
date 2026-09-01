@echo off
REM Starts the FastAPI backend (src\replan_to_learn\api\app.py) on :8000.
REM Plain HTTP by default (zero-friction for the console's LIVE button).
REM Set USE_TLS=1 to serve over HTTPS with a self-signed dev cert instead
REM (auto-generated via gen_dev_cert.bat) -- if you do, the console needs
REM VITE_API_BASE=https://localhost:8000 and a real browser will show a
REM one-time cert-warning click-through first (see README's Telemetry
REM security section). Set TELEMETRY_ENCRYPTION_KEY to additionally
REM enable POST /telemetry/encrypted for payload-level encryption.
setlocal
cd /d "%~dp0.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=%cd%\src;%PYTHONPATH%

set SSL_ARGS=
if "%USE_TLS%"=="1" (
    call scripts\gen_dev_cert.bat
    set SSL_ARGS=--ssl-keyfile "%cd%\certs\dev_key.pem" --ssl-certfile "%cd%\certs\dev_cert.pem"
)

python -m uvicorn replan_to_learn.api.app:app --host 0.0.0.0 --port 8000 --reload ^
    --app-dir "%cd%\src" --reload-dir "%cd%\src" %SSL_ARGS%
