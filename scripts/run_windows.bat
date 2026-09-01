@echo off
REM Starts the FastAPI backend (:8000) and the gcs-console dev server
REM (:5173), each in its own console window. Close either window to stop
REM that server. Run scripts\setup_windows.bat once first.

setlocal
cd /d "%~dp0.."

echo == starting API on http://localhost:8000 (new window) ==
start "replan-to-learn API" cmd /k "scripts\run_api.bat"

echo == starting gcs-console on http://localhost:5173 (new window) ==
start "replan-to-learn console" cmd /k "npm --prefix gcs-console run dev -- --host"

echo.
echo Both servers starting in separate windows. Open http://localhost:5173
echo Close either window to stop that server.
