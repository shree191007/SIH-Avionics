@echo off
REM One-time setup for Windows: creates a Python venv, installs the
REM backend package + dependencies, and installs the frontend's npm
REM dependencies. Run this once before scripts\run_windows.bat.

setlocal
cd /d "%~dp0.."

echo == checking prerequisites ==
where python >nul 2>nul
if errorlevel 1 (
    echo python not found on PATH -- install Python 3.10+ from python.org first.
    exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
    echo npm not found on PATH -- install Node.js 18+ from nodejs.org first.
    exit /b 1
)

echo == creating virtual environment ^(.venv^) ==
python -m venv .venv
call .venv\Scripts\activate.bat

echo == installing Python dependencies ==
python -m pip install --upgrade pip
pip install -e ".[dev]"

echo == installing frontend dependencies ==
call npm --prefix gcs-console install

echo.
echo Setup complete. Next: scripts\run_windows.bat
echo (model\ ships pre-calibrated; data\ -- real flight data -- is not
echo  included, see README.md's Data section if you need it.)
