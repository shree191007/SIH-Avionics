@echo off
REM Starts the FastAPI backend (src\replan_to_learn\api\app.py) on :8000.
setlocal
cd /d "%~dp0.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=%cd%\src;%PYTHONPATH%
python -m uvicorn replan_to_learn.api.app:app --host 0.0.0.0 --port 8000 --reload --app-dir "%cd%\src" --reload-dir "%cd%\src"
