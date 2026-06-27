@echo off
cd /d "%~dp0"
echo Starting VSM Analyzer Backend on http://localhost:8000
".venv\Scripts\python.exe" -m uvicorn backend.main:app --port 8000 --reload --reload-dir backend
pause
