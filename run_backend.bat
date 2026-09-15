@echo off
title ViHOS Backend (FastAPI)
echo ========================================================
echo   KHOI DONG BACKEND FASTAPI - VIHOS GUARD (PORT 8000)
echo ========================================================
cd /d "%~dp0"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
