@echo off
title ViHOS Guard Launcher
echo ========================================================
echo   KHOI DONG HE THONG VIHOS GUARD (BACKEND + FRONTEND)
echo ========================================================
cd /d "%~dp0"

echo [1/2] Khoi dong Backend FastAPI (Port 8000)...
start "ViHOS Backend API" cmd /k "run_backend.bat"

timeout /t 2 /nobreak >nul

echo [2/2] Khoi dong Frontend React Vite (Port 5173)...
start "ViHOS Frontend UI" cmd /k "run_frontend.bat"

echo ========================================================
echo   DA KHOI DONG THANH CONG!
echo   - Backend API Docs: http://localhost:8000/docs
echo   - Frontend UI:      http://localhost:5173
echo ========================================================
timeout /t 3 >nul
start http://localhost:5173
