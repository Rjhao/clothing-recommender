@echo off
chcp 65001 >nul
title Clothing Recommender - Backend

echo Starting backend services (no Docker)...

REM --- 启动主后端 API (port 8000) ---
start "Clothing-API" cmd /k "cd /d D:\clothing-recommender\api && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

REM --- 启动虚拟试穿服务 (port 8001) ---
start "Clothing-TryOn" cmd /k "cd /d D:\clothing-recommender && python -m uvicorn tryon_api:app --host 0.0.0.0 --port 8001"

REM --- 启动前端 (port 5173) ---
start "Clothing-Frontend" cmd /k "cd /d D:\clothing-recommender\frontend && npm run dev"

echo Done! API=8000, TryOn=8001, Frontend=5173
pause
