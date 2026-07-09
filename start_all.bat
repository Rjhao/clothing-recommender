@echo off
chcp 65001 >nul
title Clothing Recommender System

echo ============================================
echo   服装推荐系统 - 一键启动
echo ============================================

cd /d "%~dp0"

REM --- 启动基础设施 (Neo4j/MySQL/Redis via Docker) ---
echo [1/5] 启动数据库服务 (Neo4j, MySQL, Redis)...
start "Clothing-DB" cmd /k "docker compose up -d"
timeout /t 5 /nobreak >nul

REM --- 启动主后端 API (port 8000) ---
echo [2/5] 启动主后端 API (port 8000)...
start "Clothing-API" cmd /k "cd /d %~dp0api ^&^& python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

REM --- 启动虚拟试穿服务 (port 8001, MediaPipe pose-aware) ---
echo [3/5] 启动虚拟试穿服务 (port 8001)...
start "Clothing-TryOn" cmd /k "cd /d %~dp0 ^&^& python -m uvicorn tryon_api:app --host 0.0.0.0 --port 8001"

REM --- 启动 ComfyUI (port 8188, RTX GPU 加速) ---
echo [4/5] 启动 ComfyUI AI 引擎 (port 8188, RTX 5060)...
if exist "D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\main.py" (
    start "ComfyUI" cmd /k "cd /d D:\ComfyUI\ComfyUI_windows_portable\ComfyUI ^&^& python main.py --lowvram --force-fp16 --preview-method latent2rgb --port 8188 --listen 0.0.0.0 --dont-print-server"
) else (
    echo    (ComfyUI 未安装，跳过)
)

REM --- 启动前端 Vite 开发服务器 (port 3000) ---
echo [5/5] 启动前端 (port 3000)...
start "Clothing-Frontend" cmd /k "cd /d %~dp0frontend ^&^& npm run dev"

echo.
echo ============================================
echo   所有服务已启动！
echo ============================================
echo   主后端 API:   http://localhost:8000
echo   虚拟试穿:     http://localhost:8001
echo   ComfyUI:      http://localhost:8188
echo   前端:         http://localhost:3000
echo   Neo4j:        http://localhost:7474
echo ============================================
pause
