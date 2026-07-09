@echo off
chcp 65001 >nul
title ComfyUI - Virtual Try-On Engine

REM =====================================================================
REM ComfyUI 启动脚本 (RTX 5060 8GB 优化)
REM 模型: D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\v1-5-pruned.safetensors
REM =====================================================================

cd /d "D:\ComfyUI\ComfyUI_windows_portable\ComfyUI"

REM --lowvram         : 启用低显存模式 (8GB GPU)
REM --force-fp16      : 强制 FP16 推理，显存减半
REM --preview-method latent2rgb : 更清晰的预览
REM --listen 0.0.0.0 : 允许外部访问
REM --dont-print-server : 不打印服务器信息到窗口

python main.py --lowvram --force-fp16 --preview-method latent2rgb --port 8188 --listen 0.0.0.0 --dont-print-server

pause
