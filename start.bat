@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Study Score Assistant

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 未检测到虚拟环境，正在创建...
    python -m venv .venv
    if errorlevel 1 (
        echo 创建失败：请先安装 Python 3.11+ 并确保 python 命令可用。
        pause
        exit /b 1
    )
)

echo [2/3] 检查并安装依赖（已安装则很快跳过）...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt

if not exist ".env" (
    echo  [注意] 没有找到 .env，请先复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY
    echo         否则页面会提示"未配置 API Key"。
)

echo [3/3] 启动服务，浏览器将自动打开 http://127.0.0.1:8000
start "" /b cmd /c "timeout /t 3 >nul && start "" http://127.0.0.1:8000"
echo.
echo 关闭本窗口即停止服务（或按 Ctrl+C）。
echo.
".venv\Scripts\python.exe" -m uvicorn app.main:app --port 8000
pause
