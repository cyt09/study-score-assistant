@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Verification

echo ============================================
echo  1/2  单元与流程测试（不需要 API Key）
echo ============================================
".venv\Scripts\python.exe" -m unittest tests.test_pipeline -v

echo.
echo ============================================
echo  2/2  数值验证（独立实现 + 手算答案核对）
echo ============================================
".venv\Scripts\python.exe" scripts\verify_calc.py

echo.
echo 如需重新生成 docs\验证记录.md（需服务已启动）：
echo    ".venv\Scripts\python.exe" scripts\run_validation.py --base http://127.0.0.1:8000
echo.
pause
