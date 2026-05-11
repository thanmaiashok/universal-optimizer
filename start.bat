@echo off
cd /d "%~dp0"

echo ========================================
echo   OptiLLM Universal Optimizer
echo ========================================
echo.

REM Activate venv
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install torch numpy transformers fastapi uvicorn pydantic psutil python-multipart
)

echo.
echo [OK] Starting API server...
echo.
echo   Dashboard: http://localhost:8000/dashboard
echo.      API: http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo.

python -m opti_llm.api.main

pause