@echo off
REM PredycatAI Universal Optimizer - Setup Script
REM Run this once to install everything

echo.
echo ========================================
echo   PredycatAI Setup
echo ========================================
echo.

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found

echo.
echo [2/4] Creating directories...
if not exist outputs mkdir outputs
if not exist outputs\models mkdir outputs\models
if not exist outputs\reports mkdir outputs\reports
if not exist outputs\logs mkdir outputs\logs
if not exist models mkdir models
if not exist cache mkdir cache
echo [OK] Directories created

echo.
echo [3/4] Installing core dependencies...
pip install --quiet torch numpy
if errorlevel 1 (
    echo [ERROR] Failed to install torch/numpy
    pause
    exit /b 1
)
echo [OK] torch, numpy

echo.
echo [4/4] Installing PredycatAI dependencies...
pip install --quiet transformers fastapi uvicorn pydantic psutil
if errorlevel 1 (
    echo [WARNING] Some optional deps failed
)
echo [OK] Done

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To start: double-click start.bat
echo.
pause