@echo off
REM OptiLLM Universal Optimizer - Kill Script

echo.
echo [INFO] Stopping OptiLLM...

REM Use venv Python if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Kill by PID
if exist .optillm_pid (
    for /f "delims=" %%i in (.optillm_pid) do (
        taskkill /pid %%i /f 2>nul
    )
    del .optillm_pid
)

echo [OK] OptiLLM stopped