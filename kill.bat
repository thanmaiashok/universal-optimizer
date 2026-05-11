@echo off
REM PredycatAI Universal Optimizer - Kill Script

echo.
echo [INFO] Stopping PredycatAI...

REM Use venv Python if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Kill by PID
if exist .predycat_pid (
    for /f "delims=" %%i in (.predycat_pid) do (
        taskkill /pid %%i /f 2>nul
    )
    del .predycat_pid
)

echo [OK] PredycatAI stopped