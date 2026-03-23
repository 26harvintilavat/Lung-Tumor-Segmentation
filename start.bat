@echo off
setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
set VENV_PYTHON="%PROJECT_DIR%venv\Scripts\python.exe"

echo.
echo ========================================
echo   LungSeg AI -- Starting up (Debug Mode)
echo ========================================
echo.

:: -- Check if venv python exists --
if not exist %VENV_PYTHON% (
    echo [ERROR] Virtual environment not found at %VENV_PYTHON%
    pause
    exit /b 1
)

:: -- Start API --
:: We use cmd /k so the window stays open if the API crashes
echo ^-^> Starting API on http://127.0.0.1:8000 ...
start "LungSeg API" cmd /k "cd /d %PROJECT_DIR% && %VENV_PYTHON% -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

:: -- Start Frontend --
echo ^-^> Starting Frontend on http://127.0.0.1:3000 ...
start "LungSeg Frontend" cmd /k "cd /d %PROJECT_DIR%frontend && %VENV_PYTHON% -m http.server 3000 --bind 127.0.0.1"

echo.
echo ^-^> Waiting 5 seconds for model to load...
timeout /t 5 /nobreak > nul

echo ^-^> Opening Tool...
start http://127.0.0.1:3000/tool.html

echo.
echo ========================================
echo   If you see "Failed to fetch":
echo   1. Check the "LungSeg API" window for errors.
echo   2. Ensure no other program is using port 8000.
echo ========================================
echo.
pause
