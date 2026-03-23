@echo off
setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0

echo.
echo ========================================
echo   LungSeg AI -- Starting up
echo ========================================
echo.

:: -- Check venv exists --
if not exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo.
    echo   Run this first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: -- Check model weights exist --
if not exist "%PROJECT_DIR%checkpoints\best_model.pth" (
    echo [ERROR] Model weights not found at checkpoints\best_model.pth
    echo.
    echo   The API cannot start without the trained model.
    echo   Ask the author for best_model.pth and place it at:
    echo   %PROJECT_DIR%checkpoints\best_model.pth
    echo.
    pause
    exit /b 1
)

:: -- Activate venv --
call "%PROJECT_DIR%venv\Scripts\activate.bat"

:: -- Get local IP --
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    set LOCAL_IP=%%a
    goto :got_ip
)
:got_ip
set LOCAL_IP=%LOCAL_IP: =%

:: -- Start API in new window --
echo ^-^> Starting API on http://0.0.0.0:8000 ...
start "LungSeg API" cmd /k "cd /d %PROJECT_DIR% && uvicorn api.main:app --host 0.0.0.0 --port 8000"

:: -- Wait for API to boot --
echo ^-^> Waiting for API to load model...
timeout /t 6 /nobreak > nul

:: -- Start frontend server in new window --
echo ^-^> Starting frontend on http://localhost:3000 ...
start "LungSeg Frontend" cmd /k "cd /d %PROJECT_DIR%frontend && python -m http.server 3000"

:: -- Wait a moment then open browser --
timeout /t 2 /nobreak > nul
echo ^-^> Opening browser...
start http://localhost:3000/index.html

echo.
echo ========================================
echo   LungSeg AI is running
echo.
echo   Landing page : http://localhost:3000/index.html
echo   Tool         : http://localhost:3000/tool.html
echo   API docs     : http://localhost:8000/docs
echo.
echo   Share with others on your network:
echo   Set API_BASE_URL = http://%LOCAL_IP%:8000 in frontend/app.js
echo ========================================
echo.
echo   Close the API and Frontend windows to stop.
echo.
pause
