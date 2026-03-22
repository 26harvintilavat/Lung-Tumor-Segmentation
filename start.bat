@echo off
echo Starting LungSeg AI...

:: Get project directory
set PROJECT_DIR=%~dp0

:: Activate virtual environment
call %PROJECT_DIR%venv\Scripts\activate.bat

:: Start API in background
echo Starting API...
start "API" cmd /k "cd %PROJECT_DIR% && python api/main.py"

:: Wait for API to load
timeout /t 3 /nobreak > nul

:: Start frontend server
echo Starting Frontend...
start "Frontend" cmd /k "cd %PROJECT_DIR%frontend && python -m http.server 3000"

:: Wait for frontend to load
timeout /t 2 /nobreak > nul

:: Open browser
echo Opening browser...
start http://localhost:3000/index.html

echo.
echo LungSeg AI is running!
echo    Landing page: http://localhost:3000/index.html
echo    API:          http://localhost:8000
echo    API Docs:     http://localhost:8000/docs
echo.
echo Close the terminal windows to stop everything.