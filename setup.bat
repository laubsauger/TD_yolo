@echo off
REM Windows batch file to setup virtual environment
REM This handles the cross-platform Python discovery

echo TD_yolo Virtual Environment Setup (Windows)
echo ==========================================

REM Try Python Launcher first (most reliable on Windows)
py -3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python via Python Launcher
    py -3 setup_env.py
    goto :end
)

REM Try python command
python --version >nul 2>&1
if %errorlevel% == 0 (
    python --version | findstr "Python 3." >nul
    if %errorlevel% == 0 (
        echo Found Python 3
        python setup_env.py
        goto :end
    )
)

REM Try python3 command
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found python3
    python3 setup_env.py
    goto :end
)

REM No Python found
echo ERROR: Python 3.9 or later not found!
echo.
echo Please install Python from https://python.org
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:end
pause