@echo off
title Portfolio Optimization Tool
echo ===================================================
echo   Starting Portfolio Optimization Tool...
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in your PATH.
    echo Please install Python 3.9 or newer from https://www.python.org/
    pause
    exit /b
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo [1/3] Creating virtual environment (venv)...
    python -m venv venv
)

:: Activate venv
echo [2/3] Activating virtual environment...
call venv\Scripts\activate

:: Install requirements
echo [3/3] Checking and installing dependencies...
pip install -q -r requirements.txt

:: Run the app
echo.
echo ===================================================
echo   Launching Web Application in your browser...
echo   (Keep this window open while using the app)
echo ===================================================
streamlit run app.py

pause