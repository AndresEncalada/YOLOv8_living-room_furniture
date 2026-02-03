@echo off
title Furniture Detection System - Environment Setup & Startup

echo ===================================================
echo   FURNITURE DETECTION SYSTEM - AUTO SETUP
echo ===================================================

:: 1. Check Python Installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b
)

:: 2. Create Virtual Environment (if missing)
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
) else (
    echo [INFO] Virtual environment detected.
)

:: 3. Activate Virtual Environment
echo [INFO] Activating environment...
call venv\Scripts\activate.bat

:: 4. Install Dependencies
echo [INFO] Checking dependencies...
pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Dependency check failed. Attempting full install...
    pip install -r requirements.txt
)

:: 5. Model Integrity Check
if not exist "models\best_v1.pt" (
    echo.
    echo [CRITICAL ERROR] Model 'models\best_v1.pt' not found!
    echo ---------------------------------------------------
    echo Please run 'Notebook 02' first to generate the base model.
    echo The system cannot start without a base model.
    echo ---------------------------------------------------
    pause
    exit /b
) else (
    echo [INFO] Base model 'best_v1.pt' found. Ready to launch.
)

:: 6. Start Application Server
echo.
echo [SUCCESS] System ready. Starting server at http://127.0.0.1:8000
echo.
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000


pause