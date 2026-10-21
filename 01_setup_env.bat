@echo off
chcp 65001 > nul
echo ==============================================================================
echo [PLANT DOCTOR POC] Initializing Windows Environment in C:\Work\Plant Doctor
echo ==============================================================================
echo.

:: Step 1: Check Python availability
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your system PATH!
    echo Please install Python 3.10 or 3.11 from python.org and check "Add to PATH".
    pause
    exit /b 1
)

:: Step 2: Create directory structure
echo [1/5] Creating workspace directories in C:\Work\Plant Doctor...
if not exist "datasets" mkdir "datasets"
if not exist "models" mkdir "models"
if not exist "test_photos" mkdir "test_photos"
if not exist "output" mkdir "output"

:: Step 3: Initialize Virtual Environment
echo [2/5] Creating isolated Python virtual environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
    echo Virtual environment created successfully.
) else (
    echo Virtual environment .venv already exists.
)

:: Step 4: Activate Virtual Environment
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Step 5: Upgrade pip and install wheel
echo [4/5] Upgrading pip package manager...
python -m pip install --upgrade pip setuptools wheel

:: Step 6: Install PyTorch and Ultralytics
echo [5/5] Installing PyTorch, Ultralytics YOLOv8, Roboflow SDK, and dependencies...
pip install -r requirements.txt

echo.
echo ==============================================================================
echo [SUCCESS] Environment setup complete!
echo To begin, run:
echo   1. python 02_download_and_structure_dataset.py
echo   2. python 03_finetune_yolov8.py
echo   3. python 04_infer_offline_phone_photos.py
echo ==============================================================================
pause
