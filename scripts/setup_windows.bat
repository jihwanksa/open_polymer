@echo off
REM Setup script for Windows
REM Run this first before using the project

echo ==========================================
echo Open Polymer - Windows Setup
echo ==========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if errorlevel 1 (
    echo Error: Conda not found!
    echo Please install Anaconda or Miniconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Step 1: Creating conda environment...
call conda create -n polymer_pred python=3.10 -y
if errorlevel 1 (
    echo Error creating environment
    pause
    exit /b 1
)

echo.
echo Step 2: Activating environment...
call conda activate polymer_pred
if errorlevel 1 (
    echo Error activating environment
    pause
    exit /b 1
)

echo.
echo Step 3: Installing RDKit...
call conda install -c conda-forge rdkit -y
if errorlevel 1 (
    echo Error installing RDKit
    pause
    exit /b 1
)

echo.
echo Step 4: Installing Python packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing packages
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To use the project:
echo   1. conda activate polymer_pred
echo   2. python src\train.py
echo.
echo Or double-click: scripts\run_training.bat
echo.

pause

