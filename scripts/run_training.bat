@echo off
REM Training script for polymer property prediction (Windows)
REM Usage: run_training.bat

echo ==========================================
echo Polymer Properties Prediction - Training
echo ==========================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Activate conda environment
echo Activating conda environment...
call conda activate polymer_pred
if errorlevel 1 (
    echo Error: Could not activate conda environment 'polymer_pred'
    echo Please create it first:
    echo   conda create -n polymer_pred python=3.10 -y
    echo   conda activate polymer_pred
    echo   conda install -c conda-forge rdkit -y
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Run training
echo.
echo Starting training...
echo.

python src\train.py

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Training failed! Check errors above.
    echo ==========================================
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Training complete!
echo ==========================================
echo.
echo Results saved to:
echo   - models\: Trained model checkpoints
echo   - results\: Performance metrics and plots
echo.

pause

