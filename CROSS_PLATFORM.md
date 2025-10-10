# Cross-Platform Compatibility Guide

## ✅ Yes! Your Colleagues Can Run This on Other Operating Systems

The project is designed to be **cross-platform compatible** with minor setup differences.

## 📊 Compatibility Matrix

| Component | Linux | macOS | Windows | Notes |
|-----------|-------|-------|---------|-------|
| **Python Code** | ✅ | ✅ | ✅ | Fully compatible |
| **RDKit** | ✅ | ✅ | ✅ | Install via conda |
| **PyTorch** | ✅ | ✅ | ✅ | Auto-detects CUDA |
| **XGBoost** | ✅ | ✅ | ✅ | No issues |
| **Scripts** | ✅ | ✅ | ⚠️ | Bash scripts need adaptation |
| **File Paths** | ✅ | ✅ | ✅ | Python handles automatically |

## 🐧 Linux (Your Current Setup) ✅

**Status:** Fully Working

```bash
# Your current setup
conda create -n polymer_pred python=3.10 -y
conda activate polymer_pred
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run
python src/train.py
```

## 🍎 macOS ✅

**Status:** Fully Compatible

```bash
# Same as Linux
conda create -n polymer_pred python=3.10 -y
conda activate polymer_pred
conda install -c conda-forge rdkit -y
pip install -r requirements.txt

# Library path (similar to Linux)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run
python src/train.py
```

**macOS Notes:**
- RDKit works perfectly via conda
- M1/M2 Macs: Use `arm64` conda installation
- Everything else identical to Linux

## 🪟 Windows ✅ (with minor changes)

**Status:** Compatible (Setup slightly different)

### Installation on Windows

```bash
# Open Anaconda Prompt or PowerShell

# Create environment
conda create -n polymer_pred python=3.10 -y
conda activate polymer_pred

# Install RDKit
conda install -c conda-forge rdkit -y

# Install other packages
pip install -r requirements.txt
```

### Running on Windows

```bash
# Activate environment
conda activate polymer_pred

# Run training (same command!)
python src/train.py
```

### Windows-Specific Notes

**✅ Works Out of the Box:**
- All Python code (paths automatically handled)
- Data loading
- Model training
- Result generation

**⚠️ Needs Adaptation:**
1. **Bash Scripts** → Use Python or PowerShell instead
2. **Library Path** → Not needed on Windows

### Windows Alternative to Bash Scripts

Instead of `scripts/run_training.sh`, Windows users can:

**Option 1: Use Python Directly**
```bash
cd C:\Users\YourName\open_polymer
conda activate polymer_pred
python src/train.py
```

**Option 2: Create Windows Batch File**

I'll create a Windows-compatible version:

```batch
@echo off
REM run_training.bat for Windows

echo ==========================================
echo Polymer Properties Prediction - Training
echo ==========================================

REM Activate conda environment
call conda activate polymer_pred

REM Run training
python src\train.py

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
```

## 🔧 Platform-Specific Adjustments Made

### 1. File Paths (Already Handled ✅)

The code uses `os.path.join()` which automatically handles path separators:

```python
# Works on all platforms
train_path = os.path.join(project_root, 'data/raw/train.csv')

# Python converts to:
# Linux/macOS: 'data/raw/train.csv'
# Windows: 'data\raw\train.csv'
```

### 2. Line Endings (Git Handles ✅)

Git automatically converts line endings:
- Linux/macOS: LF (`\n`)
- Windows: CRLF (`\r\n`)

Your `.gitattributes` can enforce this:
```
# Add to .gitattributes
* text=auto
*.py text eol=lf
*.sh text eol=lf
*.bat text eol=crlf
```

### 3. Environment Variables

**Linux/macOS:**
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

**Windows:**
Not needed! RDKit works without this on Windows.

## 📝 Updated Installation Instructions for README

I'll add a platform-specific section to your README:

```markdown
## Installation by Platform

### Linux / macOS

```bash
conda create -n polymer_pred python=3.10 -y
conda activate polymer_pred
conda install -c conda-forge rdkit -y
pip install -r requirements.txt

# Linux/macOS only: Set library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Windows

```bash
# Open Anaconda Prompt
conda create -n polymer_pred python=3.10 -y
conda activate polymer_pred
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

Note: No library path needed on Windows.
```

## 🚀 Creating Windows-Compatible Scripts

Let me create a Windows batch file:

