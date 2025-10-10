#!/bin/bash

# Training script for polymer property prediction
# Usage: bash scripts/run_training.sh

set -e  # Exit on error

echo "=========================================="
echo "Polymer Properties Prediction - Training"
echo "=========================================="

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate polymer_pred
    
    # Set library path for RDKit
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
else
    echo "Warning: conda not found. Make sure environment is activated manually."
fi

# Run training
echo ""
echo "Starting training..."
echo ""

python src/train.py

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - models/: Trained model checkpoints"
echo "  - results/: Performance metrics and plots"
echo ""
