"""
Pseudo-Label Generation Module

This module provides tools to generate pseudo-labels for unlabeled polymers
using trained machine learning models. It replicates the strategy used by
the 1st place Kaggle solution.

Key Components:
- generate_pseudolabels.py: Main script to generate pseudo-labels
- ensemble_predictor.py: Framework for ensemble predictions
- analyze_pseudolabels.py: Compare generated vs reference pseudo-labels
- README.md: Comprehensive documentation

Usage:
    python pseudolabel/generate_pseudolabels.py --model_path models/random_forest_v85_best.pkl \\
                                                 --input_data data/PI1M_50000_v2.1.csv
"""

__version__ = "1.0.0"
__author__ = "Polymer Research Team"

from .ensemble_predictor import (
    Predictor,
    RandomForestPredictor,
    EnsemblePseudoLabelGenerator,
    PseudoLabelQualityAnalyzer
)

__all__ = [
    'Predictor',
    'RandomForestPredictor',
    'EnsemblePseudoLabelGenerator',
    'PseudoLabelQualityAnalyzer',
]

