"""
AutoGluon predictor wrapper for production inference.

This module provides a unified interface to the pre-trained AutoGluon models
trained in train_autogluon_production.py.

The AutoGluon models are stacked ensembles (WeightedEnsemble_L2) that intelligently
combine multiple base models (RF, XGBoost, LightGBM, etc.) for robust predictions.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class AutoGluonPredictor:
    """
    Wrapper for pre-trained AutoGluon models.
    
    🥇 AutoGluon Production Configuration:
    - Models: WeightedEnsemble_L2 (stacked ensemble of 8 base models)
    - Features: 34 total (10 simple + 11 hand-crafted + 13 RDKit)
    - Training: 60K+ samples, 1800s per property, medium_quality preset
    - Targets: Tg, FFV, Tc, Density, Rg
    """
    
    def __init__(self, model_dir: str = "models/autogluon_production", cpu_only: bool = True):
        """
        Initialize AutoGluon predictor.
        
        Args:
            model_dir: Directory containing trained AutoGluon models
            cpu_only: Force CPU-only mode (recommended for Apple Silicon)
        """
        self.model_dir = Path(model_dir)
        self.cpu_only = cpu_only
        self.predictors = {}
        self.feature_names = None
        self.target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        if cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MPS_ENABLED'] = '0'
    
    def load(self) -> bool:
        """
        Load all pre-trained AutoGluon models.
        
        Returns:
            bool: True if all models loaded successfully
        """
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            print("❌ AutoGluon not installed. Install with: pip install autogluon")
            return False
        
        print("\n" + "="*70)
        print("LOADING AUTOGLUON MODELS")
        print("="*70)
        
        all_loaded = True
        for target in self.target_cols:
            target_path = self.model_dir / target
            
            if not target_path.exists():
                print(f"❌ {target}: Model directory not found at {target_path}")
                all_loaded = False
                continue
            
            try:
                print(f"\n📂 Loading {target}...", end=" ")
                predictor = TabularPredictor.load(str(target_path))
                self.predictors[target] = predictor
                
                # Store feature names from first loaded model
                if self.feature_names is None and hasattr(predictor, 'features'):
                    self.feature_names = predictor.features
                
                print(f"✅")
                print(f"   - Label: {predictor.label}")
                print(f"   - Features: {len(predictor.features) if hasattr(predictor, 'features') else '?'}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                all_loaded = False
        
        if all_loaded:
            print("\n" + "="*70)
            print("✅ ALL AUTOGLUON MODELS LOADED SUCCESSFULLY!")
            print("="*70)
        
        return all_loaded
    
    def predict(self, X: np.ndarray, target: str) -> np.ndarray:
        """
        Generate predictions for a single target property.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            target: Target property name (Tg, FFV, Tc, Density, or Rg)
        
        Returns:
            Predictions array (n_samples,)
        """
        if target not in self.predictors:
            print(f"❌ No model loaded for {target}")
            return np.zeros(len(X))
        
        try:
            predictor = self.predictors[target]
            
            # Handle NaN/inf in input
            X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Convert to DataFrame if needed (AutoGluon expects this)
            if isinstance(X_clean, np.ndarray):
                if self.feature_names:
                    X_df = pd.DataFrame(X_clean, columns=self.feature_names)
                else:
                    # Fallback: generic column names
                    X_df = pd.DataFrame(X_clean, columns=[f'feat_{i}' for i in range(X_clean.shape[1])])
            else:
                X_df = X_clean
            
            # Predict
            predictions = predictor.predict(X_df)
            
            # Handle Series/DataFrame output
            if isinstance(predictions, (pd.Series, pd.DataFrame)):
                predictions = predictions.values.flatten()
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction failed for {target}: {e}")
            return np.zeros(len(X))
    
    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for all target properties.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predictions array (n_samples, n_targets)
        """
        predictions = np.zeros((len(X), len(self.target_cols)))
        
        for i, target in enumerate(self.target_cols):
            predictions[:, i] = self.predict(X, target)
        
        return predictions


def apply_tg_transformation(tg_values: np.ndarray) -> np.ndarray:
    """
    Apply Tg transformation discovered by 2nd place winner.
    
    This corrects for distribution shift between train and test data.
    Transformation: (9/5) × Tg + 45
    
    Similar to Celsius → Fahrenheit conversion, suggesting a scale/units issue.
    
    Args:
        tg_values: Tg predictions array
    
    Returns:
        Transformed Tg values
    """
    return (9/5) * tg_values + 45

