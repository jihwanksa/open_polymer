"""
Ensemble prediction utilities for pseudo-label generation.

This module provides a framework for combining predictions from multiple models
(Random Forest, BERT, AutoGluon, Uni-Mol, etc.) to generate robust pseudo-labels.

The 1st place solution used BERT + AutoGluon + Uni-Mol ensemble for their pseudo-labels.
This module provides a template for building similar ensembles.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class Predictor(ABC):
    """Base class for different model predictors"""
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Predictions of shape (n_samples, n_targets)
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return model name for logging"""
        pass


class RandomForestPredictor(Predictor):
    """Random Forest ensemble predictor using v85 trained model"""
    
    def __init__(self, model_data: Dict):
        """
        Args:
            model_data: Loaded model dictionary with 'models' and 'scalers'
        """
        self.models = model_data.get('models', {})
        self.scalers = model_data.get('scalers', {})
        self.target_names = list(self.models.keys())
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate Random Forest predictions"""
        predictions = np.zeros((len(features), len(self.target_names)))
        
        for i, target in enumerate(self.target_names):
            try:
                if target in self.models and target in self.scalers:
                    scaler = self.scalers[target]
                    ensemble_models = self.models[target]
                    
                    # Scale features
                    X_scaled = scaler.transform(features)
                    
                    # Ensemble prediction (average across 5 models)
                    ensemble_preds = np.array([model.predict(X_scaled) for model in ensemble_models])
                    pred = ensemble_preds.mean(axis=0)
                    predictions[:, i] = pred
                else:
                    predictions[:, i] = 0.0
            except Exception as e:
                print(f"   Warning: {target} prediction failed: {e}")
                predictions[:, i] = 0.0
        
        return predictions
    
    def name(self) -> str:
        return "RandomForest_v85"


class EnsemblePseudoLabelGenerator:
    """
    Combine predictions from multiple models to generate robust pseudo-labels.
    
    Strategy from 1st place solution:
    - Use multiple models (BERT, AutoGluon, Uni-Mol, etc.)
    - Average their predictions for each property
    - Apply post-processing (Tg transformation, etc.)
    """
    
    def __init__(self, predictors: List[Predictor]):
        """
        Args:
            predictors: List of Predictor instances
        """
        self.predictors = predictors
        self.target_names = None
    
    def generate(self, features: np.ndarray, weights: List[float] = None) -> np.ndarray:
        """
        Generate ensemble pseudo-labels by averaging predictions.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            weights: Optional weights for each model (should sum to 1)
        
        Returns:
            Ensemble predictions of shape (n_samples, n_targets)
        """
        if not self.predictors:
            raise ValueError("No predictors provided")
        
        if weights is None:
            # Equal weighting
            weights = [1.0 / len(self.predictors)] * len(self.predictors)
        else:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
        
        print(f"\n🔄 Generating ensemble predictions from {len(self.predictors)} models:")
        for predictor, weight in zip(self.predictors, weights):
            print(f"   - {predictor.name()}: weight={weight:.3f}")
        
        # Collect predictions from all models
        all_predictions = []
        for i, predictor in enumerate(self.predictors):
            print(f"\n   Predicting with {predictor.name()}...")
            pred = predictor.predict(features)
            all_predictions.append(pred)
            print(f"   ✅ Shape: {pred.shape}, Mean: {pred.mean(axis=0)}")
        
        # Weighted average
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples, n_targets)
        weights = np.array(weights).reshape(-1, 1, 1)  # Shape: (n_models, 1, 1)
        
        ensemble_pred = (all_predictions * weights).sum(axis=0)  # Shape: (n_samples, n_targets)
        
        return ensemble_pred
    
    def apply_post_processing(self, predictions: np.ndarray, target_names: List[str]) -> np.ndarray:
        """
        Apply domain-specific post-processing to predictions.
        
        - Tg transformation: (9/5) × Tg + 45 (2nd place solution)
        - Clip Density to [0.8, 1.5]
        - Clip Tg to [-100, 500]
        """
        predictions = predictions.copy()
        
        tg_idx = target_names.index('Tg') if 'Tg' in target_names else None
        density_idx = target_names.index('Density') if 'Density' in target_names else None
        
        # Tg transformation
        if tg_idx is not None:
            predictions[:, tg_idx] = (9/5) * predictions[:, tg_idx] + 45
            print(f"   ✅ Applied Tg transformation: (9/5) × Tg + 45")
        
        # Clipping for physical constraints
        if density_idx is not None:
            predictions[:, density_idx] = np.clip(predictions[:, density_idx], 0.8, 1.5)
            print(f"   ✅ Clipped Density to [0.8, 1.5]")
        
        if tg_idx is not None:
            predictions[:, tg_idx] = np.clip(predictions[:, tg_idx], -100, 500)
            print(f"   ✅ Clipped Tg to [-100, 500]")
        
        return predictions


class PseudoLabelQualityAnalyzer:
    """Analyze quality of generated pseudo-labels"""
    
    @staticmethod
    def compare_with_reference(generated: np.ndarray, reference: np.ndarray, 
                               target_names: List[str]) -> Dict:
        """
        Compare generated pseudo-labels with reference dataset
        
        Args:
            generated: Generated predictions shape (n_samples, n_targets)
            reference: Reference dataset predictions shape (n_samples, n_targets)
            target_names: List of target property names
        
        Returns:
            Dictionary with comparison statistics
        """
        stats = {}
        
        for i, target in enumerate(target_names):
            gen = generated[:, i]
            ref = reference[:, i]
            
            stats[target] = {
                'mae': np.mean(np.abs(gen - ref)),
                'rmse': np.sqrt(np.mean((gen - ref) ** 2)),
                'correlation': np.corrcoef(gen, ref)[0, 1],
                'gen_mean': gen.mean(),
                'ref_mean': ref.mean(),
                'gen_std': gen.std(),
                'ref_std': ref.std(),
            }
        
        return stats
    
    @staticmethod
    def print_stats(stats: Dict):
        """Print quality statistics"""
        print("\n📊 Pseudo-Label Quality Analysis:")
        print(f"{'Property':<12} {'MAE':>10} {'RMSE':>10} {'Corr':>8}")
        print("-" * 42)
        
        for target, s in stats.items():
            print(f"{target:<12} {s['mae']:>10.4f} {s['rmse']:>10.4f} {s['correlation']:>8.3f}")


def demo_ensemble_generation():
    """Demo: How to use ensemble predictor with multiple models"""
    print("\n" + "="*80)
    print("ENSEMBLE PSEUDO-LABEL GENERATION DEMO")
    print("="*80)
    
    print("""
Example usage with multiple models:

    import pickle
    from ensemble_predictor import RandomForestPredictor, EnsemblePseudoLabelGenerator
    
    # Load models
    with open('models/random_forest_v85_best.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # Create predictors
    predictors = [
        RandomForestPredictor(rf_model),
        # BertPredictor(bert_model),       # TODO: Implement BERT predictor
        # AutoGluonPredictor(ag_model),     # TODO: Implement AutoGluon predictor
        # UniMolPredictor(unimol_model),    # TODO: Implement Uni-Mol predictor
    ]
    
    # Generate ensemble pseudo-labels
    generator = EnsemblePseudoLabelGenerator(predictors)
    weights = [0.33, 0.33, 0.34]  # Equal weighting for 3 models
    
    # Generate predictions
    predictions = generator.generate(features, weights=weights)
    
    # Apply post-processing
    target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    predictions = generator.apply_post_processing(predictions, target_names)
    
    # Analyze quality
    analyzer = PseudoLabelQualityAnalyzer()
    stats = analyzer.compare_with_reference(predictions, reference_predictions, target_names)
    analyzer.print_stats(stats)
    """)


if __name__ == "__main__":
    demo_ensemble_generation()

