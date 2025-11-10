"""
Competition metric implementation for Polymer Property Prediction
Implements weighted Mean Absolute Error (wMAE) as defined in the competition rules
"""

try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available, using math module")
    import math
    class np:
        @staticmethod
        def isnan(x):
            return math.isnan(x) if isinstance(x, (int, float)) else [math.isnan(i) for i in x]
        @staticmethod
        def mean(x):
            return sum(x) / len(x)
        @staticmethod
        def abs(x):
            return abs(x)
        @staticmethod
        def sqrt(x):
            return math.sqrt(x)
        nan = float('nan')
        ndarray = list

from typing import Dict, List, Tuple, Union


def calculate_wmae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    n_samples_per_property: Dict[str, int],
    ranges_per_property: Dict[str, float]
) -> float:
    """
    Calculate weighted Mean Absolute Error (wMAE) according to competition rules.
    
    wMAE = (1/|X|) * ΣΣ w_i · |ŷ_i(X) - y_i(X)|
    
    where w_i = (1/r_i) · (K·√(1/n_i)) / (Σ√(1/n_j))
    
    Args:
        y_true: True values, shape (n_samples, n_properties)
        y_pred: Predicted values, shape (n_samples, n_properties)
        target_names: List of property names
        n_samples_per_property: Dict mapping property name to sample count
        ranges_per_property: Dict mapping property name to value range (max - min)
    
    Returns:
        wMAE: Weighted mean absolute error
    """
    K = len(target_names)  # Number of properties (5)
    total_error = 0.0
    total_count = 0
    
    # Calculate weights for each property
    weights = {}
    sqrt_inv_n_sum = sum(np.sqrt(1.0 / n_samples_per_property[prop]) 
                         for prop in target_names)
    
    for i, prop in enumerate(target_names):
        n_i = n_samples_per_property[prop]
        r_i = ranges_per_property[prop]
        
        # w_i = (1/r_i) · (K·√(1/n_i)) / (Σ√(1/n_j))
        scale_norm = 1.0 / r_i
        inverse_sqrt_scaling = K * np.sqrt(1.0 / n_i) / sqrt_inv_n_sum
        weights[prop] = scale_norm * inverse_sqrt_scaling
        
        print(f"{prop:8s}: n={n_i:5d}, range={r_i:7.2f}, weight={weights[prop]:.6f}")
    
    # Calculate weighted MAE
    for i, prop in enumerate(target_names):
        # Get non-NaN values for this property
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() == 0:
            continue
            
        y_true_prop = y_true[mask, i]
        y_pred_prop = y_pred[mask, i]
        
        # MAE for this property
        mae = np.mean(np.abs(y_pred_prop - y_true_prop))
        
        # Weight and accumulate
        weighted_error = weights[prop] * mae * mask.sum()
        total_error += weighted_error
        total_count += mask.sum()
    
    wmae = total_error / total_count if total_count > 0 else 0.0
    return wmae


def get_property_stats(y: np.ndarray, target_names: List[str]) -> Tuple[Dict, Dict]:
    """
    Calculate sample counts and ranges for each property.
    
    Args:
        y: Target values, shape (n_samples, n_properties)
        target_names: List of property names
    
    Returns:
        n_samples_per_property: Dict of sample counts
        ranges_per_property: Dict of ranges (max - min)
    """
    n_samples = {}
    ranges = {}
    
    for i, prop in enumerate(target_names):
        mask = ~np.isnan(y[:, i])
        values = y[mask, i]
        
        n_samples[prop] = mask.sum()
        ranges[prop] = values.max() - values.min() if len(values) > 0 else 1.0
    
    return n_samples, ranges


# Example usage:
if __name__ == "__main__":
    # Example with your polymer data
    target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Sample counts from your augmented data (Phase 2)
    n_samples = {
        'Tg': 9814,    # 54.4% of data
        'FFV': 7030,   # 39.0%
        'Tc': 867,     # 4.8% (rarest!)
        'Density': 1242,  # 6.9%
        'Rg': 1243,    # 6.9%
    }
    
    # Estimated ranges from training data
    ranges = {
        'Tg': 650.0,      # Celsius (e.g., -150 to 500)
        'FFV': 0.35,      # Free volume fraction
        'Tc': 0.5,        # Normalized
        'Density': 0.8,   # g/cm³
        'Rg': 40.0,       # Radius of gyration
    }
    
    # Calculate weights
    K = 5
    sqrt_inv_n_sum = sum(np.sqrt(1.0 / n_samples[prop]) for prop in target_names)
    
    print("Competition Metric Weights:")
    print("="*60)
    for prop in target_names:
        n_i = n_samples[prop]
        r_i = ranges[prop]
        
        scale_norm = 1.0 / r_i
        inverse_sqrt = K * np.sqrt(1.0 / n_i) / sqrt_inv_n_sum
        w_i = scale_norm * inverse_sqrt
        
        print(f"{prop:8s}: n={n_i:5d}, range={r_i:7.2f}, "
              f"√(1/n)={np.sqrt(1/n_i):.4f}, weight={w_i:.6f}")
    
    print("\nKey insight:")
    print("- Tc has HIGHEST weight (rarest property: 867 samples)")
    print("- Tg has LOWEST weight (most common: 9814 samples)")
    print("- This balances the optimization across all properties!")

