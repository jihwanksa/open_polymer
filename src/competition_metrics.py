"""
Official competition evaluation metrics for NeurIPS Open Polymer Prediction 2025
Implements weighted Mean Absolute Error (wMAE)
"""
import numpy as np
import pandas as pd


def calculate_weights(y_true, property_names):
    """
    Calculate reweighting factors w_i for each property
    
    w_i = (1/r_i) * (K * sqrt(1/n_i)) / sum_j(sqrt(1/n_j))
    
    where:
    - n_i = number of available values for property i
    - r_i = max(y_i) - min(y_i) = range of property i (from test data)
    - K = total number of tasks (5 properties)
    """
    K = len(property_names)
    weights = {}
    
    # Calculate n_i and r_i for each property
    property_stats = {}
    for i, prop in enumerate(property_names):
        # Get non-NaN values
        valid_mask = ~np.isnan(y_true[:, i])
        valid_values = y_true[valid_mask, i]
        
        n_i = len(valid_values)
        if n_i > 0:
            r_i = np.max(valid_values) - np.min(valid_values)
            property_stats[prop] = {'n': n_i, 'r': r_i}
        else:
            property_stats[prop] = {'n': 0, 'r': 1.0}  # Avoid division by zero
    
    # Calculate denominator: sum of sqrt(1/n_j) for all properties
    denominator = sum(np.sqrt(1.0 / stats['n']) for stats in property_stats.values() if stats['n'] > 0)
    
    # Calculate weight for each property
    for prop, stats in property_stats.items():
        if stats['n'] > 0 and stats['r'] > 0:
            numerator = K * np.sqrt(1.0 / stats['n'])
            w_i = (1.0 / stats['r']) * (numerator / denominator)
            weights[prop] = w_i
        else:
            weights[prop] = 0.0
    
    return weights


def weighted_mae(y_true, y_pred, property_names, weights=None):
    """
    Calculate weighted Mean Absolute Error (wMAE)
    
    wMAE = (1/|X|) * sum_X sum_i w_i * |y_hat_i(X) - y_i(X)|
    
    Args:
        y_true: array of shape (n_samples, n_properties) with true values
        y_pred: array of shape (n_samples, n_properties) with predictions
        property_names: list of property names
        weights: dict of weights per property (if None, will calculate)
    
    Returns:
        wMAE score (lower is better)
    """
    if weights is None:
        weights = calculate_weights(y_true, property_names)
    
    n_samples = len(y_true)
    total_weighted_error = 0.0
    
    # Sum over all samples and properties
    for i, prop in enumerate(property_names):
        # Only consider samples where true value is available (not NaN)
        valid_mask = ~np.isnan(y_true[:, i])
        
        if valid_mask.sum() > 0:
            # Calculate MAE for this property
            mae_i = np.mean(np.abs(y_pred[valid_mask, i] - y_true[valid_mask, i]))
            # Weight it
            total_weighted_error += weights[prop] * mae_i * valid_mask.sum()
    
    # Average over all samples
    wMAE = total_weighted_error / n_samples
    
    return wMAE, weights


def evaluate_model_competition(y_true, y_pred, property_names):
    """
    Evaluate model using competition metric and return detailed results
    
    Args:
        y_true: array of shape (n_samples, n_properties)
        y_pred: array of shape (n_samples, n_properties)
        property_names: list of property names
    
    Returns:
        dict with wMAE score, weights, and per-property metrics
    """
    # Calculate weights
    weights = calculate_weights(y_true, property_names)
    
    # Calculate overall wMAE
    wMAE_score, _ = weighted_mae(y_true, y_pred, property_names, weights)
    
    # Calculate per-property metrics
    property_metrics = {}
    for i, prop in enumerate(property_names):
        valid_mask = ~np.isnan(y_true[:, i])
        
        if valid_mask.sum() > 0:
            y_true_prop = y_true[valid_mask, i]
            y_pred_prop = y_pred[valid_mask, i]
            
            mae = np.mean(np.abs(y_pred_prop - y_true_prop))
            rmse = np.sqrt(np.mean((y_pred_prop - y_true_prop) ** 2))
            
            # R² score
            ss_res = np.sum((y_true_prop - y_pred_prop) ** 2)
            ss_tot = np.sum((y_true_prop - np.mean(y_true_prop)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            property_metrics[prop] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'weight': weights[prop],
                'n_samples': int(valid_mask.sum()),
                'weighted_MAE': weights[prop] * mae
            }
    
    return {
        'wMAE': wMAE_score,
        'weights': weights,
        'property_metrics': property_metrics
    }


def print_competition_evaluation(results, model_name="Model"):
    """Print formatted competition evaluation results"""
    print("=" * 80)
    print(f"{model_name} - Competition Evaluation (wMAE)")
    print("=" * 80)
    print(f"\n🏆 Overall wMAE Score: {results['wMAE']:.6f}\n")
    
    print("Property-wise Breakdown:")
    print("-" * 80)
    print(f"{'Property':<12} {'n':<8} {'Weight':<10} {'MAE':<10} {'wMAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 80)
    
    for prop, metrics in results['property_metrics'].items():
        print(f"{prop:<12} {metrics['n_samples']:<8} "
              f"{metrics['weight']:<10.4f} "
              f"{metrics['MAE']:<10.4f} "
              f"{metrics['weighted_MAE']:<10.6f} "
              f"{metrics['RMSE']:<10.4f} "
              f"{metrics['R²']:<10.4f}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test with dummy data
    print("Testing competition metrics...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_properties = 5
    
    y_true = np.random.randn(n_samples, n_properties) * 10
    y_pred = y_true + np.random.randn(n_samples, n_properties) * 2
    
    # Add some NaN values to simulate sparse labels
    mask = np.random.rand(n_samples, n_properties) > 0.3
    y_true[~mask] = np.nan
    
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Evaluate
    results = evaluate_model_competition(y_true, y_pred, property_names)
    print_competition_evaluation(results, "Test Model")

