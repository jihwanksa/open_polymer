"""
Evaluate trained models using official competition metric (wMAE)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing import MolecularDataProcessor
from models.traditional import TraditionalMLModel
from competition_metrics import evaluate_model_competition, print_competition_evaluation


def evaluate_traditional_models():
    """Evaluate XGBoost and Random Forest with competition metric"""
    print("=" * 80)
    print("EVALUATING MODELS WITH COMPETITION METRIC (wMAE)")
    print("=" * 80)
    
    # Load data
    project_root = os.path.dirname(os.path.dirname(__file__))
    processor = MolecularDataProcessor()
    
    train_df, test_df, target_cols = processor.load_and_process_data(
        os.path.join(project_root, 'data/raw/train.csv'),
        os.path.join(project_root, 'data/raw/test.csv')
    )
    
    # Prepare features
    print("\nPreparing features...")
    train_desc = processor.create_descriptor_features(train_df)
    train_fp = processor.create_fingerprint_features(train_df, n_bits=1024)
    
    # Combine features
    train_features = pd.concat([train_desc, train_fp], axis=1)
    
    # Align indices
    common_indices = train_df.index.intersection(train_features.index)
    train_df_filtered = train_df.loc[common_indices]
    train_features = train_features.loc[common_indices]
    
    # Prepare targets
    y = train_df_filtered[target_cols].values
    X = train_features.values
    
    # Check for NaN in features
    nan_mask_X = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[nan_mask_X]
    y = y[nan_mask_X]
    
    print(f"Total samples: {len(X)}")
    
    # Split data (same split as training)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Validation samples: {len(X_val)}")
    
    # Load and evaluate models
    models = {
        'XGBoost': 'xgboost_model.pkl',
        'Random Forest': 'random_forest_model.pkl'
    }
    
    results_summary = []
    
    for model_name, model_file in models.items():
        model_path = os.path.join(project_root, 'models', model_file)
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found: {model_path}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Loading {model_name}...")
        
        # Load model
        model = TraditionalMLModel()
        model.load(model_path)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Evaluate with competition metric
        results = evaluate_model_competition(y_val, y_pred, target_cols)
        print_competition_evaluation(results, model_name)
        
        # Store summary
        results_summary.append({
            'Model': model_name,
            'wMAE': results['wMAE'],
            **{f'{prop}_MAE': metrics['MAE'] for prop, metrics in results['property_metrics'].items()},
            **{f'{prop}_wMAE': metrics['weighted_MAE'] for prop, metrics in results['property_metrics'].items()}
        })
    
    # Create comparison table
    if results_summary:
        print("\n" + "=" * 80)
        print("SUMMARY - Competition Metric Comparison")
        print("=" * 80)
        
        summary_df = pd.DataFrame(results_summary)
        
        print(f"\n{'Model':<20} {'Overall wMAE':<15}")
        print("-" * 35)
        for _, row in summary_df.iterrows():
            print(f"{row['Model']:<20} {row['wMAE']:<15.6f}")
        
        # Save to file
        output_path = os.path.join(project_root, 'results', 'competition_metrics.csv')
        summary_df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
    
    return results_summary


if __name__ == "__main__":
    evaluate_traditional_models()

