"""
Main script to train and compare all models for polymer property prediction
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import MolecularDataProcessor
from models.traditional import TraditionalMLModel
from models.gnn import GNNModel
from models.transformer import TransformerModel


def load_data():
    """Load and prepare data"""
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    processor = MolecularDataProcessor()
    train_df, test_df, target_cols = processor.load_and_process_data(
        os.path.join(project_root, 'data/raw/train.csv'),
        os.path.join(project_root, 'data/raw/test.csv')
    )
    
    return processor, train_df, test_df, target_cols


def prepare_features_for_traditional_ml(processor, train_df, test_df, feature_type='descriptors'):
    """Prepare features for traditional ML models"""
    print(f"\nPreparing {feature_type} features for traditional ML...")
    
    if feature_type == 'descriptors':
        train_features = processor.create_descriptor_features(train_df)
        test_features = processor.create_descriptor_features(test_df)
    elif feature_type == 'fingerprints':
        train_features = processor.create_fingerprint_features(train_df)
        test_features = processor.create_fingerprint_features(test_df)
    elif feature_type == 'combined':
        train_desc = processor.create_descriptor_features(train_df)
        test_desc = processor.create_descriptor_features(test_df)
        train_fp = processor.create_fingerprint_features(train_df, n_bits=1024)
        test_fp = processor.create_fingerprint_features(test_df, n_bits=1024)
        
        # Combine features
        train_features = pd.concat([train_desc, train_fp], axis=1)
        test_features = pd.concat([test_desc, test_fp], axis=1)
    
    return train_features, test_features


def train_traditional_models(processor, train_df, test_df, target_cols, feature_type='combined'):
    """Train traditional ML models"""
    print("\n" + "=" * 80)
    print("TRAINING TRADITIONAL ML MODELS")
    print("=" * 80)
    
    # Prepare features
    train_features, test_features = prepare_features_for_traditional_ml(
        processor, train_df, test_df, feature_type
    )
    
    # Align indices
    common_indices = train_df.index.intersection(train_features.index)
    train_df_filtered = train_df.loc[common_indices]
    train_features = train_features.loc[common_indices]
    
    print(f"\nAligned {len(common_indices)} samples with valid features")
    
    # Prepare targets
    y = train_df_filtered[target_cols].values
    X = train_features.values
    
    # Check for NaN/inf values in FEATURES only (targets are sparse, handled per-target in models)
    nan_mask_X = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    
    if not nan_mask_X.all():
        print(f"Removing {(~nan_mask_X).sum()} samples with NaN/inf features")
        X = X[nan_mask_X]
        y = y[nan_mask_X]
    
    print(f"Final training set: {len(X)} samples with {X.shape[1]} features")
    print(f"Target availability per property:")
    for i, col in enumerate(target_cols):
        n_avail = (~np.isnan(y[:, i])).sum()
        print(f"  {col}: {n_avail} samples ({n_avail/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = {}
    
    # XGBoost
    print("\n" + "=" * 80)
    print("Training XGBoost...")
    xgb_model = TraditionalMLModel(model_type='xgboost', n_targets=len(target_cols))
    xgb_metrics = xgb_model.train(X_train, y_train, X_val, y_val, target_cols)
    xgb_model.save('models/xgboost_model.pkl')
    results['XGBoost'] = xgb_metrics
    
    # Random Forest
    print("\n" + "=" * 80)
    print("Training Random Forest...")
    rf_model = TraditionalMLModel(model_type='random_forest', n_targets=len(target_cols))
    rf_metrics = rf_model.train(X_train, y_train, X_val, y_val, target_cols)
    rf_model.save('models/random_forest_model.pkl')
    results['RandomForest'] = rf_metrics
    
    return results, {'xgboost': xgb_model, 'random_forest': rf_model}, test_features


def train_gnn_model(train_df, test_df, target_cols):
    """Train GNN model"""
    print("\n" + "=" * 80)
    print("TRAINING GRAPH NEURAL NETWORK")
    print("=" * 80)
    
    # Prepare targets
    y = train_df[target_cols].values
    
    # Split data
    train_indices, val_indices = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    
    # Create GNN model
    gnn_model = GNNModel(
        hidden_dim=128, 
        num_layers=3, 
        num_targets=len(target_cols),
        gnn_type='gcn'
    )
    
    # Prepare graph data
    print("\nPreparing training graphs...")
    train_smiles = train_df.iloc[train_indices]['SMILES'].tolist()
    train_targets = y[train_indices]
    train_graphs, train_valid_idx = gnn_model.prepare_data(train_smiles, train_targets)
    
    print("Preparing validation graphs...")
    val_smiles = train_df.iloc[val_indices]['SMILES'].tolist()
    val_targets = y[val_indices]
    val_graphs, val_valid_idx = gnn_model.prepare_data(val_smiles, val_targets)
    
    print(f"\nTrain graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    
    # Train
    gnn_model.train(train_graphs, val_graphs, epochs=100, batch_size=32, lr=0.001)
    
    # Evaluate
    val_loss, val_metrics = gnn_model.evaluate(
        gnn_model.prepare_data(val_smiles, val_targets)[0]
    )
    
    gnn_model.save('models/gnn_model.pt')
    
    # Format metrics
    results = {}
    for i, target in enumerate(target_cols):
        if f'target_{i}' in val_metrics:
            results[target] = val_metrics[f'target_{i}']
    
    return {'GNN': results}, gnn_model


def train_transformer_model(train_df, test_df, target_cols):
    """Train transformer model"""
    print("\n" + "=" * 80)
    print("TRAINING TRANSFORMER MODEL")
    print("=" * 80)
    
    # Prepare targets
    y = train_df[target_cols].values
    
    # Split data
    train_indices, val_indices = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    
    # Create transformer model
    transformer_model = TransformerModel(
        model_name='seyonec/ChemBERTa-zinc-base-v1',
        num_targets=len(target_cols),
        hidden_dim=256
    )
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_smiles = train_df.iloc[train_indices]['SMILES'].tolist()
    train_targets = y[train_indices]
    train_dataset = transformer_model.prepare_data(train_smiles, train_targets)
    
    val_smiles = train_df.iloc[val_indices]['SMILES'].tolist()
    val_targets = y[val_indices]
    val_dataset = transformer_model.prepare_data(val_smiles, val_targets)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train
    transformer_model.train(train_dataset, val_dataset, epochs=30, batch_size=16, lr=2e-5)
    
    # Evaluate
    val_loss, val_metrics = transformer_model.evaluate(
        transformer_model.prepare_data(val_smiles, val_targets)
    )
    
    transformer_model.save('models/transformer_model.pt')
    
    # Format metrics
    results = {}
    for i, target in enumerate(target_cols):
        if f'target_{i}' in val_metrics:
            results[target] = val_metrics[f'target_{i}']
    
    return {'Transformer': results}, transformer_model


def compare_results(all_results, target_cols):
    """Compare results from all models"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Create comparison dataframe
    comparison_data = []
    
    for model_name, results in all_results.items():
        for target in target_cols:
            if target in results:
                metrics = results[target]
                comparison_data.append({
                    'Model': model_name,
                    'Target': target,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'R²': metrics['r2']
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print summary
    print("\nDetailed Results:")
    print(comparison_df.to_string(index=False))
    
    # Save to CSV
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\nComparison saved to results/model_comparison.csv")
    
    # Create visualizations
    create_comparison_plots(comparison_df, target_cols)
    
    return comparison_df


def create_comparison_plots(comparison_df, target_cols):
    """Create comparison plots"""
    print("\nCreating comparison plots...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot RMSE comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['RMSE', 'MAE', 'R²']
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        pivot_data = comparison_df.pivot(index='Target', columns='Model', values=metric)
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric} Comparison Across Models', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Property', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Plots saved to results/model_comparison.png")
    plt.close()


def generate_predictions(models, test_df, target_cols):
    """Generate predictions for test set"""
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    # This will be implemented based on which models are available
    print("\nNote: Test set has only 3 samples for demonstration")
    print("In a real competition, you would generate predictions here")
    

def main():
    """Main execution"""
    # Create directories
    project_root = os.path.dirname(os.path.dirname(__file__))
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)
    
    # Change to project root for relative paths
    os.chdir(project_root)
    
    # Load data
    processor, train_df, test_df, target_cols = load_data()
    
    # Store all results
    all_results = {}
    
    # Train traditional ML models
    trad_results, trad_models, test_features = train_traditional_models(
        processor, train_df, test_df, target_cols, feature_type='combined'
    )
    all_results.update(trad_results)
    
    # Train GNN
    try:
        gnn_results, gnn_model = train_gnn_model(train_df, test_df, target_cols)
        all_results.update(gnn_results)
    except Exception as e:
        print(f"\nWarning: GNN training failed: {e}")
        print("Continuing with other models...")
    
    # Train Transformer
    try:
        transformer_results, transformer_model = train_transformer_model(train_df, test_df, target_cols)
        all_results.update(transformer_results)
    except Exception as e:
        print(f"\nWarning: Transformer training failed: {e}")
        print("Continuing with other models...")
    
    # Compare results
    comparison_df = compare_results(all_results, target_cols)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - models/: Trained model files")
    print("  - results/: Comparison metrics and plots")
    

if __name__ == "__main__":
    main()

