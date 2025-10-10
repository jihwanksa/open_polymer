"""
Train only GNN model (simplified version)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from data_preprocessing import MolecularDataProcessor
from models.gnn import GNNModel
from competition_metrics import evaluate_model_competition, print_competition_evaluation


def main():
    print("=" * 80)
    print("TRAINING GRAPH NEURAL NETWORK (GNN)")
    print("=" * 80)
    
    # Load data
    project_root = os.path.dirname(os.path.dirname(__file__))
    processor = MolecularDataProcessor()
    
    train_df, test_df, target_cols = processor.load_and_process_data(
        os.path.join(project_root, 'data/raw/train.csv'),
        os.path.join(project_root, 'data/raw/test.csv')
    )
    
    # Prepare targets
    y = train_df[target_cols].values
    
    # Split data
    train_indices, val_indices = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    
    # Create GNN model (use CPU for compatibility)
    device = 'cpu'
    print(f"Using device: {device}")
    
    gnn_model = GNNModel(
        hidden_dim=64,  # Reduced for faster training
        num_layers=2,
        num_targets=len(target_cols),
        gnn_type='gcn',
        device=device
    )
    
    # Prepare graph data
    print("\nPreparing training graphs...")
    train_smiles = train_df.iloc[train_indices]['SMILES'].tolist()
    train_targets = y[train_indices]
    
    # Convert NaN to 0 for now (will be masked during training)
    train_targets_clean = np.nan_to_num(train_targets, nan=0.0)
    train_graphs, train_valid_idx = gnn_model.prepare_data(train_smiles, train_targets_clean)
    
    print("Preparing validation graphs...")
    val_smiles = train_df.iloc[val_indices]['SMILES'].tolist()
    val_targets = y[val_indices]
    val_targets_clean = np.nan_to_num(val_targets, nan=0.0)
    val_graphs, val_valid_idx = gnn_model.prepare_data(val_smiles, val_targets_clean)
    
    print(f"\nTrain graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    
    if len(train_graphs) < 10 or len(val_graphs) < 5:
        print("⚠️  Insufficient graph data for training")
        return
    
    # Train
    print("\nTraining GNN...")
    gnn_model.train(
        train_graphs, val_graphs,
        epochs=30,  # Reduced for faster training
        batch_size=32,
        lr=0.001
    )
    
    # Save model
    model_path = os.path.join(project_root, 'models', 'gnn_model.pt')
    gnn_model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Evaluate with competition metric
    print("\n" + "=" * 80)
    print("Evaluating GNN with competition metric...")
    
    y_pred = gnn_model.predict(val_graphs)
    
    # Align targets with predictions (in case some graphs failed to convert)
    val_targets_aligned = val_targets[val_valid_idx]
    
    # Ensure shapes match
    min_samples = min(len(val_targets_aligned), len(y_pred))
    val_targets_aligned = val_targets_aligned[:min_samples]
    y_pred = y_pred[:min_samples]
    
    results = evaluate_model_competition(val_targets_aligned, y_pred, target_cols)
    print_competition_evaluation(results, "GNN (GCN)")
    
    # Save results
    output_path = os.path.join(project_root, 'results', 'gnn_results.csv')
    pd.DataFrame([{
        'Model': 'GNN',
        'wMAE': results['wMAE'],
        **{f'{prop}_MAE': metrics['MAE'] for prop, metrics in results['property_metrics'].items()}
    }]).to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("GNN TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

