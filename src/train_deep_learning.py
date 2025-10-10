"""
Train and evaluate deep learning models (GNN and Transformer)
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
from models.transformer import TransformerModel
from competition_metrics import evaluate_model_competition, print_competition_evaluation


def train_gnn_model(train_df, target_cols):
    """Train Graph Neural Network"""
    print("\n" + "=" * 80)
    print("TRAINING GRAPH NEURAL NETWORK (GNN)")
    print("=" * 80)
    
    # Prepare targets
    y = train_df[target_cols].values
    
    # Split data
    train_indices, val_indices = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    
    # Create GNN model (use CPU for compatibility)
    device = 'cpu'  # Use CPU to avoid CUDA compatibility issues
    print(f"Using device: {device}")
    
    gnn_model = GNNModel(
        hidden_dim=128,
        num_layers=3,
        num_targets=len(target_cols),
        gnn_type='gcn',
        device=device
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
    
    if len(train_graphs) < 10 or len(val_graphs) < 5:
        print("⚠️  Insufficient graph data for training")
        return None, None
    
    # Train
    try:
        gnn_model.train(
            train_graphs, val_graphs,
            epochs=50,  # Reduced for faster training
            batch_size=32,
            lr=0.001
        )
        
        # Save model
        project_root = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(project_root, 'models', 'gnn_model.pt')
        gnn_model.save(model_path)
        
        # Evaluate with competition metric
        print("\n" + "=" * 80)
        print("Evaluating GNN with competition metric...")
        
        y_pred = gnn_model.predict(val_graphs)
        results = evaluate_model_competition(val_targets, y_pred, target_cols)
        print_competition_evaluation(results, "GNN (GCN)")
        
        return gnn_model, results
        
    except Exception as e:
        print(f"\n⚠️  GNN training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_transformer_model(train_df, target_cols):
    """Train Transformer model"""
    print("\n" + "=" * 80)
    print("TRAINING TRANSFORMER MODEL (SMILES-based)")
    print("=" * 80)
    
    # Prepare targets
    y = train_df[target_cols].values
    
    # Split data
    train_indices, val_indices = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    
    # Create transformer model (use CPU for compatibility)
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Use DistilBERT for faster training
    print("Using DistilBERT (lightweight transformer)...")
    transformer_model = TransformerModel(
        model_name='distilbert-base-uncased',
        num_targets=len(target_cols),
        hidden_dim=256,
        device=device
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
    try:
        transformer_model.train(
            train_dataset, val_dataset,
            epochs=20,  # Reduced for faster training
            batch_size=16,
            lr=2e-5
        )
        
        # Save model
        project_root = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(project_root, 'models', 'transformer_model.pt')
        transformer_model.save(model_path)
        
        # Evaluate with competition metric
        print("\n" + "=" * 80)
        print("Evaluating Transformer with competition metric...")
        
        y_pred = transformer_model.predict(val_dataset)
        results = evaluate_model_competition(val_targets, y_pred, target_cols)
        print_competition_evaluation(results, "Transformer")
        
        return transformer_model, results
        
    except Exception as e:
        print(f"\n⚠️  Transformer training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main training function"""
    print("=" * 80)
    print("DEEP LEARNING MODELS TRAINING")
    print("=" * 80)
    
    # Load data
    project_root = os.path.dirname(os.path.dirname(__file__))
    processor = MolecularDataProcessor()
    
    train_df, test_df, target_cols = processor.load_and_process_data(
        os.path.join(project_root, 'data/raw/train.csv'),
        os.path.join(project_root, 'data/raw/test.csv')
    )
    
    results_summary = []
    
    # Train GNN
    print("\n" + "=" * 80)
    print("1. GRAPH NEURAL NETWORK")
    print("=" * 80)
    gnn_model, gnn_results = train_gnn_model(train_df, target_cols)
    
    if gnn_results:
        results_summary.append({
            'Model': 'GNN (GCN)',
            'wMAE': gnn_results['wMAE']
        })
    
    # Train Transformer
    print("\n" + "=" * 80)
    print("2. TRANSFORMER MODEL")
    print("=" * 80)
    transformer_model, transformer_results = train_transformer_model(train_df, target_cols)
    
    if transformer_results:
        results_summary.append({
            'Model': 'Transformer',
            'wMAE': transformer_results['wMAE']
        })
    
    # Summary
    if results_summary:
        print("\n" + "=" * 80)
        print("DEEP LEARNING MODELS - SUMMARY")
        print("=" * 80)
        
        for result in results_summary:
            print(f"{result['Model']:<20} wMAE: {result['wMAE']:.6f}")
        
        # Save to CSV
        output_path = os.path.join(project_root, 'results', 'deep_learning_results.csv')
        pd.DataFrame(results_summary).to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

