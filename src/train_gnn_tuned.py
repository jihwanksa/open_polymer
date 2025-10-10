"""
Train GNN with hyperparameter tuning for better performance
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import time

from data_preprocessing import MolecularDataProcessor
from models.gnn import GNNModel
from competition_metrics import evaluate_model_competition, print_competition_evaluation


def train_gnn_with_config(train_df, val_df, target_cols, config, run_name="GNN"):
    """Train GNN with specific configuration"""
    print(f"\n{'='*80}")
    print(f"Training {run_name}")
    print(f"Configuration: {config}")
    print(f"{'='*80}")
    
    # Prepare targets
    y_train = train_df[target_cols].values
    y_val = val_df[target_cols].values
    
    # Create GNN model - Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    gnn_model = GNNModel(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_targets=len(target_cols),
        gnn_type=config['gnn_type'],
        dropout=config['dropout'],
        device=device
    )
    
    # Prepare graph data
    print("\nPreparing training graphs...")
    train_smiles = train_df['SMILES'].tolist()
    train_targets = np.nan_to_num(y_train, nan=0.0)
    train_graphs, train_valid_idx = gnn_model.prepare_data(train_smiles, train_targets)
    
    print("Preparing validation graphs...")
    val_smiles = val_df['SMILES'].tolist()
    val_targets = np.nan_to_num(y_val, nan=0.0)
    val_graphs, val_valid_idx = gnn_model.prepare_data(val_smiles, val_targets)
    
    print(f"\nTrain graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    
    if len(train_graphs) < 10 or len(val_graphs) < 5:
        print("⚠️  Insufficient graph data")
        return None, float('inf')
    
    # Train
    start_time = time.time()
    gnn_model.train(
        train_graphs, val_graphs,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr']
    )
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = gnn_model.predict(val_graphs)
    val_targets_aligned = y_val[val_valid_idx]
    
    # Ensure shapes match
    min_samples = min(len(val_targets_aligned), len(y_pred))
    val_targets_aligned = val_targets_aligned[:min_samples]
    y_pred = y_pred[:min_samples]
    
    results = evaluate_model_competition(val_targets_aligned, y_pred, target_cols)
    
    print(f"\n{run_name} Results:")
    print(f"  wMAE: {results['wMAE']:.6f}")
    print(f"  Training time: {training_time:.1f}s")
    
    return gnn_model, results['wMAE'], results, training_time


def main():
    print("="*80)
    print("GNN HYPERPARAMETER TUNING")
    print("="*80)
    
    # Load data
    project_root = os.path.dirname(os.path.dirname(__file__))
    processor = MolecularDataProcessor()
    
    train_df, test_df, target_cols = processor.load_and_process_data(
        os.path.join(project_root, 'data/raw/train.csv'),
        os.path.join(project_root, 'data/raw/test.csv')
    )
    
    # Split data (use same split for fair comparison)
    train_indices, val_indices = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    
    train_split = train_df.iloc[train_indices].reset_index(drop=True)
    val_split = train_df.iloc[val_indices].reset_index(drop=True)
    
    # Hyperparameter configurations to try (optimized for reasonable training time)
    configs = [
        {
            'name': 'GNN_Baseline',
            'hidden_dim': 64,
            'num_layers': 2,
            'gnn_type': 'gcn',
            'dropout': 0.1,
            'epochs': 50,
            'batch_size': 64,  # Larger batch for GPU
            'lr': 0.001
        },
        {
            'name': 'GNN_Deeper',
            'hidden_dim': 128,
            'num_layers': 4,
            'gnn_type': 'gcn',
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 64,
            'lr': 0.001
        },
        {
            'name': 'GNN_Wider',
            'hidden_dim': 256,
            'num_layers': 3,
            'gnn_type': 'gcn',
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,  # Smaller batch for larger model
            'lr': 0.0005  # Lower learning rate
        },
        {
            'name': 'GNN_LongTrain',
            'hidden_dim': 128,
            'num_layers': 3,
            'gnn_type': 'gcn',
            'dropout': 0.2,
            'epochs': 100,  # More epochs for convergence
            'batch_size': 64,
            'lr': 0.001
        }
    ]
    
    # Store results
    all_results = []
    best_wMAE = float('inf')
    best_model = None
    best_config = None
    
    # Try each configuration
    for config in configs:
        print(f"\n\n{'#'*80}")
        print(f"# Testing: {config['name']}")
        print(f"{'#'*80}")
        
        try:
            model, wMAE, results, train_time = train_gnn_with_config(
                train_split, val_split, target_cols, config, config['name']
            )
            
            if wMAE < best_wMAE:
                best_wMAE = wMAE
                best_model = model
                best_config = config
                print(f"\n🎯 New best model! wMAE: {wMAE:.6f}")
            
            all_results.append({
                'Model': config['name'],
                'wMAE': wMAE,
                'hidden_dim': config['hidden_dim'],
                'num_layers': config['num_layers'],
                'gnn_type': config['gnn_type'],
                'dropout': config['dropout'],
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'lr': config['lr'],
                'training_time_s': train_time
            })
            
        except Exception as e:
            print(f"\n⚠️  Configuration failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*80)
    
    if all_results:
        results_df = pd.DataFrame(all_results).sort_values('wMAE')
        
        print("\n🏆 Ranking (by wMAE):\n")
        print(f"{'Rank':<6} {'Model':<20} {'wMAE':<12} {'Config':<40}")
        print("-"*80)
        
        for rank, (_, row) in enumerate(results_df.iterrows(), 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            config_str = f"h={row['hidden_dim']}, l={row['num_layers']}, {row['gnn_type']}, e={row['epochs']}"
            print(f"{medal:<6} {row['Model']:<20} {row['wMAE']:<12.6f} {config_str:<40}")
        
        # Save best model
        if best_model is not None:
            model_path = os.path.join(project_root, 'models', 'gnn_best_tuned.pt')
            best_model.save(model_path)
            print(f"\n✅ Best model saved to: {model_path}")
            print(f"   Configuration: {best_config['name']}")
            print(f"   wMAE: {best_wMAE:.6f}")
            
            # Detailed evaluation of best model
            print("\n" + "="*80)
            print(f"BEST MODEL: {best_config['name']}")
            print("="*80)
            
            # Re-evaluate to get full metrics
            y_pred = best_model.predict(best_model.prepare_data(
                val_split['SMILES'].tolist(),
                np.nan_to_num(val_split[target_cols].values, nan=0.0)
            )[0])
            
            val_targets = val_split[target_cols].values[:len(y_pred)]
            best_results = evaluate_model_competition(val_targets, y_pred, target_cols)
            print_competition_evaluation(best_results, best_config['name'])
        
        # Save all results
        output_path = os.path.join(project_root, 'results', 'gnn_tuning_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ All tuning results saved to: {output_path}")
        
        # Comparison with traditional models
        print("\n" + "="*80)
        print("COMPARISON WITH TRADITIONAL MODELS")
        print("="*80)
        print(f"""
Traditional Models:
  🥇 XGBoost:       wMAE = 0.030429
  🥈 Random Forest:  wMAE = 0.031638

Best GNN:
  {best_config['name']}:  wMAE = {best_wMAE:.6f}
  
Gap: {((best_wMAE / 0.030429) - 1) * 100:.1f}% worse than XGBoost
        """)
        
        if best_wMAE < 0.05:
            print("🎉 GNN is competitive with traditional models!")
        elif best_wMAE < 0.10:
            print("👍 GNN shows promise, further tuning could close the gap")
        else:
            print("💡 GNN needs more work - consider:")
            print("   - Pre-training on molecular datasets")
            print("   - Adding edge features")
            print("   - Using graph attention mechanisms")
            print("   - Increasing dataset size")
    
    print("\n" + "="*80)
    print("GNN TUNING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

