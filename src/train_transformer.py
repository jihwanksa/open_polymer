"""
Train Transformer-based models (BERT/DistilBERT) for polymer property prediction
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
from models.transformer import TransformerModel
from competition_metrics import evaluate_model_competition, print_competition_evaluation


def main():
    print("=" * 80)
    print("TRANSFORMER MODEL TRAINING (BERT for SMILES)")
    print("=" * 80)
    
    # Check device - try CUDA first, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        device_info = f"GPU: {device_name}\nGPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    elif torch.backends.mps.is_available():
        device = 'mps'
        device_name = "Apple Silicon (MPS)"
        device_info = "Transformer layers fully support MPS acceleration"
    else:
        device = 'cpu'
        device_name = "CPU"
        device_info = "Running on CPU (slower)"
    
    print(f"\nUsing device: {device}")
    print(f"Device name: {device_name}")
    print(device_info)
    
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
    
    train_split = train_df.iloc[train_indices].reset_index(drop=True)
    val_split = train_df.iloc[val_indices].reset_index(drop=True)
    
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    # Create transformer model
    print("\n" + "=" * 80)
    print("Model: DistilBERT (lightweight, fast)")
    print("=" * 80)
    
    try:
        transformer_model = TransformerModel(
            model_name='distilbert-base-uncased',
            num_targets=len(target_cols),
            hidden_dim=256,
            dropout=0.2,
            device=device
        )
        
        # Prepare datasets
        print("\nPreparing training dataset...")
        train_smiles = train_split['SMILES'].tolist()
        train_dataset = transformer_model.prepare_data(train_smiles, y_train)
        
        print("Preparing validation dataset...")
        val_smiles = val_split['SMILES'].tolist()
        val_dataset = transformer_model.prepare_data(val_smiles, y_val)
        
        print(f"\nTrain samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Train
        print("\n" + "=" * 80)
        print("Training Transformer...")
        print("=" * 80)
        
        start_time = time.time()
        transformer_model.train(
            train_dataset, val_dataset,
            epochs=20,
            batch_size=16,  # Smaller batch for transformer
            lr=2e-5  # Lower learning rate for fine-tuning
        )
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.1f}s ({training_time/60:.1f} min)")
        
        # Save model
        model_path = os.path.join(project_root, 'models', 'transformer_model.pt')
        transformer_model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Evaluate with competition metric
        print("\n" + "=" * 80)
        print("Evaluating Transformer with competition metric...")
        print("=" * 80)
        
        y_pred = transformer_model.predict(val_dataset)
        
        # Ensure shapes match
        min_samples = min(len(y_val), len(y_pred))
        y_val_aligned = y_val[:min_samples]
        y_pred = y_pred[:min_samples]
        
        results = evaluate_model_competition(y_val_aligned, y_pred, target_cols)
        print_competition_evaluation(results, "Transformer (DistilBERT)")
        
        # Save results
        output_path = os.path.join(project_root, 'results', 'transformer_results.csv')
        pd.DataFrame([{
            'Model': 'Transformer',
            'Model_Type': 'DistilBERT',
            'wMAE': results['wMAE'],
            'training_time_s': training_time,
            **{f'{prop}_MAE': metrics['MAE'] for prop, metrics in results['property_metrics'].items()}
        }]).to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        
        # Compare with traditional models
        print("\n" + "=" * 80)
        print("COMPARISON WITH OTHER MODELS")
        print("=" * 80)
        
        trad_results = pd.read_csv(os.path.join(project_root, 'results', 'competition_metrics.csv'))
        
        print(f"""
Traditional Models:
  🥇 XGBoost:       wMAE = {trad_results[trad_results['Model'] == 'XGBoost']['wMAE'].values[0]:.6f}
  🥈 Random Forest:  wMAE = {trad_results[trad_results['Model'] == 'Random Forest']['wMAE'].values[0]:.6f}

Transformer:
  DistilBERT:      wMAE = {results['wMAE']:.6f}
  Training time:   {training_time/60:.1f} minutes
        """)
        
        if results['wMAE'] < 0.035:
            print("🎉 Transformer achieves competitive performance!")
        elif results['wMAE'] < 0.05:
            print("👍 Transformer shows good performance!")
        else:
            print("💡 Transformer may benefit from:")
            print("   - Using ChemBERTa (chemistry-specific pretrained model)")
            print("   - More training epochs")
            print("   - Hyperparameter tuning")
        
    except Exception as e:
        print(f"\n⚠️  Transformer training failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("TROUBLESHOOTING")
        print("=" * 80)
        print("""
Possible solutions:
1. Protobuf version conflict:
   pip install --upgrade protobuf==3.20.3

2. Missing transformers library:
   pip install transformers

3. Memory issues:
   - Reduce batch_size (try 8 or 4)
   - Use CPU instead of GPU
   - Close other applications

4. Model download issues:
   - Check internet connection
   - Try different model: 'bert-base-uncased'
        """)
    
    print("\n" + "=" * 80)
    print("TRANSFORMER TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

