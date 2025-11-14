"""Generate pseudo-labels using trained AutoGluon models (simple approach).

This script:
1. Extracts simple features from 50K unlabeled SMILES
2. Uses trained AutoGluon models to generate predictions
3. Applies Tg transformation
4. Saves final pseudo-labels

⚠️ IMPORTANT: Runs on CPU only to avoid Apple Silicon (MPS) hanging issues.

Usage:
    python pseudolabel/generate_with_autogluon_simple.py \
        --input_data data/PI1M_50000_v2.1.csv \
        --models_dir models/autogluon_simple \
        --output_path pseudolabel/pi1m_pseudolabels_autogluon.csv
"""

# FORCE CPU MODE BEFORE IMPORTING AUTOGLUON (fixes Apple Silicon MPS hanging)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
os.environ['OMP_NUM_THREADS'] = '1'      # Limit threads
os.environ['MPS_ENABLED'] = '0'          # Disable MPS on Apple Silicon

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

# Ensure RDKit is available
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available - SMILES canonicalization will be skipped")

def make_smile_canonical(smile):
    if not RDKIT_AVAILABLE or Chem is None:
        return smile
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

def extract_simple_features(smiles: str) -> dict:
    """Extract simple features - let AutoGluon do the heavy lifting!"""
    smiles_str = str(smiles).strip()
    
    if not smiles_str:
        return None
    
    try:
        features = {
            'smiles_length': len(smiles_str),
            'carbon_count': smiles_str.count('C'),
            'nitrogen_count': smiles_str.count('N'),
            'oxygen_count': smiles_str.count('O'),
            'sulfur_count': smiles_str.count('S'),
            'fluorine_count': smiles_str.count('F'),
            'chlorine_count': smiles_str.count('Cl'),
            'bromine_count': smiles_str.count('Br'),
            'aromatic_atoms': smiles_str.count('c'),
            'double_bonds': smiles_str.count('='),
            'triple_bonds': smiles_str.count('#'),
            'branches': smiles_str.count('('),
            'rings': smiles_str.count('1') + smiles_str.count('2'),
        }
        return features
    except Exception as e:
        print(f"Warning: Failed to extract features from SMILES '{smiles}': {e}", file=sys.stderr)
        return None

def apply_tg_transformation(tg_value: float) -> float:
    return (9/5) * tg_value + 45

def main(args):
    print("\n" + "=" * 80)
    print("PSEUDO-LABEL GENERATION WITH AUTOGLUON (Simple Features)")
    print("=" * 80 + "\n")
    
    # Check models directory
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("   First train models with: python pseudolabel/train_autogluon_simple.py")
        return
    
    # Load input data
    print(f"📂 Loading SMILES from {args.input_data}...")
    if not Path(args.input_data).exists():
        print(f"❌ Input file not found: {args.input_data}")
        return
    
    df_smiles = pd.read_csv(args.input_data)
    smiles_list = df_smiles['SMILES'].tolist()
    print(f"   Loaded {len(smiles_list)} SMILES")
    
    # Canonicalize SMILES
    if RDKIT_AVAILABLE:
        print("   Canonicalizing SMILES...")
        df_smiles['SMILES'] = df_smiles['SMILES'].apply(make_smile_canonical)
        df_smiles.dropna(subset=['SMILES'], inplace=True)
        smiles_list = df_smiles['SMILES'].tolist()
        print(f"   ✅ After canonicalization: {len(smiles_list)} samples\n")
    
    # Extract simple features
    print("🔍 Extracting simple features from all 50K SMILES...", flush=True)
    sys.stdout.flush()
    feature_rows = []
    for idx, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Extracting"):
        features = extract_simple_features(smiles)
        if features:
            feature_rows.append(features)
        else:
            # Add zero features for failed extraction
            feature_rows.append({f: 0.0 for f in ['smiles_length', 'carbon_count', 'nitrogen_count', 
                                                     'oxygen_count', 'sulfur_count', 'fluorine_count',
                                                     'chlorine_count', 'bromine_count', 'aromatic_atoms',
                                                     'double_bonds', 'triple_bonds', 'branches', 'rings']})
    
    print(f"\n   Converting to DataFrame...", flush=True)
    sys.stdout.flush()
    X_features = pd.DataFrame(feature_rows)
    print(f"✅ Extracted features: {X_features.shape}\n", flush=True)
    sys.stdout.flush()
    
    # Load AutoGluon models and generate predictions
    print("🤖 Loading AutoGluon models and generating predictions...", flush=True)
    print("   (This may take 1-2 minutes on first load)\n", flush=True)
    sys.stdout.flush()
    
    try:
        print("   Importing AutoGluon...", flush=True)
        sys.stdout.flush()
        from autogluon.tabular import TabularPredictor
        print("   ✅ AutoGluon imported\n", flush=True)
        sys.stdout.flush()
    except ImportError as e:
        print("❌ AutoGluon not installed!")
        print(f"   Error: {e}")
        print("   Install with: pip install autogluon")
        return
    except Exception as e:
        print(f"❌ Failed to import AutoGluon: {e}")
        return
    
    target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    predictions_df = pd.DataFrame({'SMILES': smiles_list})
    
    for target in target_names:
        model_path = models_dir / target
        
        if not model_path.exists():
            print(f"   ⚠️  Model not found for {target} at {model_path}", flush=True)
            print(f"      Filling with NaN", flush=True)
            predictions_df[target] = np.nan
            continue
        
        print(f"   {target}:", flush=True)
        sys.stdout.flush()
        try:
            print(f"      Loading model...", flush=True)
            sys.stdout.flush()
            predictor = TabularPredictor.load(str(model_path))
            print(f"      Generating predictions (batch of 50K)...", flush=True)
            sys.stdout.flush()
            
            # Predict in smaller batches to avoid hanging
            batch_size = 5000
            preds_list = []
            for i in tqdm(range(0, len(X_features), batch_size), desc=f"   {target} batches", leave=False):
                batch = X_features.iloc[i:i+batch_size]
                batch_preds = predictor.predict(batch)
                preds_list.append(batch_preds.values if hasattr(batch_preds, 'values') else batch_preds)
            
            preds = np.concatenate(preds_list) if preds_list else np.zeros(len(X_features))
            predictions_df[target] = preds
            
            print(f"      Mean: {np.mean(preds):.4f}, Std: {np.std(preds):.4f}", flush=True)
            print(f"      ✅ Predictions generated", flush=True)
            sys.stdout.flush()
        
        except Exception as e:
            print(f"      ❌ Failed to generate predictions: {e}", flush=True)
            predictions_df[target] = np.nan
    
    # Apply Tg transformation
    if args.apply_tg_transform and 'Tg' in predictions_df.columns:
        print(f"\n🔧 Applying Tg transformation: (9/5) × Tg + 45...")
        predictions_df['Tg'] = predictions_df['Tg'].apply(apply_tg_transformation)
    
    # Save
    print(f"\n💾 Saving pseudo-labels to {args.output_path}...")
    predictions_df.to_csv(args.output_path, index=False)
    print(f"✅ Saved {len(predictions_df)} pseudo-labeled samples")
    
    print("\n" + "=" * 80)
    print("✅ AUTOGLUON PSEUDO-LABEL GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Output: {args.output_path}")
    print(f"   Samples: {len(predictions_df)}")
    print(f"   Properties: {', '.join(target_names)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-labels using trained AutoGluon models.")
    parser.add_argument("--input_data", type=str, default="data/PI1M_50000_v2.1.csv",
                        help="Path to input CSV with SMILES column")
    parser.add_argument("--models_dir", type=str, default="models/autogluon_simple",
                        help="Path to directory with trained AutoGluon models")
    parser.add_argument("--output_path", type=str, default="pseudolabel/pi1m_pseudolabels_autogluon.csv",
                        help="Path to save pseudo-labels")
    parser.add_argument("--apply_tg_transform", action="store_true", default=True,
                        help="Apply Tg transformation")
    args = parser.parse_args()
    main(args)

