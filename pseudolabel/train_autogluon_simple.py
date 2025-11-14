"""Train AutoGluon models with minimal setup - let AutoGluon handle feature engineering.

This script:
1. Loads 7,973 labeled training samples
2. Extracts SIMPLE features (just basic counts, no hand-crafting)
3. Trains AutoGluon - it automatically does feature engineering
4. Saves trained models for inference

AutoGluon will automatically:
- Handle missing values
- Detect feature types
- Do feature selection
- Try multiple algorithms
- Ensemble them

Usage:
    python pseudolabel/train_autogluon_simple.py \
        --time_limit 600 \
        --preset medium
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
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
    """Extract minimal SIMPLE features - let AutoGluon do feature engineering!
    
    Just extract raw counts and let AutoGluon's feature selection figure out
    what's actually important. No hand-crafted domain knowledge needed.
    """
    smiles_str = str(smiles).strip()
    
    if not smiles_str:
        return None
    
    try:
        # Just basic atomic and structural counts
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

def main(args):
    print("\n" + "=" * 80)
    print("TRAINING AUTOGLUON (Simple Features - AutoGluon Handles Engineering)")
    print("=" * 80 + "\n")
    
    # Load data
    print("📂 Loading training data...")
    train_df = pd.read_csv('data/raw/train.csv')
    print(f"   Loaded {len(train_df)} training samples\n")
    
    # Canonicalize SMILES
    if RDKIT_AVAILABLE:
        print("🧪 Canonicalizing SMILES...")
        train_df['SMILES'] = train_df['SMILES'].apply(make_smile_canonical)
        train_df.dropna(subset=['SMILES'], inplace=True)
        print(f"   ✅ After canonicalization: {len(train_df)} samples\n")
    
    # Extract SIMPLE features (minimal effort)
    print("🔍 Extracting simple features from all SMILES...")
    feature_rows = []
    for idx, smiles in tqdm(enumerate(train_df['SMILES']), total=len(train_df), desc="Extracting"):
        features = extract_simple_features(smiles)
        if features:
            feature_rows.append(features)
        else:
            print(f"   ⚠️  Failed to extract features for SMILES at row {idx}")
    
    X_features = pd.DataFrame(feature_rows)
    print(f"✅ Extracted features: {X_features.shape}")
    print(f"   Features: {list(X_features.columns)}\n")
    
    # Align with training data
    train_df_filtered = train_df.iloc[:len(X_features)].reset_index(drop=True)
    
    # Train AutoGluon models
    print("🤖 Training AutoGluon models for each property...")
    print(f"   Time limit per model: {args.time_limit} seconds")
    print(f"   Preset: {args.preset}")
    print(f"   (Let AutoGluon do feature engineering automatically!)\n")
    
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("❌ AutoGluon not installed!")
        print("   Install with: pip install autogluon")
        return
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    output_dir = Path('models/autogluon_simple')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for target in target_cols:
        print(f"\n   Training AutoGluon for {target}...")
        
        # Get valid targets
        valid_mask = train_df_filtered[target].notna().values
        X_train = X_features[valid_mask]
        y_train = train_df_filtered[target].values[valid_mask]
        
        if len(y_train) == 0:
            print(f"      ⚠️  No valid targets for {target}, skipping")
            continue
        
        print(f"      Using {len(y_train)} valid samples")
        
        # Create training dataframe
        train_data = X_train.copy()
        train_data[target] = y_train
        
        # Train AutoGluon - let it handle everything!
        model_path = output_dir / target
        try:
            predictor = TabularPredictor(
                label=target,
                path=str(model_path),
                problem_type='regression',
                eval_metric='mean_absolute_error'
            )
            
            predictor.fit(
                train_data,
                time_limit=args.time_limit,
                presets=args.preset,
                verbosity=1
            )
            
            print(f"      ✅ {target} model trained and saved")
        
        except Exception as e:
            print(f"      ❌ Failed to train {target}: {e}")
    
    print("\n" + "=" * 80)
    print("✅ AUTOGLUON TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved in: {output_dir}")
    print("\nReady to generate pseudo-labels with:")
    print("  python pseudolabel/generate_with_autogluon_simple.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AutoGluon models (simple features, let AutoGluon do feature engineering).")
    parser.add_argument("--time_limit", type=int, default=600,
                        help="Time limit in seconds per model (default: 600s = 10 min)")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=['fast', 'medium', 'high', 'best'],
                        help="AutoGluon preset: fast (quick), medium (balanced), high (slow), best (very slow)")
    args = parser.parse_args()
    main(args)

