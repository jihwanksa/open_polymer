"""Train AutoGluon tabular models for each polymer property.

This script:
1. Loads 7,973 labeled training samples
2. Extracts 21 chemistry-based features from SMILES
3. Trains separate AutoGluon models for each property
4. Automatically tuned and ensembled by AutoGluon
5. Saves trained models for inference

AutoGluon will try multiple models (RF, XGBoost, LightGBM, NN) and pick best!

⚠️  IMPORTANT: Runs on CPU only to avoid Apple Silicon (MPS) hanging issues.

Usage:
    python pseudolabel/train_autogluon_models.py \
        --time_limit 600 \
        --preset medium
"""

# FORCE CPU MODE BEFORE IMPORTING AUTOGLUON (fixes Apple Silicon MPS hanging)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
os.environ['OMP_NUM_THREADS'] = '1'      # Limit threads
os.environ['MPS_ENABLED'] = '0'          # Disable MPS on Apple Silicon

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

def extract_chemistry_features(smiles: str) -> dict:
    """Extract 21 chemistry-based features from SMILES string."""
    smiles_str = str(smiles).strip()
    
    if not smiles_str:
        return None
    
    try:
        # Basic counts (10 features)
        basic = {
            'smiles_length': float(len(smiles_str)),
            'carbon_count': float(smiles_str.count('C')),
            'nitrogen_count': float(smiles_str.count('N')),
            'oxygen_count': float(smiles_str.count('O')),
            'sulfur_count': float(smiles_str.count('S')),
            'fluorine_count': float(smiles_str.count('F')),
            'ring_count': float(smiles_str.count('c') + smiles_str.count('C1')),
            'double_bond_count': float(smiles_str.count('=')),
            'triple_bond_count': float(smiles_str.count('#')),
            'branch_count': float(smiles_str.count('(')),
        }
        
        # Chemistry-based features (11 additional)
        num_side_chains = float(smiles_str.count('('))
        backbone_carbons = float(smiles_str.count('C') - smiles_str.count('C('))
        aromatic_count = float(smiles_str.count('c'))
        h_bond_donors = float(smiles_str.count('O') + smiles_str.count('N'))
        h_bond_acceptors = float(smiles_str.count('O') + smiles_str.count('N'))
        num_rings = float(smiles_str.count('1') + smiles_str.count('2'))
        single_bonds = float(len(smiles_str) - smiles_str.count('=') - smiles_str.count('#') - aromatic_count)
        halogen_count = float(smiles_str.count('F') + smiles_str.count('Cl') + smiles_str.count('Br'))
        heteroatom_count = float(smiles_str.count('N') + smiles_str.count('O') + smiles_str.count('S'))
        mw_estimate = float(
            smiles_str.count('C') * 12 + smiles_str.count('O') * 16 +
            smiles_str.count('N') * 14 + smiles_str.count('S') * 32 + smiles_str.count('F') * 19
        )
        branching_ratio = num_side_chains / max(backbone_carbons, 1.0)
        
        features = {
            **basic,
            'num_side_chains': num_side_chains,
            'backbone_carbons': backbone_carbons,
            'aromatic_count': aromatic_count,
            'h_bond_donors': h_bond_donors,
            'h_bond_acceptors': h_bond_acceptors,
            'num_rings': num_rings,
            'single_bonds': single_bonds,
            'halogen_count': halogen_count,
            'heteroatom_count': heteroatom_count,
            'mw_estimate': mw_estimate,
            'branching_ratio': branching_ratio,
        }
        
        return features
    except Exception as e:
        print(f"Warning: Failed to extract features from SMILES '{smiles}': {e}", file=sys.stderr)
        return None

def main(args):
    print("\n" + "=" * 80)
    print("TRAINING AUTOGLUON MODELS FOR PSEUDO-LABEL GENERATION")
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
    
    # Extract features
    print("🔍 Extracting 21 chemistry features from all SMILES...")
    feature_rows = []
    for idx, smiles in tqdm(enumerate(train_df['SMILES']), total=len(train_df), desc="Extracting"):
        features = extract_chemistry_features(smiles)
        if features:
            feature_rows.append(features)
        else:
            print(f"   ⚠️  Failed to extract features for SMILES at row {idx}")
    
    X_features = pd.DataFrame(feature_rows)
    print(f"✅ Extracted features: {X_features.shape}\n")
    
    # Align with training data
    train_df_filtered = train_df.iloc[:len(X_features)].reset_index(drop=True)
    
    # Train AutoGluon models
    print("🤖 Training AutoGluon models for each property...")
    print(f"   Time limit per model: {args.time_limit} seconds")
    print(f"   Preset: {args.preset}\n")
    
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("❌ AutoGluon not installed!")
        print("   Install with: pip install autogluon")
        return
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    output_dir = Path('models/autogluon_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for target in target_cols:
        print(f"   Training AutoGluon for {target}...")
        
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
        
        # Train AutoGluon
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
            
            print(f"      ✅ {target} model trained and saved to {model_path}\n")
        
        except Exception as e:
            print(f"      ❌ Failed to train {target}: {e}\n")
    
    print("=" * 80)
    print("✅ AUTOGLUON MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved in: {output_dir}")
    print("\nReady to generate pseudo-labels with:")
    print("  python pseudolabel/generate_with_autogluon.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AutoGluon models for polymer property prediction.")
    parser.add_argument("--time_limit", type=int, default=600,
                        help="Time limit in seconds per model (default: 600s = 10 min)")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=['fast', 'medium', 'high', 'best'],
                        help="AutoGluon preset: fast (quick), medium (balanced), high (slow), best (very slow)")
    args = parser.parse_args()
    main(args)

