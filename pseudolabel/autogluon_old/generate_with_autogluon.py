"""Generate pseudo-labels using trained AutoGluon models (21 features).

This script:
1. Extracts 21 chemistry features from 50K unlabeled SMILES
2. Uses trained AutoGluon models to generate predictions
3. Applies Tg transformation
4. Saves final pseudo-labels

⚠️  IMPORTANT: Runs on CPU only to avoid Apple Silicon (MPS) hanging issues.

Usage:
    python pseudolabel/generate_with_autogluon.py \
        --input_data data/PI1M_50000_v2.1.csv \
        --models_dir models/autogluon_models \
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

def apply_tg_transformation(tg_value: float) -> float:
    return (9/5) * tg_value + 45

def main(args):
    print("\n" + "=" * 80)
    print("PSEUDO-LABEL GENERATION WITH AUTOGLUON")
    print("=" * 80 + "\n")
    
    # Check models directory
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("   First train models with: python pseudolabel/train_autogluon_models.py")
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
    
    # Extract features
    print("🔍 Extracting 21 chemistry features from all 50K SMILES...")
    feature_rows = []
    for idx, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Extracting"):
        features = extract_chemistry_features(smiles)
        if features:
            feature_rows.append(features)
        else:
            # Add zero features for failed extraction
            feature_rows.append({f: 0.0 for f in range(21)})
    
    X_features = pd.DataFrame(feature_rows)
    print(f"✅ Extracted features: {X_features.shape}\n")
    
    # Load AutoGluon models and generate predictions
    print("🤖 Loading AutoGluon models and generating predictions...")
    
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("❌ AutoGluon not installed!")
        print("   Install with: pip install autogluon")
        return
    
    target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    predictions_df = pd.DataFrame({'SMILES': smiles_list})
    
    for target in target_names:
        model_path = models_dir / target
        
        if not model_path.exists():
            print(f"   ⚠️  Model not found for {target} at {model_path}")
            print(f"      Filling with NaN")
            predictions_df[target] = np.nan
            continue
        
        print(f"\n   {target}:")
        try:
            predictor = TabularPredictor.load(str(model_path))
            preds = predictor.predict(X_features)
            predictions_df[target] = preds.values
            
            print(f"      Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")
            print(f"      ✅ Predictions generated")
        
        except Exception as e:
            print(f"      ❌ Failed to generate predictions: {e}")
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
    print("✅ PSEUDO-LABEL GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Output: {args.output_path}")
    print(f"   Samples: {len(predictions_df)}")
    print(f"   Properties: {', '.join(target_names)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-labels using trained AutoGluon models.")
    parser.add_argument("--input_data", type=str, default="data/PI1M_50000_v2.1.csv",
                        help="Path to input CSV with SMILES column")
    parser.add_argument("--models_dir", type=str, default="models/autogluon_models",
                        help="Path to directory with trained AutoGluon models")
    parser.add_argument("--output_path", type=str, default="pseudolabel/pi1m_pseudolabels_autogluon.csv",
                        help="Path to save pseudo-labels")
    parser.add_argument("--apply_tg_transform", action="store_true", default=True,
                        help="Apply Tg transformation")
    args = parser.parse_args()
    main(args)

