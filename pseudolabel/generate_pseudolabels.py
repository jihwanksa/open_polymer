"""
Generate pseudo-labels for 50K unlabeled polymers from PI1M dataset
using the trained Random Forest v85 model.

This script replicates the 1st place solution approach:
- Load trained Random Forest ensemble model (v85)
- Extract 21 chemistry-based features from 50K SMILES
- Generate predictions for all 5 properties (Tg, FFV, Tc, Density, Rg)
- Save pseudo-labeled dataset

The 1st place solution used ensemble predictions from BERT, AutoGluon, and Uni-Mol,
but this script demonstrates how to generate pseudo-labels using our best Random Forest model.

Usage:
    python pseudolabel/generate_pseudolabels.py --model_path models/random_forest_v85_best.pkl \\
                                                 --input_data data/PI1M_50000_v2.1.csv \\
                                                 --output_path pseudolabel/pi1m_pseudolabels.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import RDKit for SMILES canonicalization
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available - SMILES canonicalization will be skipped")


def make_smile_canonical(smile):
    """Canonicalize SMILES to avoid duplicates"""
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


def create_chemistry_features(df):
    """Create 21 chemistry-based features from SMILES (same as v85)"""
    features = []
    
    print("Creating chemistry-based features...")
    for idx, smiles in tqdm(df['SMILES'].items(), total=len(df), desc="Feature extraction"):
        try:
            smiles_str = str(smiles) if pd.notna(smiles) else ""
            
            # Basic counts (10 features)
            basic = {
                'smiles_length': len(smiles_str),
                'carbon_count': smiles_str.count('C'),
                'nitrogen_count': smiles_str.count('N'),
                'oxygen_count': smiles_str.count('O'),
                'sulfur_count': smiles_str.count('S'),
                'fluorine_count': smiles_str.count('F'),
                'ring_count': smiles_str.count('c') + smiles_str.count('C1'),
                'double_bond_count': smiles_str.count('='),
                'triple_bond_count': smiles_str.count('#'),
                'branch_count': smiles_str.count('('),
            }
            
            # Chemistry-based features (11 additional features)
            num_side_chains = smiles_str.count('(')
            backbone_carbons = smiles_str.count('C') - smiles_str.count('C(')
            aromatic_count = smiles_str.count('c')
            h_bond_donors = smiles_str.count('O') + smiles_str.count('N')
            h_bond_acceptors = smiles_str.count('O') + smiles_str.count('N')
            num_rings = smiles_str.count('1') + smiles_str.count('2')
            single_bonds = len(smiles_str) - smiles_str.count('=') - smiles_str.count('#') - aromatic_count
            halogen_count = smiles_str.count('F') + smiles_str.count('Cl') + smiles_str.count('Br')
            heteroatom_count = smiles_str.count('N') + smiles_str.count('O') + smiles_str.count('S')
            mw_estimate = (smiles_str.count('C') * 12 + smiles_str.count('O') * 16 + 
                          smiles_str.count('N') * 14 + smiles_str.count('S') * 32 + smiles_str.count('F') * 19)
            branching_ratio = num_side_chains / max(backbone_carbons, 1)
            
            # Combine all features (21 total)
            desc = {
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
            features.append(desc)
        except:
            # Fallback: zeros
            features.append({
                'smiles_length': 0, 'carbon_count': 0, 'nitrogen_count': 0,
                'oxygen_count': 0, 'sulfur_count': 0, 'fluorine_count': 0,
                'ring_count': 0, 'double_bond_count': 0, 'triple_bond_count': 0,
                'branch_count': 0, 'num_side_chains': 0, 'backbone_carbons': 0,
                'aromatic_count': 0, 'h_bond_donors': 0, 'h_bond_acceptors': 0,
                'num_rings': 0, 'single_bonds': 0, 'halogen_count': 0,
                'heteroatom_count': 0, 'mw_estimate': 0, 'branching_ratio': 0,
            })
    
    features_df = pd.DataFrame(features, index=df.index)
    print(f"✅ Created {len(features_df)} feature vectors with {len(features_df.columns)} features")
    return features_df


def load_model(model_path):
    """Load trained Random Forest model"""
    print(f"\n📂 Loading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Model loaded successfully!")
    return data


def generate_pseudolabels(model_data, features_df, target_names=['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    """
    Generate pseudo-labels using the trained ensemble models
    
    Args:
        model_data: Loaded model dictionary with 'models' and 'scalers'
        features_df: Feature matrix (n_samples, 21)
        target_names: List of target property names
    
    Returns:
        predictions: Array of shape (n_samples, 5) with predictions for each property
    """
    print(f"\n🔮 Generating pseudo-labels for {len(features_df)} samples...")
    
    models = model_data['models']
    scalers = model_data['scalers']
    
    predictions = np.zeros((len(features_df), len(target_names)))
    
    for i, target in enumerate(target_names):
        print(f"\n   Predicting {target}...", end='\r')
        
        try:
            if target in models and target in scalers:
                scaler = scalers[target]
                ensemble_models = models[target]
                
                # Prepare features
                X_test_clean = np.nan_to_num(features_df.values, nan=0.0, posinf=1e6, neginf=-1e6)
                X_test_scaled = scaler.transform(X_test_clean)
                
                # Ensemble prediction
                ensemble_preds = np.array([model.predict(X_test_scaled) for model in ensemble_models])
                pred = ensemble_preds.mean(axis=0)
                predictions[:, i] = pred
                
                print(f"   ✅ {target}: Generated {len(pred)} predictions")
            else:
                print(f"   ⚠️  {target}: Model or scaler not found, using zeros")
                predictions[:, i] = 0.0
                
        except Exception as e:
            print(f"   ❌ {target}: Prediction failed: {e}")
            predictions[:, i] = 0.0
    
    return predictions


def apply_tg_transformation(predictions, target_names=['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    """
    Apply Tg transformation from 2nd place solution
    Tg_transformed = (9/5) * Tg + 45
    """
    tg_idx = target_names.index('Tg')
    predictions[:, tg_idx] = (9/5) * predictions[:, tg_idx] + 45
    return predictions


def save_pseudolabels(smiles, predictions, output_path, target_names=['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    """Save pseudo-labeled dataset to CSV"""
    print(f"\n💾 Saving pseudo-labels to {output_path}...")
    
    df = pd.DataFrame({
        'SMILES': smiles,
        **{target: predictions[:, i] for i, target in enumerate(target_names)}
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} pseudo-labeled samples to {output_path}")
    
    # Print summary
    print(f"\n📊 Pseudo-label Summary:")
    print(f"   Total samples: {len(df)}")
    for target in target_names:
        print(f"   {target}: min={df[target].min():.2f}, max={df[target].max():.2f}, mean={df[target].mean():.2f}")
    
    return df


def main(args):
    print("\n" + "="*80)
    print("PSEUDO-LABEL GENERATION USING TRAINED RANDOM FOREST MODEL (v85)")
    print("="*80)
    
    # Load input SMILES
    print(f"\n📂 Loading input SMILES from {args.input_data}...")
    df = pd.read_csv(args.input_data)
    print(f"   Loaded {len(df)} samples")
    
    # If CSV has columns already, extract SMILES
    if 'SMILES' not in df.columns:
        print(f"   Warning: 'SMILES' column not found. Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    smiles = df['SMILES'].values
    df_smiles = df[['SMILES']].copy()
    
    # Canonicalize SMILES
    if RDKIT_AVAILABLE:
        print("\n🔄 Canonicalizing SMILES...")
        df_smiles['SMILES'] = df_smiles['SMILES'].apply(make_smile_canonical)
        print(f"   ✅ SMILES canonicalization complete!")
    
    # Extract features
    features = create_chemistry_features(df_smiles)
    
    # Load trained model
    model_data = load_model(args.model_path)
    
    # Generate predictions
    predictions = generate_pseudolabels(model_data, features)
    
    # Apply Tg transformation (optional, based on 2nd place solution)
    if args.apply_tg_transform:
        print("\n🔧 Applying Tg transformation: (9/5) × Tg + 45...")
        predictions = apply_tg_transformation(predictions)
    
    # Save results
    pseudolabel_df = save_pseudolabels(smiles, predictions, args.output_path)
    
    print(f"\n{'='*80}")
    print(f"✅ PSEUDO-LABEL GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Review pseudo-labels in {args.output_path}")
    print(f"2. Concatenate with original training data for augmentation")
    print(f"3. Train final model with augmented data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-labels using trained Random Forest model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/random_forest_v85_best.pkl",
        help="Path to trained Random Forest model"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="data/PI1M_50000_v2.1.csv",
        help="Path to input CSV with SMILES column"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="pseudolabel/pi1m_pseudolabels_generated.csv",
        help="Path to save generated pseudo-labels"
    )
    parser.add_argument(
        "--apply_tg_transform",
        action="store_true",
        default=True,
        help="Apply Tg transformation: (9/5) × Tg + 45"
    )
    
    args = parser.parse_args()
    main(args)

