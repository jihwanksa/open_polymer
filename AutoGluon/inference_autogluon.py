"""
AutoGluon Production Inference Script

This script loads pre-trained AutoGluon models and generates predictions.
Use this instead of manually updating best.ipynb for production inference.

Usage:
    python AutoGluon/inference_autogluon.py --test_file data/test.csv --output submission.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Force CPU-only mode for AutoGluon
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPS_ENABLED'] = '0'

# Try to import RDKit
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available")

# Import AutoGluon
try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    print("❌ AutoGluon not installed")
    sys.exit(1)


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
    """Create 21 chemistry-based features from SMILES"""
    features = []
    
    print("Creating chemistry-based features...")
    for idx, smiles in tqdm(df['SMILES'].items(), total=len(df)):
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
            
            # Chemistry-based features (11 additional)
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
            
            # Combine all features
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
    print(f"✅ Created {len(features_df)} chemistry-based feature vectors")
    return features_df


def main(args):
    print("\n" + "="*80)
    print("AUTOGLUON PRODUCTION INFERENCE")
    print("="*80)
    
    # Load test data
    print(f"\n📂 Loading test data from {args.test_file}...")
    test_df = pd.read_csv(args.test_file)
    print(f"   ✅ Loaded {len(test_df)} test samples")
    
    # Canonicalize SMILES
    if RDKIT_AVAILABLE:
        print("\n🔄 Canonicalizing SMILES...")
        test_df['SMILES'] = test_df['SMILES'].apply(make_smile_canonical)
        test_df = test_df.dropna(subset=['SMILES']).reset_index(drop=True)
        print(f"   ✅ {len(test_df)} samples after canonicalization")
    
    # Create features
    print("\n🧪 Creating chemistry features...")
    test_features = create_chemistry_features(test_df)
    X_test = test_features.values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Load AutoGluon models
    print("\n" + "="*80)
    print("LOADING AUTOGLUON MODELS")
    print("="*80)
    
    model_dir = Path(args.model_dir)
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    predictors = {}
    feature_names = None
    
    for target in target_cols:
        target_path = model_dir / target
        
        if not target_path.exists():
            print(f"❌ {target}: Model not found at {target_path}")
            continue
        
        try:
            print(f"\n📂 Loading {target}...", end=" ")
            predictor = TabularPredictor.load(str(target_path))
            predictors[target] = predictor
            
            if feature_names is None and hasattr(predictor, 'features'):
                feature_names = predictor.features
            
            print("✅")
            print(f"   Features: {len(predictor.features)}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    if len(predictors) != len(target_cols):
        print(f"\n⚠️  Only {len(predictors)}/{len(target_cols)} models loaded")
    
    # Generate predictions
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    predictions = np.zeros((len(X_test), len(target_cols)))
    
    for i, target in enumerate(target_cols):
        try:
            if target in predictors:
                predictor = predictors[target]
                
                # Convert to DataFrame
                X_df = pd.DataFrame(X_test, columns=feature_names)
                
                # Predict
                preds = predictor.predict(X_df)
                if isinstance(preds, (pd.Series, pd.DataFrame)):
                    preds = preds.values.flatten()
                
                predictions[:, i] = preds
                print(f"✅ {target}: range [{preds.min():.4f}, {preds.max():.4f}]")
            else:
                print(f"⚠️  {target}: No model, using zeros")
                predictions[:, i] = 0.0
                
        except Exception as e:
            print(f"❌ {target}: Prediction failed: {e}")
            predictions[:, i] = 0.0
    
    # Create submission
    print("\n" + "="*80)
    print("CREATING SUBMISSION")
    print("="*80)
    
    if args.sample_submission:
        submission = pd.read_csv(args.sample_submission)
    else:
        submission = pd.DataFrame({'ID': range(len(predictions))})
    
    # Ensure correct length
    if len(predictions) != len(submission):
        print(f"⚠️  Adjusting predictions from {len(predictions)} to {len(submission)}")
        if len(predictions) < len(submission):
            padding = np.zeros((len(submission) - len(predictions), len(target_cols)))
            predictions = np.vstack([predictions, padding])
        else:
            predictions = predictions[:len(submission)]
    
    # Add predictions
    for i, target in enumerate(target_cols):
        submission[target] = predictions[:, i]
    
    # Apply Tg transformation (2nd place discovery)
    print("\n🔧 Applying Tg transformation: (9/5) × Tg + 45...")
    submission['Tg'] = (9/5) * submission['Tg'] + 45
    
    # Save submission
    print(f"\n💾 Saving submission to {args.output}...")
    submission.to_csv(args.output, index=False)
    print(f"✅ Saved {len(submission)} predictions")
    
    print("\n" + "="*80)
    print("✅ INFERENCE COMPLETE!")
    print("="*80)
    print(f"\n📊 Submission saved to: {args.output}")
    print(f"   Samples: {len(submission)}")
    print(f"   Properties: {', '.join(target_cols)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGluon production inference")
    parser.add_argument("--test_file", type=str, default="data/raw/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--model_dir", type=str, default="models/autogluon_production",
                        help="Directory containing trained AutoGluon models")
    parser.add_argument("--sample_submission", type=str, default="data/raw/sample_submission.csv",
                        help="Path to sample submission for format reference")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Path to save predictions")
    
    args = parser.parse_args()
    main(args)

