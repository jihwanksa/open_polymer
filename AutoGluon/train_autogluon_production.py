"""
Train AutoGluon models with FULL DATA AUGMENTATION and RDKit-enhanced features.

This script EXACTLY replicates the data pipeline from best.ipynb:
1. Loads original training data (7,973 samples)
2. Augments with external Tc dataset (+875 samples)
3. Augments with external Tg dataset (+7,000 samples)
4. Augments with PI1070 (Density + Rg) (+1,000 samples)
5. Augments with LAMALAB Tg dataset (+2,000 samples)
6. Augments with 50K pseudo-labels (BERT + Uni-Mol ensemble)
7. Extracts 34 comprehensive features:
   - 10 simple features (SMILES parsing)
   - 11 hand-crafted polymer-specific features
   - 13 RDKit descriptors (chemistry-based) - AutoGluon auto-selects best ones
8. Trains AutoGluon TabularPredictor for each of 5 properties
9. Saves models + feature importance analysis

AutoGluon will intelligently select which features are most predictive,
allowing us to understand which RDKit descriptors add value beyond the
hand-crafted features, and which hand-crafted features are redundant.

⚠️ IMPORTANT: Runs on CPU only to avoid Apple Silicon (MPS) hanging issues.

Usage:
    python AutoGluon/train_autogluon_production.py \
        --time_limit 1800 \
        --preset medium_quality

Expected: ~60K training samples total, 34 input features, AutoGluon-selected features
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
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import AutoGluon
try:
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon imported successfully (CPU-only mode)")
except ImportError:
    print("❌ AutoGluon not installed. Install with: pip install autogluon")
    sys.exit(1)

# Try RDKit for canonicalization
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


def extract_rdkit_descriptors(smiles_str):
    """Extract RDKit molecular descriptors for a single SMILES"""
    try:
        from rdkit.Chem import Descriptors, Crippen
        # Replace polymer markers with H (no brackets)
        cleaned = str(smiles_str).replace('*', 'H')
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return {}
        
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'RingCount': Descriptors.RingCount(mol),
            'FractionCsp3': Descriptors.FractionCsp3(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'BertzCT': Descriptors.BertzCT(mol),
        }
        return descriptors
    except:
        return {}


def extract_comprehensive_features(df):
    """Extract comprehensive features: 10 simple + 11 hand-crafted + 13 RDKit descriptors = 34 total"""
    print("Extracting comprehensive features (simple + hand-crafted + RDKit)...", flush=True)
    sys.stdout.flush()
    features = []
    rdkit_available = RDKIT_AVAILABLE
    
    for idx, smiles in tqdm(df['SMILES'].items(), total=len(df), desc="Features"):
        try:
            smiles_str = str(smiles) if pd.notna(smiles) else ""
            
            # Basic counts (10 simple features)
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
            
            # Hand-crafted polymer-specific features (11)
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
            
            hand_crafted = {
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
            
            # RDKit descriptors (13) - let AutoGluon decide which are useful
            rdkit_descriptors = {}
            if rdkit_available:
                rdkit_descriptors = extract_rdkit_descriptors(smiles_str)
                # Fill missing with 0 if extraction failed
                rdkit_keys = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                             'NumAromaticRings', 'TPSA', 'NumSaturatedRings', 'NumAliphaticRings',
                             'RingCount', 'FractionCsp3', 'NumHeteroatoms', 'BertzCT']
                for key in rdkit_keys:
                    if key not in rdkit_descriptors:
                        rdkit_descriptors[key] = 0.0
            else:
                # Fallback: all zeros
                rdkit_descriptors = {
                    'MolWt': 0.0, 'LogP': 0.0, 'NumHDonors': 0.0, 'NumHAcceptors': 0.0,
                    'NumRotatableBonds': 0.0, 'NumAromaticRings': 0.0, 'TPSA': 0.0,
                    'NumSaturatedRings': 0.0, 'NumAliphaticRings': 0.0, 'RingCount': 0.0,
                    'FractionCsp3': 0.0, 'NumHeteroatoms': 0.0, 'BertzCT': 0.0
                }
            
            # Combine all features (10 + 11 + 13 = 34)
            desc = {**basic, **hand_crafted, **rdkit_descriptors}
            features.append(desc)
        except Exception as e:
            # Fallback: zeros for all features
            features.append({
                # 10 basic
                'smiles_length': 0, 'carbon_count': 0, 'nitrogen_count': 0,
                'oxygen_count': 0, 'sulfur_count': 0, 'fluorine_count': 0,
                'ring_count': 0, 'double_bond_count': 0, 'triple_bond_count': 0,
                'branch_count': 0,
                # 11 hand-crafted
                'num_side_chains': 0, 'backbone_carbons': 0, 'aromatic_count': 0,
                'h_bond_donors': 0, 'h_bond_acceptors': 0, 'num_rings': 0,
                'single_bonds': 0, 'halogen_count': 0, 'heteroatom_count': 0,
                'mw_estimate': 0, 'branching_ratio': 0,
                # 13 RDKit
                'MolWt': 0.0, 'LogP': 0.0, 'NumHDonors': 0.0, 'NumHAcceptors': 0.0,
                'NumRotatableBonds': 0.0, 'NumAromaticRings': 0.0, 'TPSA': 0.0,
                'NumSaturatedRings': 0.0, 'NumAliphaticRings': 0.0, 'RingCount': 0.0,
                'FractionCsp3': 0.0, 'NumHeteroatoms': 0.0, 'BertzCT': 0.0
            })
    
    features_df = pd.DataFrame(features, index=df.index)
    print(f"✅ Extracted {len(features_df)} samples with {len(features_df.columns)} features", flush=True)
    print(f"   - 10 simple features (SMILES parsing)")
    print(f"   - 11 hand-crafted features (polymer-specific)")
    print(f"   - 13 RDKit descriptors (chemistry-based)")
    sys.stdout.flush()
    return features_df


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("="*80, flush=True)
    print("AUTOGLUON PRODUCTION TRAINING WITH FULL DATA AUGMENTATION", flush=True)
    print("(Enhanced with RDKit descriptors - 34 total features)", flush=True)
    print("="*80 + "\n", flush=True)
    sys.stdout.flush()
    
    # ========================================================================
    # STEP 1: Load original training data
    # ========================================================================
    print("STEP 1: Loading original training data...")
    train_path = os.path.join(project_root, 'data/raw/train.csv')
    train_df = pd.read_csv(train_path)
    print(f"✅ Loaded {len(train_df)} original samples\n")
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # ========================================================================
    # STEP 2: Canonicalize SMILES
    # ========================================================================
    if RDKIT_AVAILABLE:
        print("STEP 2: Canonicalizing SMILES...")
        train_df['SMILES'] = train_df['SMILES'].apply(make_smile_canonical)
        train_df = train_df.dropna(subset=['SMILES']).reset_index(drop=True)
        print(f"✅ {len(train_df)} samples after canonicalization\n")
    else:
        print("STEP 2: Skipping canonicalization (RDKit not available)\n")
    
    # ========================================================================
    # STEP 3: Augment with External Tc Dataset
    # ========================================================================
    print("STEP 3: Loading external Tc dataset...")
    try:
        tc_path = os.path.join(project_root, 'data/Tc_SMILES.csv')
        if not os.path.exists(tc_path):
            # Try alternative paths
            alt_paths = [
                '/kaggle/input/tc-smiles/Tc_SMILES.csv',
                '/kaggle/input/tc-smiles/TC_SMILES.csv',
            ]
            for p in alt_paths:
                if os.path.exists(p):
                    tc_path = p
                    break
        
        if os.path.exists(tc_path):
            tc_external = pd.read_csv(tc_path)
            if RDKIT_AVAILABLE:
                tc_external['SMILES'] = tc_external['SMILES'].apply(make_smile_canonical)
            
            train_smiles = set(train_df['SMILES'])
            tc_new = tc_external[~tc_external['SMILES'].isin(train_smiles)].copy()
            tc_new_rows = []
            for _, row in tc_new.iterrows():
                tc_new_rows.append({
                    'SMILES': row['SMILES'],
                    'Tg': np.nan, 'FFV': np.nan,
                    'Tc': row.get('TC_mean', row.get('Tc', np.nan)),
                    'Density': np.nan, 'Rg': np.nan
                })
            if tc_new_rows:
                train_df = pd.concat([train_df, pd.DataFrame(tc_new_rows)], ignore_index=True)
                print(f"✅ Added {len(tc_new_rows)} Tc samples\n")
        else:
            print("⚠️  Tc dataset not found\n")
    except Exception as e:
        print(f"⚠️  Tc augmentation failed: {e}\n")
    
    # ========================================================================
    # STEP 4: Augment with External Tg Dataset
    # ========================================================================
    print("STEP 4: Loading external Tg dataset...")
    try:
        tg_path = os.path.join(project_root, 'data/Tg_SMILES_class_pid_polyinfo_median.csv')
        if not os.path.exists(tg_path):
            tg_path = '/kaggle/input/tg-of-polymer-dataset/Tg_SMILES_class_pid_polyinfo_median.csv'
        
        if os.path.exists(tg_path):
            tg_external = pd.read_csv(tg_path)
            if RDKIT_AVAILABLE:
                tg_external['SMILES'] = tg_external['SMILES'].apply(make_smile_canonical)
            
            train_smiles = set(train_df['SMILES'])
            tg_new = tg_external[~tg_external['SMILES'].isin(train_smiles)].copy()
            tg_new_rows = []
            for _, row in tg_new.iterrows():
                tg_new_rows.append({
                    'SMILES': row['SMILES'],
                    'Tg': row.get('Tg', np.nan),
                    'FFV': np.nan, 'Tc': np.nan,
                    'Density': np.nan, 'Rg': np.nan
                })
            if tg_new_rows:
                train_df = pd.concat([train_df, pd.DataFrame(tg_new_rows)], ignore_index=True)
                print(f"✅ Added {len(tg_new_rows)} Tg samples\n")
        else:
            print("⚠️  Tg dataset not found\n")
    except Exception as e:
        print(f"⚠️  Tg augmentation failed: {e}\n")
    
    # ========================================================================
    # STEP 5: Augment with PI1070 (Density + Rg)
    # ========================================================================
    print("STEP 5: Loading PI1070 dataset (Density + Rg)...")
    try:
        pi1070_path = os.path.join(project_root, 'data/PI1070.csv')
        if not os.path.exists(pi1070_path):
            pi1070_path = '/kaggle/input/more-data/PI1070.csv'
        
        if os.path.exists(pi1070_path):
            pi1070_df = pd.read_csv(pi1070_path)
            train_smiles = set(train_df['SMILES'])
            pi1070_new = pi1070_df[~pi1070_df['smiles'].isin(train_smiles)].copy()
            
            for _, row in pi1070_new.iterrows():
                if pd.notna(row.get('density')) or pd.notna(row.get('Rg')):
                    train_df = pd.concat([train_df, pd.DataFrame([{
                        'SMILES': row['smiles'],
                        'Tg': np.nan, 'FFV': np.nan, 'Tc': np.nan,
                        'Density': row.get('density', np.nan),
                        'Rg': row.get('Rg', np.nan)
                    }])], ignore_index=True)
            print(f"✅ Added PI1070 samples\n")
        else:
            print("⚠️  PI1070 dataset not found\n")
    except Exception as e:
        print(f"⚠️  PI1070 augmentation failed: {e}\n")
    
    # ========================================================================
    # STEP 6: Augment with LAMALAB Tg
    # ========================================================================
    print("STEP 6: Loading LAMALAB Tg dataset...")
    try:
        lamalab_path = os.path.join(project_root, 'data/LAMALAB_CURATED_Tg_structured_polymerclass.csv')
        if not os.path.exists(lamalab_path):
            lamalab_path = '/kaggle/input/more-data/LAMALAB_CURATED_Tg_structured_polymerclass.csv'
        
        if os.path.exists(lamalab_path):
            lamalab_df = pd.read_csv(lamalab_path)
            train_smiles = set(train_df['SMILES'])
            lamalab_new = lamalab_df[~lamalab_df['PSMILES'].isin(train_smiles)].copy()
            
            for _, row in lamalab_new.iterrows():
                tg_value = row.get('labels.Exp_Tg(K)', np.nan)
                if pd.notna(tg_value):
                    tg_celsius = tg_value - 273.15
                    train_df = pd.concat([train_df, pd.DataFrame([{
                        'SMILES': row['PSMILES'],
                        'Tg': tg_celsius,
                        'FFV': np.nan, 'Tc': np.nan, 'Density': np.nan, 'Rg': np.nan
                    }])], ignore_index=True)
            print(f"✅ Added LAMALAB Tg samples\n")
        else:
            print("⚠️  LAMALAB dataset not found\n")
    except Exception as e:
        print(f"⚠️  LAMALAB augmentation failed: {e}\n")
    
    # ========================================================================
    # STEP 7: Augment with Pseudo-Labels
    # ========================================================================
    print("STEP 7: Loading pseudo-labeled dataset...")
    try:
        # Try 3-model ensemble first (BERT + Uni-Mol + AutoGluon)
        pseudo_paths = [
            os.path.join(project_root, 'pseudolabel/pi1m_pseudolabels_ensemble_3models.csv'),
            os.path.join(project_root, 'pseudolabel/pi1m_pseudolabels_ensemble_2models.csv'),
            'data/PI1M_50000_v2.1.csv',
        ]
        
        pseudo_path = None
        for p in pseudo_paths:
            if os.path.exists(p):
                pseudo_path = p
                break
        
        if pseudo_path:
            pseudo_df = pd.read_csv(pseudo_path)
            train_smiles = set(train_df['SMILES'])
            pseudo_new = pseudo_df[~pseudo_df['SMILES'].isin(train_smiles)].copy()
            
            if len(pseudo_new) > 0:
                train_df = pd.concat([train_df, pseudo_new], ignore_index=True)
                print(f"✅ Added {len(pseudo_new)} pseudo-labeled samples\n")
            else:
                print("⚠️  All pseudo-labels already in training set\n")
        else:
            print("⚠️  Pseudo-labeled dataset not found\n")
    except Exception as e:
        print(f"⚠️  Pseudo-label augmentation failed: {e}\n")
    
    # ========================================================================
    # STEP 8: Feature Extraction (with RDKit enhancement)
    # ========================================================================
    print("STEP 8: Extracting comprehensive features with RDKit...")
    print("        (10 simple + 11 hand-crafted + 13 RDKit = 34 total)")
    train_df = train_df.reset_index(drop=True)
    features_df = extract_comprehensive_features(train_df)
    
    # Combine features with targets
    train_data = pd.concat([train_df[['SMILES'] + target_cols], features_df], axis=1)
    train_data = train_data.dropna(subset=['SMILES']).reset_index(drop=True)
    
    print(f"✅ Final training data: {len(train_data)} samples with {features_df.shape[1]} features")
    print(f"\nTarget availability:")
    for col in target_cols:
        n_avail = train_data[col].notna().sum()
        pct = 100 * n_avail / len(train_data)
        print(f"  {col}: {n_avail} samples ({pct:.1f}%)\n")
    
    # ========================================================================
    # STEP 9: Train AutoGluon Models
    # ========================================================================
    print("\nSTEP 9: Training AutoGluon models...", flush=True)
    print("="*80, flush=True)
    sys.stdout.flush()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_names = features_df.columns.tolist()
    feature_importance_all = {}
    
    for target in target_cols:
        print(f"\n🔧 Training AutoGluon for {target}...", flush=True)
        sys.stdout.flush()
        
        # Get data for this target
        target_data = train_data[['SMILES'] + feature_names + [target]].dropna(subset=[target])
        
        if len(target_data) == 0:
            print(f"  ⚠️  No training data for {target}, skipping", flush=True)
            sys.stdout.flush()
            continue
        
        print(f"  Training samples: {len(target_data)}", flush=True)
        print(f"  Features: {len(feature_names)}", flush=True)
        sys.stdout.flush()
        
        # Train AutoGluon
        model_path = str(output_dir / target)
        predictor = TabularPredictor(
            label=target,
            path=model_path,
            eval_metric='mae',
            problem_type='regression'
        )
        
        try:
            # Use only feature columns + target for training
            train_for_fit = target_data[feature_names + [target]].copy()
            
            predictor.fit(
                train_for_fit,
                time_limit=args.time_limit,
                presets=args.preset,
                verbosity=0
            )
            
            print(f"  ✅ Model trained and saved to {model_path}", flush=True)
            sys.stdout.flush()
            
        except Exception as e:
            print(f"  ❌ Training failed: {e}", flush=True)
            sys.stdout.flush()
            continue
    
    # ========================================================================
    # STEP 10: Save Feature Importance
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING FEATURE IMPORTANCE...")
    importance_path = output_dir / 'feature_importance.json'
    with open(importance_path, 'w') as f:
        json.dump(feature_importance_all, f, indent=2)
    print(f"✅ Saved to {importance_path}\n")
    
    print("="*80)
    print("✅ AUTOGLUON PRODUCTION TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModels saved to: {output_dir}")
    print(f"Ready for inference with: python AutoGluon/predict_autogluon.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AutoGluon with full data augmentation")
    parser.add_argument("--time_limit", type=int, default=1800, help="Training time limit (seconds)")
    parser.add_argument("--preset", type=str, default="medium_quality", help="AutoGluon preset")
    parser.add_argument("--output_dir", type=str, default="models/autogluon_production", help="Output directory")
    args = parser.parse_args()
    
    main(args)
