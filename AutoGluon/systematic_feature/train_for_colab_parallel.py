"""
Systematic Feature Analysis for Google Colab with GPU/TPU acceleration

Supports 8 configurations (A-H) as defined in PLAN.md:
- A: Simple features only (10)
- B: Hand-crafted only (11)
- C: Current baseline (34)
- D: Expanded RDKit (56)
- E: All RDKit (~221)
- F: RDKit only expanded (45)
- G: No simple features (24)
- H: No hand-crafted features (23)

Usage in Colab:
    !git clone https://github.com/jihwanksa/open_polymer.git
    %cd open_polymer
    !python AutoGluon/systematic_feature/train_for_colab.py --config C --time_limit 300
"""

import os
import sys

# Detect if running in Colab
try:
    from google.colab import drive
    IN_COLAB = True
    print("✅ Running in Google Colab", flush=True)
except ImportError:
    IN_COLAB = False
    print("Running locally", flush=True)

# GPU settings for Colab
if IN_COLAB:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MPS_ENABLED'] = '0'

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Import AutoGluon
try:
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon imported successfully")
except ImportError:
    print("❌ AutoGluon not installed")
    sys.exit(1)

# Try RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    RDKIT_AVAILABLE = True
    print("✅ RDKit available")
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available")


# ============================================================================
# RDKit Descriptors: Current, Expanded, and All Available
# ============================================================================

RDKIT_CURRENT_13 = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumAromaticRings', 'TPSA', 'NumSaturatedRings', 'NumAliphaticRings',
    'RingCount', 'FractionCsp3', 'NumHeteroatoms', 'BertzCT'
]

RDKIT_EXPANDED_35 = [
    # Original 13
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumAromaticRings', 'TPSA', 'NumSaturatedRings', 'NumAliphaticRings',
    'RingCount', 'FractionCsp3', 'NumHeteroatoms', 'BertzCT',
    # Additional descriptors (22 more)
    'ExactMolWt', 'HeavyAtomCount', 'NAtoms', 'NumAmideBonds',
    'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumSaturatedCarbocycles',
    'NumSaturatedHeterocycles', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
    'LabuteASA', 'PercentVSA_EState1', 'PercentVSA_EState2', 'PercentVSA_EState3',
    'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3',
    'Chi0', 'Chi1'
]

RDKIT_ALL_DESCRIPTORS = [
    # Molecular weight & size
    'MolWt', 'ExactMolWt', 'HeavyAtomCount', 'NAtoms',
    # Lipophilicity
    'LogP', 'TPSA', 'LabuteASA',
    # H-bonding
    'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms',
    # Aromaticity
    'NumAromaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'FractionCsp3',
    # Rings & cycles
    'RingCount', 'NumSaturatedRings', 'NumAliphaticRings', 
    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
    'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
    # Bonds
    'NumRotatableBonds', 'NumAmideBonds', 'NumSulfonamideBonds',
    # Topological
    'BertzCT', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'Chi0', 'Chi1', 'Chi2', 'Chi3',
    'LabuteELF10', 'PercentVSA_EState1', 'PercentVSA_EState2', 'PercentVSA_EState3',
    'PercentVSA_EState4', 'PercentVSA_EState5', 'PercentVSA_EState6', 'PercentVSA_EState7',
    'PercentVSA_EState8', 'PercentVSA_EState9', 'PercentVSA_EState10',
    'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6',
    'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
    'PEOE_VSA13', 'PEOE_VSA14',
    # Additional descriptors
    'NumRotors', 'NumValenceElectrons', 'NumAliphaticCarbocycles',
    'NumAliphaticHeterocycles', 'NumAliphaticRings', 'SASA', 'TPSA',
]

# Remove duplicates and limit to ~60 most useful descriptors for "all"
RDKIT_ALL_DESCRIPTORS = list(set(RDKIT_ALL_DESCRIPTORS))[:60]


# ============================================================================
# Feature Configurations (from PLAN.md)
# ============================================================================

CONFIGURATIONS = {
    'A': {
        'name': 'A_simple_only',
        'description': 'Simple features only (SMILES counting) - 10 features',
        'simple': True,
        'hand_crafted': False,
        'rdkit_descriptors': []
    },
    'B': {
        'name': 'B_hand_crafted_only',
        'description': 'Hand-crafted features only (domain knowledge) - 11 features',
        'simple': False,
        'hand_crafted': True,
        'rdkit_descriptors': []
    },
    'C': {
        'name': 'C_current_baseline',
        'description': 'Current baseline (10 simple + 11 hand-crafted + 13 RDKit) - 34 features',
        'simple': True,
        'hand_crafted': True,
        'rdkit_descriptors': RDKIT_CURRENT_13
    },
    'D': {
        'name': 'D_expanded_rdkit',
        'description': 'Expanded RDKit (10 simple + 11 hand-crafted + 35 RDKit) - 56 features',
        'simple': True,
        'hand_crafted': True,
        'rdkit_descriptors': RDKIT_EXPANDED_35
    },
    'E': {
        'name': 'E_all_rdkit',
        'description': 'All RDKit (10 simple + 11 hand-crafted + ~60 RDKit) - ~81 features',
        'simple': True,
        'hand_crafted': True,
        'rdkit_descriptors': RDKIT_ALL_DESCRIPTORS
    },
    'F': {
        'name': 'F_rdkit_only_expanded',
        'description': 'RDKit only expanded (35 RDKit) - 35 features',
        'simple': False,
        'hand_crafted': False,
        'rdkit_descriptors': RDKIT_EXPANDED_35
    },
    'G': {
        'name': 'G_no_simple',
        'description': 'No simple features (11 hand-crafted + 13 RDKit) - 24 features',
        'simple': False,
        'hand_crafted': True,
        'rdkit_descriptors': RDKIT_CURRENT_13
    },
    'H': {
        'name': 'H_no_hand_crafted',
        'description': 'No hand-crafted features (10 simple + 13 RDKit) - 23 features',
        'simple': True,
        'hand_crafted': False,
        'rdkit_descriptors': RDKIT_CURRENT_13
    }
}


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_rdkit_descriptors(smiles_str, descriptor_names):
    """Extract specific RDKit descriptors for a SMILES string"""
    if not RDKIT_AVAILABLE or not descriptor_names:
        return {}
    
    try:
        cleaned = str(smiles_str).replace('[*]', '[H]').replace('*', '[H]')
        cleaned = cleaned.replace('[[H]]', '[H]')
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return {name: 0.0 for name in descriptor_names}
        
        descriptors = {}
        for name in descriptor_names:
            try:
                if hasattr(Descriptors, name):
                    descriptors[name] = getattr(Descriptors, name)(mol)
                elif hasattr(Crippen, name):
                    descriptors[name] = getattr(Crippen, name)(mol)
                else:
                    descriptors[name] = 0.0
            except:
                descriptors[name] = 0.0
        return descriptors
    except:
        return {name: 0.0 for name in descriptor_names}


def extract_simple_features(smiles_str):
    """Extract 10 simple features from SMILES"""
    s = str(smiles_str)
    return {
        'smiles_length': len(s),
        'carbon_count': s.count('C'),
        'nitrogen_count': s.count('N'),
        'oxygen_count': s.count('O'),
        'sulfur_count': s.count('S'),
        'fluorine_count': s.count('F'),
        'ring_count': s.count('c') + s.count('C1'),
        'double_bond_count': s.count('='),
        'triple_bond_count': s.count('#'),
        'branch_count': s.count('('),
    }


def extract_hand_crafted_features(smiles_str):
    """Extract 11 hand-crafted polymer-specific features"""
    s = str(smiles_str)
    num_side_chains = s.count('(')
    backbone_carbons = s.count('C') - s.count('C(')
    aromatic_count = s.count('c')
    
    return {
        'num_side_chains': num_side_chains,
        'backbone_carbons': backbone_carbons,
        'aromatic_count': aromatic_count,
        'h_bond_donors': s.count('O') + s.count('N'),
        'h_bond_acceptors': s.count('O') + s.count('N'),
        'num_rings': s.count('1') + s.count('2'),
        'single_bonds': len(s) - s.count('=') - s.count('#') - aromatic_count,
        'halogen_count': s.count('F') + s.count('Cl') + s.count('Br'),
        'heteroatom_count': s.count('N') + s.count('O') + s.count('S'),
        'mw_estimate': (s.count('C') * 12 + s.count('O') * 16 + 
                       s.count('N') * 14 + s.count('S') * 32 + s.count('F') * 19),
        'branching_ratio': num_side_chains / max(backbone_carbons, 1),
    }


def extract_features_for_config(df, config_key):
    """Extract features based on configuration"""
    config = CONFIGURATIONS[config_key]
    all_features = []
    
    for smiles in tqdm(df['SMILES'], desc=f"Extracting features for config {config_key}"):
        row = {}
        
        if config['simple']:
            row.update(extract_simple_features(smiles))
        
        if config['hand_crafted']:
            row.update(extract_hand_crafted_features(smiles))
        
        if config['rdkit_descriptors']:
            row.update(extract_rdkit_descriptors(smiles, config['rdkit_descriptors']))
        
        all_features.append(row)
    
    return pd.DataFrame(all_features)


# ============================================================================
# Main Training Function
# ============================================================================

def get_project_root():
    """Get project root path based on environment"""
    if IN_COLAB:
        # Try common paths
        paths = [
            '/content/open_polymer',
            '/content/drive/MyDrive/open_polymer',
            '/root/open_polymer'
        ]
        for p in paths:
            if os.path.exists(p) and os.path.exists(os.path.join(p, 'data/raw/train.csv')):
                return p
        raise FileNotFoundError("Project not found in common Colab paths")
    else:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_and_augment_data(project_root):
    """Load training data and augment with external datasets (FULL pipeline from train_autogluon_production.py)"""
    
    print("\n" + "="*70, flush=True)
    print("STEP 1: Loading and augmenting training data...", flush=True)
    print("="*70, flush=True)
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Load original training data
    train_path = os.path.join(project_root, 'data/raw/train.csv')
    train_df = pd.read_csv(train_path)
    print(f"✅ Loaded original data: {len(train_df)} samples", flush=True)
    
    # Augment with Tc dataset
    print("\n→ Augmenting with Tc dataset...", flush=True)
    try:
        tc_path = os.path.join(project_root, 'data/Tc_SMILES.csv')
        if os.path.exists(tc_path):
            tc_df = pd.read_csv(tc_path)
            tc_new_rows = []
            for _, row in tc_df.iterrows():
                tc_new_rows.append({
                    'SMILES': row['SMILES'],
                    'Tg': np.nan, 'FFV': np.nan,
                    'Tc': row.get('TC_mean', row.get('Tc', np.nan)),
                    'Density': np.nan, 'Rg': np.nan
                })
            if tc_new_rows:
                train_df = pd.concat([train_df, pd.DataFrame(tc_new_rows)], ignore_index=True)
                print(f"   ✅ Added {len(tc_new_rows)} Tc samples")
        else:
            print(f"   ⚠️  Tc dataset not found")
    except Exception as e:
        print(f"   ⚠️  Tc augmentation failed: {str(e)[:50]}")
    
    # Augment with Tg dataset
    print("→ Augmenting with Tg dataset...", flush=True)
    try:
        tg_path = os.path.join(project_root, 'data/Tg_SMILES_class_pid_polyinfo_median.csv')
        if os.path.exists(tg_path):
            tg_df = pd.read_csv(tg_path)
            tg_new_rows = []
            for _, row in tg_df.iterrows():
                tg_new_rows.append({
                    'SMILES': row['SMILES'],
                    'Tg': row.get('Tg', np.nan),
                    'FFV': np.nan, 'Tc': np.nan,
                    'Density': np.nan, 'Rg': np.nan
                })
            if tg_new_rows:
                train_df = pd.concat([train_df, pd.DataFrame(tg_new_rows)], ignore_index=True)
                print(f"   ✅ Added {len(tg_new_rows)} Tg samples")
        else:
            print(f"   ⚠️  Tg dataset not found")
    except Exception as e:
        print(f"   ⚠️  Tg augmentation failed: {str(e)[:50]}")
    
    # Augment with PI1070 (Density + Rg)
    print("→ Augmenting with PI1070 dataset...", flush=True)
    try:
        pi_path = os.path.join(project_root, 'data/PI1070.csv')
        if os.path.exists(pi_path):
            pi_df = pd.read_csv(pi_path)
            pi_count = 0
            for _, row in pi_df.iterrows():
                if pd.notna(row.get('density')) or pd.notna(row.get('Rg')):
                    train_df = pd.concat([train_df, pd.DataFrame([{
                        'SMILES': row['smiles'],
                        'Tg': np.nan, 'FFV': np.nan, 'Tc': np.nan,
                        'Density': row.get('density', np.nan),
                        'Rg': row.get('Rg', np.nan)
                    }])], ignore_index=True)
                    pi_count += 1
            if pi_count > 0:
                print(f"   ✅ Added {pi_count} PI1070 samples")
        else:
            print(f"   ⚠️  PI1070 dataset not found")
    except Exception as e:
        print(f"   ⚠️  PI1070 augmentation failed: {str(e)[:50]}")
    
    # Augment with LAMALAB Tg
    print("→ Augmenting with LAMALAB Tg dataset...", flush=True)
    try:
        lama_path = os.path.join(project_root, 'data/LAMALAB_CURATED_Tg_structured_polymerclass.csv')
        if os.path.exists(lama_path):
            lama_df = pd.read_csv(lama_path)
            lama_count = 0
            for _, row in lama_df.iterrows():
                tg_value = row.get('labels.Exp_Tg(K)', np.nan)
                if pd.notna(tg_value):
                    tg_celsius = tg_value - 273.15
                    train_df = pd.concat([train_df, pd.DataFrame([{
                        'SMILES': row['PSMILES'],
                        'Tg': tg_celsius,
                        'FFV': np.nan, 'Tc': np.nan, 'Density': np.nan, 'Rg': np.nan
                    }])], ignore_index=True)
                    lama_count += 1
            if lama_count > 0:
                print(f"   ✅ Added {lama_count} LAMALAB samples")
        else:
            print(f"   ⚠️  LAMALAB dataset not found")
    except Exception as e:
        print(f"   ⚠️  LAMALAB augmentation failed: {str(e)[:50]}")
    
    # Augment with pseudo-labels
    print("→ Augmenting with pseudo-labeled dataset...", flush=True)
    try:
        pseudo_paths = [
            os.path.join(project_root, 'pseudolabel/pi1m_pseudolabels_ensemble_3models.csv'),
            os.path.join(project_root, 'pseudolabel/pi1m_pseudolabels_ensemble_2models.csv'),
            os.path.join(project_root, 'data/PI1M_50000_v2.1.csv'),  # Main pseudo-label file
        ]
        pseudo_path = None
        for p in pseudo_paths:
            if os.path.exists(p):
                pseudo_path = p
                break
        
        if pseudo_path:
            pseudo_df = pd.read_csv(pseudo_path)
            train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
            print(f"   ✅ Added {len(pseudo_df)} pseudo-labeled samples")
        else:
            print(f"   ⚠️  Pseudo-labeled dataset not found")
    except Exception as e:
        print(f"   ⚠️  Pseudo-label augmentation failed: {str(e)[:50]}")
    
    train_df = train_df.reset_index(drop=True)
    print(f"\n📊 Total training data after augmentation: {len(train_df)} samples\n", flush=True)
    return train_df, target_cols


def train_single_property(target, target_data, feature_names, output_dir, time_limit):
    """Train AutoGluon model for a single property (for parallel execution)"""
    try:
        print(f"🔧 Training model for {target}...", flush=True)
        
        if len(target_data) == 0:
            print(f"   ⚠️  No training data available for {target}", flush=True)
            return target, None
        
        print(f"   Training samples: {len(target_data)}", flush=True)
        print(f"   Input features: {len(feature_names)}", flush=True)
        print(f"   Time limit: {time_limit}s", flush=True)
        print(f"   AutoGluon preset: good_quality", flush=True)
        
        model_path = str(output_dir / target)
        predictor = TabularPredictor(
            label=target,
            path=model_path,
            eval_metric='mae',
            problem_type='regression'
        )
        
        print(f"   → Fitting model...", flush=True)
        predictor.fit(
            target_data,
            time_limit=time_limit,
            presets='good_quality',
            verbosity=0
        )
        
        # Get selected features
        try:
            selected = predictor.features() if callable(predictor.features) else predictor.features
            n_selected = len(selected)
        except:
            n_selected = len(feature_names)
        
        reduction = 100 * (1 - n_selected / len(feature_names))
        print(f"   ✅ Model trained successfully!", flush=True)
        print(f"      Selected features: {n_selected}/{len(feature_names)} ({reduction:.1f}% reduction)", flush=True)
        print(f"      Model saved to: {model_path}\n", flush=True)
        
        return target, {
            'selected_features': n_selected,
            'model_path': model_path
        }
        
    except Exception as e:
        print(f"   ❌ Training failed: {str(e)[:100]}", flush=True)
        print(f"   Status: ERROR\n", flush=True)
        return target, None


def train_config(config_key, time_limit=300):
    """Train models for a specific configuration"""
    
    config = CONFIGURATIONS[config_key]
    print(f"\n{'='*70}")
    print(f"Configuration {config_key}: {config['description']}")
    print(f"{'='*70}\n")
    
    # Get project root
    project_root = get_project_root()
    
    # Load and augment data
    train_df, target_cols = load_and_augment_data(project_root)
    
    # Extract features
    print("="*70, flush=True)
    print("STEP 2: Extracting features...", flush=True)
    print("="*70, flush=True)
    print(f"Configuration {config_key} feature breakdown:", flush=True)
    if config['simple']:
        print("  ✅ Simple features (10): SMILES length, element counts, rings, bonds", flush=True)
    if config['hand_crafted']:
        print("  ✅ Hand-crafted features (11): Polymer-specific domain knowledge", flush=True)
    if config['rdkit_descriptors']:
        print(f"  ✅ RDKit descriptors ({len(config['rdkit_descriptors'])}): Chemistry-based molecular features", flush=True)
    print(flush=True)
    
    features_df = extract_features_for_config(train_df, config_key)
    print(f"✅ Extracted {len(features_df.columns)} total features\n", flush=True)
    
    # Create training data
    train_data = pd.concat([train_df[['SMILES'] + target_cols], features_df], axis=1)
    train_data = train_data.dropna(subset=['SMILES']).reset_index(drop=True)
    
    print("="*70, flush=True)
    print("STEP 3: Preparing training data...", flush=True)
    print("="*70, flush=True)
    print(f"Training data shape: {train_data.shape[0]} samples × {train_data.shape[1]} columns\n", flush=True)
    
    print("Target property availability:", flush=True)
    for col in target_cols:
        n_avail = train_data[col].notna().sum()
        pct = 100 * n_avail / len(train_data)
        print(f"  {col:10s}: {n_avail:6d} samples ({pct:5.1f}%)", flush=True)
    print(flush=True)
    
    # Output directory
    output_dir = Path('/content/autogluon_results' if IN_COLAB else 'models/autogluon_results') / config_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train AutoGluon for each target
    results = {
        'config': config_key,
        'description': config['description'],
        'input_features': len(features_df.columns),
        'models': {}
    }
    
    feature_names = features_df.columns.tolist()
    
    print("="*70, flush=True)
    print("STEP 4: Training AutoGluon models (PARALLEL)...", flush=True)
    print("="*70 + "\n", flush=True)
    print(f"Starting {len(target_cols)} property models in parallel using {min(5, len(target_cols))} workers...\n", flush=True)
    
    # Prepare data for all targets
    target_data_dict = {}
    for target in target_cols:
        target_data = train_data[feature_names + [target]].dropna(subset=[target])
        target_data_dict[target] = target_data
    
    # Train in parallel
    with ThreadPoolExecutor(max_workers=min(5, len(target_cols))) as executor:
        futures = {}
        for target in target_cols:
            if len(target_data_dict[target]) == 0:
                print(f"⚠️  No training data for {target}, skipping\n", flush=True)
                continue
            
            future = executor.submit(
                train_single_property,
                target,
                target_data_dict[target],
                feature_names,
                output_dir,
                time_limit
            )
            futures[future] = target
        
        # Collect results as they complete
        for future in as_completed(futures):
            target, model_info = future.result()
            if model_info is not None:
                results['models'][target] = model_info
    
    # Save results
    results_file = output_dir / 'config_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Configuration: {config_key} - {config['description']}")
    print(f"Input features: {results['input_features']}")
    print(f"Models trained: {len(results['models'])}/{len(target_cols)}")
    for target, model_info in results['models'].items():
        print(f"  ✅ {target}: {model_info['selected_features']} features selected")
    print(f"\nResults saved to: {output_dir}")
    print("="*70 + "\n")
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Systematic feature analysis with AutoGluon on Colab")
    parser.add_argument('--config', type=str, choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       default='C', help='Feature configuration to train')
    parser.add_argument('--time_limit', type=int, default=300, help='Training time limit per target (seconds)')
    parser.add_argument('--all', action='store_true', help='Train all configurations (A-H)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SYSTEMATIC FEATURE ANALYSIS - COLAB VERSION")
    print("="*70 + "\n")
    
    if args.all:
        print("Training all configurations A-H...\n")
        all_results = {}
        for config_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            result = train_config(config_key, args.time_limit)
            all_results[config_key] = result
        
        # Save summary
        summary_file = Path('/content/all_results.json' if IN_COLAB else 'all_results.json')
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ All configurations trained! Summary saved to {summary_file}")
    else:
        print(f"Training configuration {args.config}...\n")
        train_config(args.config, args.time_limit)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)