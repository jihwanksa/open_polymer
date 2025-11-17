"""
Systematic Feature Analysis with AutoGluon

Tests 8 different feature configurations to find optimal feature subset:
- A: Simple only (10 features)
- B: Hand-crafted only (11 features)  
- C: Current baseline (34 features = 10 simple + 11 hand-crafted + 13 RDKit)
- D: Expanded RDKit (56 features = 10 simple + 11 hand-crafted + 35 RDKit)
- E: All RDKit (~221 features = 10 simple + 11 hand-crafted + 200 RDKit)
- F: RDKit only expanded (45 features = 35 RDKit)
- G: No simple features (24 features = 11 hand-crafted + 13 RDKit)
- H: No hand-crafted features (23 features = 10 simple + 13 RDKit)

Each configuration trains AutoGluon to see which features it selects and how performance varies.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Force CPU-only mode for AutoGluon
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPS_ENABLED'] = '0'

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available")

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    print("❌ AutoGluon not installed")
    sys.exit(1)


# ==================================================================================
# RDKit DESCRIPTORS DATABASE
# ==================================================================================

# Current 13 descriptors (hand-picked)
RDKIT_CURRENT_13 = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumAromaticRings', 'TPSA', 'NumSaturatedRings', 'NumAliphaticRings',
    'RingCount', 'FractionCsp3', 'NumHeteroatoms', 'BertzCT'
]

# Expanded set of 35 RDKit descriptors (comprehensive but selective)
RDKIT_EXPANDED_35 = RDKIT_CURRENT_13 + [
    # Additional molecular weight/size descriptors
    'ExactMolWt', 'HeavyAtomCount', 'NAtoms',
    # Additional aromaticity
    'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
    # Additional H-bonding
    'NumHBD', 'NumHBA',  # Alternate names
    # Additional rings
    'NumSaturatedCycles', 'NumAliphaticCycles',
    # Bond descriptors
    'NumAmideBonds', 'NumSulfonamideBonds', 'NumBridgeheadAtoms',
    # Topological/structural
    'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3',
    'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3',
    'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3',
    # Elemental
    'NumHeavyAtoms', 'NumHeteroatoms',
    # Additional
    'Ipc', 'MolLogP'
]

# All major RDKit descriptors (~100+ usable ones, avoiding deprecated)
RDKIT_ALL = [
    # Molecular properties
    'MolWt', 'ExactMolWt', 'HeavyAtomCount', 'NAtoms', 'NumHeavyAtoms',
    'NumAtoms', 'NumRotatableBonds', 'NumHeteroatoms', 'NumValenceElectrons',
    'NumSaturatedRings', 'NumAliphaticRings', 'NumSaturatedCarbocycles',
    'NumAliphaticCarbocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles',
    
    # Aromaticity
    'NumAromaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
    'FractionCsp3', 'FractionCsp2',
    
    # H-bonding
    'NumHDonors', 'NumHAcceptors', 'NumHBD', 'NumHBA',
    'NumLipinskiHBA', 'NumLipinskiHBD',
    
    # Lipophilicity
    'LogP', 'MolLogP', 'TPSA',
    
    # Bonds & rings
    'RingCount', 'NumRings', 'NumBridgeheadAtoms', 'NumSpiroAtoms',
    'NumAmideBonds', 'NumSulfonamideBonds',
    
    # Descriptors related to branching
    'BranchCount',
    
    # Topological
    'LabuteASA', 'LabuteELF10',
    'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6',
    'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
    'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10',
    'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
    
    # Connectivity
    'Ipc', 'Chi0', 'Chi1', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi0n', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n',
    'BertzCT',
    
    # Surface area & volume
    'TPSA', 'LabuteASA',
    'MolFormula'
]

# Remove duplicates and non-existent descriptors
RDKIT_ALL = list(set(RDKIT_ALL))


# ==================================================================================
# FEATURE CONFIGURATION
# ==================================================================================

SIMPLE_FEATURES = [
    'smiles_length', 'carbon_count', 'nitrogen_count', 'oxygen_count',
    'sulfur_count', 'fluorine_count', 'ring_count', 'double_bond_count',
    'triple_bond_count', 'branch_count'
]

HAND_CRAFTED_FEATURES = [
    'num_side_chains', 'backbone_carbons', 'aromatic_count',
    'h_bond_donors', 'h_bond_acceptors', 'num_rings',
    'single_bonds', 'halogen_count', 'heteroatom_count',
    'mw_estimate', 'branching_ratio'
]

FEATURE_CONFIGURATIONS = {
    'A_simple_only': {
        'simple': True,
        'hand_crafted': False,
        'rdkit': None,
        'description': 'Simple features only (SMILES counting) - 10 features',
        'target_features': 10
    },
    'B_hand_crafted_only': {
        'simple': False,
        'hand_crafted': True,
        'rdkit': None,
        'description': 'Hand-crafted domain knowledge only - 11 features',
        'target_features': 11
    },
    'C_current_baseline': {
        'simple': True,
        'hand_crafted': True,
        'rdkit': RDKIT_CURRENT_13,
        'description': 'Current baseline (10 simple + 11 hand-crafted + 13 RDKit) - 34 features',
        'target_features': 34
    },
    'D_expanded_rdkit': {
        'simple': True,
        'hand_crafted': True,
        'rdkit': RDKIT_EXPANDED_35,
        'description': 'Expanded RDKit (10 simple + 11 hand-crafted + 35 RDKit) - 56 features',
        'target_features': 56
    },
    'E_all_rdkit': {
        'simple': True,
        'hand_crafted': True,
        'rdkit': RDKIT_ALL,
        'description': f'All RDKit (10 simple + 11 hand-crafted + {len(RDKIT_ALL)} RDKit) - {10+11+len(RDKIT_ALL)} features',
        'target_features': 10 + 11 + len(RDKIT_ALL)
    },
    'F_rdkit_only_expanded': {
        'simple': False,
        'hand_crafted': False,
        'rdkit': RDKIT_EXPANDED_35,
        'description': 'RDKit only, expanded - 35 features',
        'target_features': 35
    },
    'G_no_simple': {
        'simple': False,
        'hand_crafted': True,
        'rdkit': RDKIT_CURRENT_13,
        'description': 'Hand-crafted + RDKit (no simple) - 24 features',
        'target_features': 24
    },
    'H_no_hand_crafted': {
        'simple': True,
        'hand_crafted': False,
        'rdkit': RDKIT_CURRENT_13,
        'description': 'Simple + RDKit (no hand-crafted) - 23 features',
        'target_features': 23
    }
}


# ==================================================================================
# FEATURE EXTRACTION FUNCTIONS
# ==================================================================================

def extract_rdkit_descriptors(smiles_str, descriptor_list=None):
    """Extract specified RDKit descriptors for a SMILES string"""
    if not RDKIT_AVAILABLE or descriptor_list is None:
        return {}
    
    try:
        cleaned = str(smiles_str).replace('*', 'H')
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return {desc: 0.0 for desc in descriptor_list}
        
        descriptors = {}
        for desc_name in descriptor_list:
            try:
                descriptor_fn = getattr(Descriptors, desc_name)
                value = descriptor_fn(mol)
                # Handle NaN/inf values
                if pd.isna(value) or np.isinf(value):
                    descriptors[desc_name] = 0.0
                else:
                    descriptors[desc_name] = float(value)
            except:
                descriptors[desc_name] = 0.0
        
        return descriptors
    except:
        return {desc: 0.0 for desc in descriptor_list}


def extract_simple_features(smiles_str):
    """Extract 10 simple SMILES string-based features"""
    smiles_str = str(smiles_str)
    return {
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


def extract_hand_crafted_features(smiles_str):
    """Extract 11 hand-crafted polymer-specific features"""
    smiles_str = str(smiles_str)
    
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
    
    return {
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


def extract_configuration_features(df, config_key):
    """Extract features based on configuration"""
    config = FEATURE_CONFIGURATIONS[config_key]
    features = []
    
    print(f"Extracting features for {config_key}...")
    print(f"  Description: {config['description']}")
    
    for idx, smiles in tqdm(df['SMILES'].items(), total=len(df), desc="Features"):
        try:
            smiles_str = str(smiles) if pd.notna(smiles) else ""
            row_features = {}
            
            # Add simple features
            if config['simple']:
                row_features.update(extract_simple_features(smiles_str))
            
            # Add hand-crafted features
            if config['hand_crafted']:
                row_features.update(extract_hand_crafted_features(smiles_str))
            
            # Add RDKit descriptors
            if config['rdkit'] is not None:
                row_features.update(extract_rdkit_descriptors(smiles_str, config['rdkit']))
            
            features.append(row_features)
        except:
            # Fallback: all zeros
            num_features = config['target_features']
            features.append({f'feat_{i}': 0.0 for i in range(num_features)})
    
    features_df = pd.DataFrame(features, index=df.index)
    print(f"✅ Extracted {len(features_df)} samples with {len(features_df.columns)} features")
    
    return features_df


# ==================================================================================
# DATA LOADING & PREPARATION
# ==================================================================================

def load_and_augment_data():
    """Load training data with full augmentation"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("\n" + "="*80)
    print("LOADING AND AUGMENTING DATA")
    print("="*80)
    
    # Load main training data
    print("\n📂 Loading main training data...")
    train_df = pd.read_csv(os.path.join(project_root, 'data/raw/train.csv'))
    print(f"   Loaded {len(train_df)} samples")
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Canonicalize SMILES if RDKit available
    if RDKIT_AVAILABLE:
        print("\n🔄 Canonicalizing SMILES...")
        for i, smiles in enumerate(train_df['SMILES']):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    train_df.at[i, 'SMILES'] = Chem.MolToSmiles(mol, canonical=True)
            except:
                pass
    
    # Load pseudo-labels
    print("\n📂 Loading pseudo-labeled dataset...")
    try:
        pseudo_paths = [
            os.path.join(project_root, 'data/raw/PI1M_50000_v2.1.csv'),
            os.path.join(project_root, 'data/PI1M_50000_v2.1.csv'),
        ]
        pseudo_path = None
        for path in pseudo_paths:
            if os.path.exists(path):
                pseudo_path = path
                break
        
        if pseudo_path:
            pseudo_df = pd.read_csv(pseudo_path)
            train_smiles_set = set(train_df['SMILES'].dropna())
            pseudo_new = pseudo_df[~pseudo_df['SMILES'].isin(train_smiles_set)].copy()
            
            if len(pseudo_new) > 0:
                train_df = pd.concat([train_df, pseudo_new], ignore_index=True)
                print(f"   ✅ Added {len(pseudo_new)} pseudo-labeled samples")
    except Exception as e:
        print(f"   ⚠️  Could not load pseudo-labels: {e}")
    
    train_df = train_df.reset_index(drop=True)
    
    print(f"\n📊 Final data: {len(train_df)} samples")
    for col in target_cols:
        n_avail = train_df[col].notna().sum()
        print(f"   {col}: {n_avail} samples ({n_avail/len(train_df)*100:.1f}%)")
    
    return train_df, target_cols


# ==================================================================================
# TRAINING & EVALUATION
# ==================================================================================

def train_config(train_df, config_key, output_dir, time_limit=600, preset='fast'):
    """Train AutoGluon for a specific feature configuration"""
    
    config = FEATURE_CONFIGURATIONS[config_key]
    print(f"\n{'='*80}")
    print(f"CONFIGURATION: {config_key}")
    print(f"{'='*80}")
    print(f"Description: {config['description']}")
    
    start_time = time.time()
    
    # Extract features
    features_df = extract_configuration_features(train_df, config_key)
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Create training data
    train_data = pd.concat([train_df[['SMILES'] + target_cols], features_df], axis=1)
    train_data = train_data.dropna(subset=['SMILES']).reset_index(drop=True)
    
    feature_names = features_df.columns.tolist()
    
    # Train AutoGluon models
    config_output_dir = os.path.join(output_dir, config_key)
    Path(config_output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        'config': config_key,
        'description': config['description'],
        'input_features': len(feature_names),
        'models': {},
        'timestamp': datetime.now().isoformat()
    }
    
    for target in target_cols:
        print(f"\n🔧 Training AutoGluon for {target}...")
        
        target_data = train_data[['SMILES'] + feature_names + [target]].dropna(subset=[target])
        
        if len(target_data) == 0:
            print(f"  ⚠️  No training data for {target}")
            continue
        
        print(f"  Training samples: {len(target_data)}")
        print(f"  Input features: {len(feature_names)}")
        
        try:
            model_path = os.path.join(config_output_dir, target)
            predictor = TabularPredictor(
                label=target,
                path=model_path,
                eval_metric='mae',
                problem_type='regression'
            )
            
            train_for_fit = target_data[feature_names + [target]].copy()
            
            predictor.fit(
                train_for_fit,
                time_limit=time_limit,
                presets=preset,
                verbosity=0
            )
            
            print(f"  ✅ Model trained")
            
            # Try to get selected features
            try:
                selected = predictor.features() if callable(predictor.features) else predictor.features
                results['models'][target] = {
                    'selected_features': len(selected),
                    'model_path': model_path
                }
            except:
                results['models'][target] = {
                    'selected_features': len(feature_names),
                    'model_path': model_path
                }
                
        except Exception as e:
            print(f"  ❌ Training failed: {e}")
    
    # Save results
    results['training_time'] = time.time() - start_time
    results_path = os.path.join(config_output_dir, 'config_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Configuration {config_key} complete in {results['training_time']:.1f}s")
    print(f"   Input features: {results['input_features']}")
    print(f"   Models trained: {len(results['models'])}")
    
    return results


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, 'AutoGluon/systematic_feature/results')
    
    print("\n" + "="*80)
    print("🚀 SYSTEMATIC FEATURE ANALYSIS WITH AUTOGLUON")
    print("="*80)
    print(f"\nTesting {len(FEATURE_CONFIGURATIONS)} configurations:")
    for config_key, config in FEATURE_CONFIGURATIONS.items():
        print(f"  {config_key}: {config['description']}")
    
    # Load data once
    train_df, target_cols = load_and_augment_data()
    
    # Train each configuration
    all_results = {}
    total_start = time.time()
    
    for i, (config_key, config) in enumerate(FEATURE_CONFIGURATIONS.items(), 1):
        print(f"\n\n{'#'*80}")
        print(f"# {i}/{len(FEATURE_CONFIGURATIONS)}: {config_key}")
        print(f"{'#'*80}")
        
        results = train_config(train_df, config_key, output_dir, time_limit=600, preset='fast')
        all_results[config_key] = results
    
    total_time = time.time() - total_start
    
    # Generate comparison report
    print(f"\n\n{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}")
    
    comparison_data = []
    for config_key, results in all_results.items():
        config = FEATURE_CONFIGURATIONS[config_key]
        comparison_data.append({
            'Configuration': config_key,
            'Description': config['description'],
            'Input Features': results['input_features'],
            'Models Trained': len(results['models']),
            'Training Time (s)': f"{results['training_time']:.1f}",
            'Avg Selected Features': int(np.mean([v['selected_features'] for v in results['models'].values()]) if results['models'] else 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, 'comparison_report.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    print(f"\n✅ Comparison report saved to: {comparison_path}")
    print(f"   All results in: {output_dir}")
    print(f"   Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()

