"""
Use AutoGluon models trained with systematic feature analysis.

Supports inference with models from train_for_colab_serial.py configurations:
- A: Simple features only (10)
- B: Hand-crafted only (11)
- C: Current baseline (34: 10 simple + 11 hand-crafted + 13 RDKit)
- D: Expanded RDKit (56: 10 + 11 + 35)
- E: All RDKit (~81: 10 + 11 + ~60)
- F: RDKit expanded only (35)
- G: No simple features (24: 11 hand-crafted + 13 RDKit)
- H: No hand-crafted features (23: 10 simple + 13 RDKit)

Usage in Colab:
    %run /content/open_polymer/AutoGluon/systematic_feature/inference.py --config G
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import pickle
import warnings
import argparse
warnings.filterwarnings('ignore')

# Detect Colab environment
try:
    from google.colab import drive
    IN_COLAB = True
    print("✅ Running in Google Colab", flush=True)
except ImportError:
    IN_COLAB = False
    print("Running locally", flush=True)

# Force CPU-only mode for local (avoids MPS hanging on Apple Silicon)
if IN_COLAB:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MPS_ENABLED'] = '0'

# Try to import RDKit for SMILES canonicalization and descriptors
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available - will use fallback features")


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


class AutoGluonModel:
    """
    AutoGluon model wrapper for systematic feature analysis inference.
    
    Loads pre-trained models from train_for_colab_serial.py configurations (A-H).
    Models are stored in: /content/autogluon_results/{CONFIG}/{TARGET}
    """
    
    def __init__(self, model_dir=None, config='C'):
        """
        Args:
            model_dir: Path to model directory (auto-detected if None)
            config: Configuration key (A-H)
        """
        self.config = config
        self.target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.predictors = {}
        
        if model_dir is None:
            if IN_COLAB:
                self.model_dir = f"/content/autogluon_results/{config}"
            else:
                self.model_dir = f"models/autogluon_results/{config}"
        else:
            self.model_dir = model_dir
    
    def load(self):
        """Load pre-trained AutoGluon models for a specific configuration"""
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            print("❌ AutoGluon not installed")
            return False
        
        print("\n" + "="*70)
        print(f"LOADING AUTOGLUON MODELS (Configuration {self.config})")
        print("="*70)
        print(f"Model directory: {self.model_dir}\n", flush=True)
        
        all_loaded = True
        for target in self.target_cols:
            target_path = os.path.join(self.model_dir, target)
            
            try:
                if not os.path.exists(target_path):
                    print(f"⚠️  {target}: Path not found: {target_path}")
                    all_loaded = False
                    continue
                    
                print(f"📂 Loading {target}...", end=" ", flush=True)
                predictor = TabularPredictor.load(target_path)
                self.predictors[target] = predictor
                print(f"✅", flush=True)
                
                # Get feature names (might be method or property depending on AutoGluon version)
                try:
                    features = predictor.features() if callable(predictor.features) else predictor.features
                    print(f"   Features: {len(features)}", flush=True)
                except:
                    print(f"   Features: N/A", flush=True)
                
            except Exception as e:
                print(f"❌ Failed: {str(e)[:100]}", flush=True)
                all_loaded = False
        
        if all_loaded:
            print("\n" + "="*70)
            print(f"✅ ALL MODELS FOR CONFIG {self.config} LOADED!")
            print("="*70 + "\n", flush=True)
        
        return all_loaded
    
    def predict(self, X_test, target_names, all_features_df=None):
        """Generate predictions for all targets"""
        predictions = np.zeros((len(X_test), len(target_names)))
        
        for i, target in enumerate(target_names):
            try:
                if target in self.predictors:
                    predictor = self.predictors[target]
                    
                    # Get feature names the model expects
                    try:
                        expected_features = predictor.features() if callable(predictor.features) else predictor.features
                    except:
                        expected_features = None
                    
                    # If we have a DataFrame with all features, select only the expected ones
                    if all_features_df is not None and expected_features is not None:
                        X_df = all_features_df[[f for f in expected_features if f in all_features_df.columns]].copy()
                    else:
                        # Fallback: Handle NaN/inf and convert to DataFrame
                        X_clean = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
                        if expected_features is None:
                            expected_features = [f'feat_{j}' for j in range(X_clean.shape[1])]
                        X_df = pd.DataFrame(X_clean, columns=expected_features[:X_clean.shape[1]])
                    
                    # Predict
                    preds = predictor.predict(X_df)
                    if isinstance(preds, (pd.Series, pd.DataFrame)):
                        preds = preds.values.flatten()
                    
                    predictions[:, i] = preds
                    print(f"✅ Predicted {target}: range [{preds.min():.4f}, {preds.max():.4f}]")
                else:
                    print(f"⚠️  No model for {target}, using zeros")
                    predictions[:, i] = 0.0
                    
            except Exception as e:
                print(f"❌ Prediction failed for {target}: {e}")
                predictions[:, i] = 0.0
        
        return predictions


# ============================================================================
# RDKit Descriptors and Feature Configurations (from train_for_colab_serial.py)
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

CONFIGURATIONS = {
    'A': {
        'name': 'A_simple_only',
        'simple': True,
        'hand_crafted': False,
        'rdkit': []
    },
    'B': {
        'name': 'B_hand_crafted_only',
        'simple': False,
        'hand_crafted': True,
        'rdkit': []
    },
    'C': {
        'name': 'C_current_baseline',
        'simple': True,
        'hand_crafted': True,
        'rdkit': RDKIT_CURRENT_13
    },
    'D': {
        'name': 'D_expanded_rdkit',
        'simple': True,
        'hand_crafted': True,
        'rdkit': RDKIT_EXPANDED_35
    },
    'E': {
        'name': 'E_all_rdkit',
        'simple': True,
        'hand_crafted': True,
        'rdkit': RDKIT_ALL_DESCRIPTORS
    },
    'F': {
        'name': 'F_rdkit_only_expanded',
        'simple': False,
        'hand_crafted': False,
        'rdkit': RDKIT_EXPANDED_35
    },
    'G': {
        'name': 'G_no_simple',
        'simple': False,
        'hand_crafted': True,
        'rdkit': RDKIT_CURRENT_13
    },
    'H': {
        'name': 'H_no_hand_crafted',
        'simple': True,
        'hand_crafted': False,
        'rdkit': RDKIT_CURRENT_13
    },
}


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


def extract_rdkit_descriptors(smiles_str, descriptor_names):
    """Extract specific RDKit descriptors"""
    if not RDKIT_AVAILABLE or not descriptor_names:
        return {}
    
    try:
        from rdkit.Chem import Crippen
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


def extract_features_for_config(df, config_key):
    """Extract features based on configuration"""
    if config_key not in CONFIGURATIONS:
        print(f"⚠️  Config {config_key} not recognized, using C (baseline)")
        config_key = 'C'
    
    config = CONFIGURATIONS[config_key]
    features = []
    
    print(f"Extracting features for config {config_key} ({config['name']})...")
    for idx, smiles in tqdm(df['SMILES'].items(), total=len(df)):
        try:
            smiles_str = str(smiles) if pd.notna(smiles) else ""
            row = {}
            
            if config['simple']:
                row.update(extract_simple_features(smiles_str))
            if config['hand_crafted']:
                row.update(extract_hand_crafted_features(smiles_str))
            if config['rdkit']:
                row.update(extract_rdkit_descriptors(smiles_str, config['rdkit']))
            
            features.append(row)
        except:
            # Fallback: all zeros
            num_features = 0
            if config['simple']: num_features += 10
            if config['hand_crafted']: num_features += 11
            if config['rdkit']: num_features += len(config['rdkit'])
            
            features.append({f'feat_{i}': 0.0 for i in range(num_features)})
    
    return pd.DataFrame(features)


def extract_comprehensive_features(df, config_key='C'):
    """Extract features based on configuration"""
    features = extract_features_for_config(df, config_key)
    print(f"✅ Created {len(features)} feature vectors with {len(features.columns)} features\n", flush=True)
    return features


def extract_comprehensive_features_old(df):
    """Extract 34 comprehensive features: 10 simple + 11 chemistry + 13 RDKit descriptors (LEGACY)"""
    features = []
    
    print("Extracting 34 comprehensive features (10 simple + 11 chemistry + 13 RDKit)...")
    for idx, smiles in tqdm(df['SMILES'].items(), total=len(df)):
        try:
            smiles_str = str(smiles) if pd.notna(smiles) else ""
            
            # 10 simple string-based features
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
            
            # 11 chemistry-based features
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
            
            chemistry = {
                'num_side_chains': num_side_chains,
                'backbone_carbons': backbone_carbons,
                'branching_ratio': branching_ratio,
                'aromatic_count': aromatic_count,
                'h_bond_donors': h_bond_donors,
                'h_bond_acceptors': h_bond_acceptors,
                'num_rings': num_rings,
                'single_bonds': single_bonds,
                'halogen_count': halogen_count,
                'heteroatom_count': heteroatom_count,
                'mw_estimate': mw_estimate,
            }
            
            # 13 RDKit descriptors
            rdkit_desc = {}
            if RDKIT_AVAILABLE and Chem is not None:
                try:
                    mol = Chem.MolFromSmiles(smiles_str)
                    if mol is not None:
                        rdkit_desc = {
                            'MolWt': Descriptors.MolWt(mol),
                            'LogP': Descriptors.MolLogP(mol),
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
                        for k, v in rdkit_desc.items():
                            if pd.isna(v) or np.isinf(v):
                                rdkit_desc[k] = 0.0
                except:
                    rdkit_desc = {k: 0.0 for k in ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                                                   'NumRotatableBonds', 'NumAromaticRings', 'TPSA',
                                                   'NumSaturatedRings', 'NumAliphaticRings', 'RingCount',
                                                   'FractionCsp3', 'NumHeteroatoms', 'BertzCT']}
            else:
                rdkit_desc = {k: 0.0 for k in ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                                               'NumRotatableBonds', 'NumAromaticRings', 'TPSA',
                                               'NumSaturatedRings', 'NumAliphaticRings', 'RingCount',
                                               'FractionCsp3', 'NumHeteroatoms', 'BertzCT']}
            
            # Combine all 34 features
            features.append({**basic, **chemistry, **rdkit_desc})
        except:
            # Fallback: all zeros (34 features)
            features.append({
                'smiles_length': 0, 'carbon_count': 0, 'nitrogen_count': 0, 'oxygen_count': 0,
                'sulfur_count': 0, 'fluorine_count': 0, 'ring_count': 0, 'double_bond_count': 0,
                'triple_bond_count': 0, 'branch_count': 0, 'num_side_chains': 0, 'backbone_carbons': 0,
                'branching_ratio': 0, 'aromatic_count': 0, 'h_bond_donors': 0, 'h_bond_acceptors': 0,
                'num_rings': 0, 'single_bonds': 0, 'halogen_count': 0, 'heteroatom_count': 0,
                'mw_estimate': 0, 'MolWt': 0, 'LogP': 0, 'NumHDonors': 0, 'NumHAcceptors': 0,
                'NumRotatableBonds': 0, 'NumAromaticRings': 0, 'TPSA': 0, 'NumSaturatedRings': 0,
                'NumAliphaticRings': 0, 'RingCount': 0, 'FractionCsp3': 0, 'NumHeteroatoms': 0, 'BertzCT': 0
            })
    
    features_df = pd.DataFrame(features, index=df.index)
    print(f"✅ Created {len(features_df)} feature vectors with {len(features_df.columns)} features")
    return features_df


def load_and_augment_data():
    """Load and augment training data (v85 strategy)"""
    print("\n" + "="*80)
    print("LOADING AND AUGMENTING DATA (v85 Configuration - 1st Place Solution!)")
    print("="*80)
    
    # Get project root - handle both Colab and local paths
    if IN_COLAB:
        # In Colab, detect project root from common paths
        paths = [
            '/content/open_polymer',
            '/content/drive/MyDrive/open_polymer',
            '/root/open_polymer'
        ]
        project_root = None
        for p in paths:
            if os.path.exists(os.path.join(p, 'data/raw/train.csv')):
                project_root = p
                break
        if project_root is None:
            raise FileNotFoundError("Project root not found in Colab paths")
    else:
        # Local: inference.py is in AutoGluon/systematic_feature/, go up 3 levels
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
    # Load main training data
    print("\n📂 Loading main training data...")
    train_df = pd.read_csv(os.path.join(project_root, 'data/raw/train.csv'))
    print(f"   Loaded {len(train_df)} samples")
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Canonicalize SMILES
    if RDKIT_AVAILABLE:
        print("\n🔄 Canonicalizing SMILES...")
        train_df['SMILES'] = train_df['SMILES'].apply(make_smile_canonical)
        train_df['SMILES'] = train_df['SMILES'].fillna(train_df['SMILES'].bfill())
        print(f"   ✅ SMILES canonicalization complete!")
    
    # Augment with external Tc data
    print("\n📂 Loading external Tc data...")
    try:
        tc_external = pd.read_csv(os.path.join(project_root, 'data/Tc_SMILES.csv'))
        tc_external = tc_external.rename(columns={'TC_mean': 'Tc'})
        
        train_smiles = set(train_df['SMILES'])
        tc_new = tc_external[~tc_external['SMILES'].isin(train_smiles)].copy()
        
        if len(tc_new) > 0:
            tc_new_rows = []
            for _, row in tc_new.iterrows():
                tc_new_rows.append({
                    'SMILES': row['SMILES'],
                    'Tg': np.nan,
                    'FFV': np.nan,
                    'Tc': row['Tc'],
                    'Density': np.nan,
                    'Rg': np.nan
                })
            train_df = pd.concat([train_df, pd.DataFrame(tc_new_rows)], ignore_index=True)
            print(f"   ✅ Added {len(tc_new)} Tc samples")
    except Exception as e:
        print(f"   ⚠️  Tc data not found: {e}")
    
    # Augment with external Tg data
    print("\n📂 Loading external Tg data...")
    try:
        tg_external = pd.read_csv(os.path.join(project_root, 'data/Tg_SMILES_class_pid_polyinfo_median.csv'))
        
        train_smiles = set(train_df['SMILES'])
        tg_new = tg_external[~tg_external['SMILES'].isin(train_smiles)].copy()
        
        if len(tg_new) > 0:
            tg_new_rows = []
            for _, row in tg_new.iterrows():
                tg_new_rows.append({
                    'SMILES': row['SMILES'],
                    'Tg': row['Tg'],
                    'FFV': np.nan,
                    'Tc': np.nan,
                    'Density': np.nan,
                    'Rg': np.nan
                })
            train_df = pd.concat([train_df, pd.DataFrame(tg_new_rows)], ignore_index=True)
            print(f"   ✅ Added {len(tg_new)} Tg samples")
    except Exception as e:
        print(f"   ⚠️  Tg data not found: {e}")
    
    # Augment with PI1070 (Density + Rg) and LAMALAB (Tg)
    print("\n📂 Loading additional external datasets...")
    try:
        train_smiles_set = set(train_df['SMILES'])
        
        # PI1070 (Density + Rg) - has 'density' and 'Rg' columns
        try:
            pi1070_df = pd.read_csv(os.path.join(project_root, 'data/PI1070.csv'))
            pi1070_subset = pi1070_df[['smiles', 'density', 'Rg']].copy()
            pi1070_subset = pi1070_subset.rename(columns={'smiles': 'SMILES'})
            pi1070_new = pi1070_subset[~pi1070_subset['SMILES'].isin(train_smiles_set)].copy()
            
            for idx, row in pi1070_new.iterrows():
                if pd.notna(row['density']) or pd.notna(row['Rg']):
                    train_df = pd.concat([train_df, pd.DataFrame([{
                        'SMILES': row['SMILES'],
                        'Tg': np.nan,
                        'FFV': np.nan,
                        'Tc': np.nan,
                        'Density': row['density'] if pd.notna(row['density']) else np.nan,
                        'Rg': row['Rg'] if pd.notna(row['Rg']) else np.nan
                    }])], ignore_index=True)
            
            print(f"   ✅ Added {len(pi1070_new)} PI1070 samples")
        except Exception as e:
            print(f"   ⚠️  PI1070 loading failed: {e}")
        
        # LAMALAB (Tg)
        try:
            lamalab_df = pd.read_csv(os.path.join(project_root, 'data/LAMALAB_CURATED_Tg_structured_polymerclass.csv'))
            lamalab_subset = lamalab_df[['PSMILES', 'labels.Exp_Tg(K)']].copy()
            lamalab_subset = lamalab_subset.rename(columns={'PSMILES': 'SMILES', 'labels.Exp_Tg(K)': 'Tg'})
            lamalab_subset['Tg'] = lamalab_subset['Tg'] - 273.15  # Convert K to C
            
            lamalab_new = lamalab_subset[~lamalab_subset['SMILES'].isin(train_smiles_set)].copy()
            lamalab_new_valid = lamalab_new[lamalab_new['Tg'].notna()].copy()
            
            for idx, row in lamalab_new_valid.iterrows():
                train_df = pd.concat([train_df, pd.DataFrame([{
                    'SMILES': row['SMILES'],
                    'Tg': row['Tg'],
                    'FFV': np.nan,
                    'Tc': np.nan,
                    'Density': np.nan,
                    'Rg': np.nan
                }])], ignore_index=True)
            
            print(f"   ✅ Added {len(lamalab_new_valid)} LAMALAB Tg samples")
        except Exception as e:
            print(f"   ⚠️  LAMALAB loading failed: {e}")
            
    except Exception as e:
        print(f"   ⚠️  Additional datasets error: {e}")
    
    train_df = train_df.reset_index(drop=True)
    
    # Load and augment with pseudo-labeled dataset (v85 NEW!)
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
            print(f"   ✅ Loaded {len(pseudo_df)} pseudo-labeled samples")
            
            # Check for overlaps and add only new samples
            train_smiles_set = set(train_df['SMILES'].dropna())
            pseudo_new = pseudo_df[~pseudo_df['SMILES'].isin(train_smiles_set)].copy()
            
            if len(pseudo_new) > 0:
                # Canonicalize pseudo-label SMILES if RDKit available
                if RDKIT_AVAILABLE:
                    pseudo_new['SMILES'] = pseudo_new['SMILES'].apply(make_smile_canonical)
                    pseudo_new['SMILES'] = pseudo_new['SMILES'].fillna(pseudo_new['SMILES'].bfill())
                
                train_df = pd.concat([train_df, pseudo_new], ignore_index=True)
                print(f"   ✅ Added {len(pseudo_new)} pseudo-labeled samples")
                print(f"      Total increase: +{len(pseudo_new)/len(train_df)*100:.1f}%")
        else:
            print("   ⚠️  Pseudo-labeled dataset not found (optional)")
    except Exception as e:
        print(f"   ⚠️  Could not load pseudo-labeled data: {e}")
    
    train_df = train_df.reset_index(drop=True)
    
    print(f"\n📊 Final training data: {len(train_df)} samples")
    for col in target_cols:
        n_avail = train_df[col].notna().sum()
        print(f"   {col}: {n_avail} samples ({n_avail/len(train_df)*100:.1f}%)")
    
    return train_df, target_cols


def get_project_root():
    """Get project root path based on environment"""
    if IN_COLAB:
        # Try common Colab paths
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
        # Local: inference.py is in AutoGluon/systematic_feature/, go up 3 levels to project root
        current_file = os.path.abspath(__file__)
        # AutoGluon/systematic_feature/inference.py -> AutoGluon/systematic_feature -> AutoGluon -> project_root
        return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))


def main(config='C'):
    """Main function - use pre-trained AutoGluon models for systematic feature analysis"""
    print("\n" + "="*80)
    print(f"🤖 AUTOGLUON INFERENCE (Configuration {config})")
    print("="*80)
    print(f"Using models from train_for_colab_serial.py (config {config})")
    print("Environment: " + ("Google Colab" if IN_COLAB else "Local"))
    print("="*80 + "\n")
    
    project_root = get_project_root()
    print(f"📂 Project root: {project_root}\n", flush=True)
    
    # Load pre-trained AutoGluon models
    print("="*80)
    print("STEP 1: LOAD PRE-TRAINED AUTOGLUON MODELS")
    print("="*80)
    
    model = AutoGluonModel(config=config)
    if not model.load():
        print("❌ Failed to load AutoGluon models")
        return
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Load and augment data
    print("\n" + "="*80)
    print("STEP 2: LOAD AND AUGMENT DATA")
    print("="*80)
    
    train_df, _ = load_and_augment_data()
    
    # Create features for this configuration
    print("\n" + "="*80)
    print(f"STEP 3: EXTRACT FEATURES FOR CONFIGURATION {config}")
    print("="*80 + "\n")
    
    train_features = extract_comprehensive_features(train_df, config_key=config)
    
    # Generate predictions using AutoGluon models
    print("="*80)
    print("STEP 4: GENERATE PREDICTIONS WITH AUTOGLUON MODELS")
    print("="*80 + "\n")
    
    # Pass features DataFrame so AutoGluon can select only the ones it needs
    train_predictions = model.predict(train_features.values, target_cols, all_features_df=train_features)
    
    # Save predictions
    print("\n" + "="*80)
    print("STEP 5: SAVE RESULTS")
    print("="*80 + "\n")
    
    results_df = pd.DataFrame(train_predictions, columns=target_cols)
    
    # Save to CSV (configuration-specific)
    if IN_COLAB:
        output_path = f"/content/inference_results_config_{config}.csv"
    else:
        output_path = os.path.join(project_root, f"inference_results_config_{config}.csv")
    
    results_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}", flush=True)
    print(f"\nPrediction statistics:", flush=True)
    print(results_df.describe())
    
    print("\n" + "="*80)
    print(f"✅ INFERENCE COMPLETE (Config {config})!")
    print("="*80)
    print(f"\nFeatures: {train_features.shape[1]}")
    print(f"Training samples: {len(train_df)}")
    print(f"Targets: {', '.join(target_cols)}")
    print(f"Configuration: {config} ({CONFIGURATIONS[config]['name']})")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGluon Inference for Systematic Feature Analysis")
    parser.add_argument('--config', type=str, choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       default='C', help='Configuration key (A-H)')
    
    args = parser.parse_args()
    main(config=args.config)

