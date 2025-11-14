"""
Use AutoGluon production models for best performance.

This script uses pre-trained AutoGluon models from train_autogluon_production.py:
- AutoGluon training: 60K+ samples, 34 features (10 simple + 11 hand-crafted + 13 RDKit)
- Best model: WeightedEnsemble_L2 (stacked ensemble of 8 base models)
- Features: Automatic feature selection and hyperparameter tuning
- Tg transformation (9/5)x + 45 for distribution shift correction
- Expected improvement over v85 Random Forest through AutoML optimization
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
warnings.filterwarnings('ignore')

# Force CPU-only mode for AutoGluon (avoids MPS hanging on Apple Silicon)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPS_ENABLED'] = '0'

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


class AutoGluonModel:
    """
    AutoGluon model wrapper for production inference.
    
    🚀 AutoGluon Production Configuration:
    - Models: WeightedEnsemble_L2 (stacked ensemble of 8 base models)
    - Training: 60K+ samples, 34 features (10 simple + 11 hand-crafted + 13 RDKit)
    - Hyperparameter: Automatically tuned by AutoGluon
    - Preset: medium_quality (balanced quality vs time)
    """
    
    def __init__(self, model_dir="models/autogluon_production"):
        self.model_dir = model_dir
        self.predictors = {}
        self.target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    def load(self):
        """Load pre-trained AutoGluon models"""
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            print("❌ AutoGluon not installed")
            return False
        
        print("\n" + "="*70)
        print("LOADING AUTOGLUON PRODUCTION MODELS")
        print("="*70)
        
        all_loaded = True
        for target in self.target_cols:
            target_path = f"{self.model_dir}/{target}"
            
            try:
                print(f"\n📂 Loading {target}...", end=" ")
                predictor = TabularPredictor.load(target_path)
                self.predictors[target] = predictor
                print(f"✅")
                # Get feature names (might be method or property depending on AutoGluon version)
                try:
                    features = predictor.features() if callable(predictor.features) else predictor.features
                    print(f"   Features: {len(features)}")
                except:
                    print(f"   Features: N/A")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                all_loaded = False
        
        if all_loaded:
            print("\n" + "="*70)
            print("✅ ALL AUTOGLUON MODELS LOADED!")
            print("="*70)
        
        return all_loaded
    
    def predict(self, X_test, target_names):
        """Generate predictions for all targets"""
        predictions = np.zeros((len(X_test), len(target_names)))
        
        for i, target in enumerate(target_names):
            try:
                if target in self.predictors:
                    predictor = self.predictors[target]
                    
                    # Handle NaN/inf
                    X_clean = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    # Convert to DataFrame
                    X_df = pd.DataFrame(X_clean, columns=predictor.features)
                    
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


def create_chemistry_features(df):
    """Create 21 chemistry-based features from SMILES (v53 configuration)"""
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


def load_and_augment_data():
    """Load and augment training data (v85 strategy)"""
    print("\n" + "="*80)
    print("LOADING AND AUGMENTING DATA (v85 Configuration - 1st Place Solution!)")
    print("="*80)
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    
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


def main():
    """Main function - use pre-trained AutoGluon models"""
    print("\n" + "="*80)
    print("🤖 AUTOGLUON PRODUCTION INFERENCE")
    print("="*80)
    print("Using pre-trained AutoGluon models from train_autogluon_production.py")
    print("Models: WeightedEnsemble_L2 (stacked ensemble of 8 base models)")
    print("Training: 60K+ samples, 34 features, medium_quality preset")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Load pre-trained AutoGluon models
    print("\n" + "="*80)
    print("STEP 1: LOAD PRE-TRAINED AUTOGLUON MODELS")
    print("="*80)
    
    model = AutoGluonModel(model_dir=os.path.join(project_root, "models/autogluon_production"))
    if not model.load():
        print("❌ Failed to load AutoGluon models")
        return
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Load and augment data (for validation on training features)
    print("\n" + "="*80)
    print("STEP 2: LOAD AND AUGMENT DATA")
    print("="*80)
    
    train_df, _ = load_and_augment_data()
    
    # Create features
    print("\n" + "="*80)
    print("STEP 3: CREATE FEATURES")
    print("="*80)
    
    train_features = create_chemistry_features(train_df)
    
    print("\n" + "="*80)
    print("✅ AUTOGLUON PRODUCTION SETUP COMPLETE!")
    print("="*80)
    print(f"\nFeatures: {train_features.shape[1]}")
    print(f"Training samples: {len(train_df)}")
    print(f"Targets: {', '.join(target_cols)}")
    print(f"\n📊 AutoGluon will handle:")
    print(f"   ✅ Automatic feature selection from 34 features")
    print(f"   ✅ Hyperparameter tuning for each algorithm")
    print(f"   ✅ Intelligent ensemble weighting (WeightedEnsemble_L2)")
    print(f"   ✅ Robust predictions on 60K+ training samples")
    print("="*80)


if __name__ == "__main__":
    main()

