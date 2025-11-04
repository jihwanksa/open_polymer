#!/usr/bin/env python3
"""
Local test script for 3D feature extraction
Tests performance and correctness before pushing to Kaggle
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# Test if RDKit is available
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
    print("✓ RDKit is available")
except ImportError:
    print("✗ RDKit not available - please install: pip install rdkit")
    exit(1)

# ============================================================================
# 3D Feature Extraction Functions
# ============================================================================

def smiles_to_3d_features(smiles: str, timeout=10):
    """
    Generate 3D descriptors from SMILES string.
    
    Replaces radicals (*) with carbon (C) for embedding.
    Returns NaN if molecule is invalid or if embedding fails.
    """
    # Replace radicals with carbon for embedding purposes
    smiles_clean = str(smiles).replace("*", "C")
    
    # Skip if still contains radicals or invalid
    if not isinstance(smiles_clean, str) or "*" in smiles_clean:
        return {
            "Asphericity": np.nan,
            "Eccentricity": np.nan,
            "InertialShapeFactor": np.nan,
            "NPR1": np.nan,
            "NPR2": np.nan,
            "SpherocityIndex": np.nan
        }
    
    try:
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            raise ValueError("Invalid molecule")
        
        # Add hydrogens for proper 3D embedding
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates with timeout handling
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != -1:
            # Optimize geometry with UFF force field
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except:
                pass  # Optimization may fail for some molecules, but embedding succeeded
        
        # Calculate 3D descriptors
        feats = {
            "Asphericity": rdMolDescriptors.CalcAsphericity(mol),
            "Eccentricity": rdMolDescriptors.CalcEccentricity(mol),
            "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor(mol),
            "NPR1": rdMolDescriptors.CalcNPR1(mol),
            "NPR2": rdMolDescriptors.CalcNPR2(mol),
            "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex(mol)
        }
        return feats
        
    except Exception as e:
        # Return NaN on any error
        return {
            "Asphericity": np.nan,
            "Eccentricity": np.nan,
            "InertialShapeFactor": np.nan,
            "NPR1": np.nan,
            "NPR2": np.nan,
            "SpherocityIndex": np.nan
        }


def add_3d_features(df, smiles_col="SMILES", sample_size=None):
    """
    Apply 3D feature extraction to DataFrame with progress bar.
    
    If sample_size is set, only process that many molecules (for testing).
    """
    if sample_size:
        df_sample = df.head(sample_size)
        print(f"Processing sample of {len(df_sample)} molecules (total: {len(df)})")
    else:
        df_sample = df
    
    print(f"Extracting 3D features from {len(df_sample)} molecules...")
    start = time.time()
    
    # Use tqdm for progress bar
    features_list = []
    for smiles in tqdm(df_sample[smiles_col], desc="3D Features", unit="mol"):
        feats = smiles_to_3d_features(smiles)
        features_list.append(feats)
    
    features_df = pd.DataFrame(features_list, index=df_sample.index)
    
    # Ensure float dtype
    features_df = features_df.astype('float64')
    
    elapsed = time.time() - start
    
    # Stats
    n_valid = features_df['Asphericity'].notna().sum()
    n_invalid = features_df['Asphericity'].isna().sum()
    
    print(f"\n✓ Extracted 3D features in {elapsed:.2f}s")
    print(f"  Valid molecules: {n_valid} / {len(df_sample)}")
    print(f"  Invalid/Failed: {n_invalid} / {len(df_sample)}")
    print(f"  Rate: {len(df_sample) / elapsed:.1f} molecules/sec")
    print(f"  Features: {list(features_df.columns)}")
    
    result = pd.concat([df_sample, features_df], axis=1)
    return result, elapsed


# ============================================================================
# Main Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("3D FEATURE EXTRACTION TEST")
    print("=" * 70)
    
    # Load Tc data
    print("\n[1] Loading Tc dataset...")
    tc_df = pd.read_csv('data/Tc_SMILES.csv')
    print(f"✓ Loaded {len(tc_df)} Tc samples")
    print(f"  Columns: {list(tc_df.columns)}")
    print(f"  First SMILES: {tc_df.iloc[0, 0]}")
    
    # Load Tg data
    print("\n[2] Loading Tg dataset...")
    tg_df = pd.read_csv('data/Tg_SMILES_class_pid_polyinfo_median.csv')
    print(f"✓ Loaded {len(tg_df)} Tg samples")
    print(f"  Columns: {list(tg_df.columns)}")
    
    # Test on small sample first
    print("\n[3] Testing on small sample (first 50 molecules)...")
    tc_sample, elapsed_tc = add_3d_features(tc_df, smiles_col='SMILES', sample_size=50)
    
    # Test on larger sample
    print("\n[4] Testing on medium sample (first 200 molecules)...")
    tc_medium, elapsed_tc_med = add_3d_features(tc_df, smiles_col='SMILES', sample_size=200)
    
    # Estimate full time
    full_estimate = (elapsed_tc_med / 200) * len(tc_df)
    print(f"\n[5] Full dataset time estimate:")
    print(f"  {len(tc_df)} molecules @ {200/elapsed_tc_med:.1f} mol/sec")
    print(f"  Estimated time: {full_estimate:.1f} seconds ({full_estimate/60:.1f} minutes)")
    
    # Show sample output
    print(f"\n[6] Sample output (first 3 rows):")
    output_cols = ['SMILES', 'Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'SpherocityIndex']
    available_cols = [col for col in output_cols if col in tc_sample.columns]
    print(tc_sample[available_cols].head(3))
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

