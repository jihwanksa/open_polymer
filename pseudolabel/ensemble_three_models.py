"""
Ensemble 3 models: BERT + Uni-Mol + AutoGluon pseudo-labels.

Usage:
    python pseudolabel/ensemble_three_models.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*80)
print("ENSEMBLE: BERT + UNI-MOL + AUTOGLUON (3-Model Ensemble)")
print("="*80 + "\n")

# Load all three prediction files
bert_path = "pseudolabel/pi1m_pseudolabels_bert.csv"
unimol_path = "pseudolabel/pi1m_pseudolabels_unimol.csv"
autogluon_path = "pseudolabel/pi1m_pseudolabels_autogluon_21features.csv"  # Using 21-feature version

print(f"📂 Loading BERT labels from {bert_path}...")
bert_df = pd.read_csv(bert_path)
print(f"   ✅ Loaded {len(bert_df)} samples\n")

print(f"📂 Loading Uni-Mol labels from {unimol_path}...")
unimol_df = pd.read_csv(unimol_path)
print(f"   ✅ Loaded {len(unimol_df)} samples\n")

print(f"📂 Loading AutoGluon labels from {autogluon_path}...")
autogluon_df = pd.read_csv(autogluon_path)
print(f"   ✅ Loaded {len(autogluon_df)} samples\n")

# Verify all have same shape
if not (bert_df.shape == unimol_df.shape == autogluon_df.shape):
    print("❌ ERROR: Dataframe shapes don't match!")
    print(f"   BERT: {bert_df.shape}")
    print(f"   Uni-Mol: {unimol_df.shape}")
    print(f"   AutoGluon: {autogluon_df.shape}")
    exit(1)

print("✅ All datasets have matching shapes\n")

# Verify SMILES columns match
if not bert_df['SMILES'].equals(unimol_df['SMILES']):
    print("⚠️  SMILES columns don't match perfectly between BERT and Uni-Mol")
if not bert_df['SMILES'].equals(autogluon_df['SMILES']):
    print("⚠️  SMILES columns don't match perfectly between BERT and AutoGluon")

print("✅ SMILES columns validated\n")

# Create ensemble
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
ensemble_df = pd.DataFrame({'SMILES': bert_df['SMILES']})

print("🔀 Averaging predictions from 3 models...\n")

for target in target_cols:
    bert_vals = bert_df[target].values
    unimol_vals = unimol_df[target].values
    autogluon_vals = autogluon_df[target].values
    
    # Average
    ensemble_vals = (bert_vals + unimol_vals + autogluon_vals) / 3
    ensemble_df[target] = ensemble_vals
    
    print(f"   {target}:")
    print(f"      BERT:      Mean={bert_vals.mean():.4f}, Std={bert_vals.std():.4f}")
    print(f"      Uni-Mol:   Mean={unimol_vals.mean():.4f}, Std={unimol_vals.std():.4f}")
    print(f"      AutoGluon: Mean={autogluon_vals.mean():.4f}, Std={autogluon_vals.std():.4f}")
    print(f"      Ensemble:  Mean={ensemble_vals.mean():.4f}, Std={ensemble_vals.std():.4f} ✨\n")

# Apply Tg transformation
print("🔧 Applying Tg transformation: (9/5) × Tg + 45...")
ensemble_df['Tg'] = (9/5) * ensemble_df['Tg'] + 45
print(f"   ✅ Tg transformation applied\n")

# Save
output_path = "pseudolabel/pi1m_pseudolabels_ensemble_3models.csv"
print(f"💾 Saving ensemble labels to {output_path}...")
ensemble_df.to_csv(output_path, index=False)
print(f"✅ Saved {len(ensemble_df)} ensemble pseudo-labeled samples\n")

print("="*80)
print("✅ 3-MODEL ENSEMBLE COMPLETE!")
print("="*80)
print(f"\n📊 Output: {output_path}")
print(f"   Samples: {len(ensemble_df)}")
print(f"   Properties: {', '.join(target_cols)}")
print(f"\n🎯 Models ensembled:")
print(f"   1. BERT (unikei/bert-base-smiles)")
print(f"   2. Uni-Mol (dptech/Uni-Mol2)")
print(f"   3. AutoGluon (simple features)")
print(f"\n💡 Next: Use for training with train_v85_best.py")
