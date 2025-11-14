"""Ensemble BERT and Uni-Mol pseudo-labels by averaging.

This script combines pseudo-labels generated from BERT and Uni-Mol models
by taking the mean of their predictions for each property.

Usage:
    python pseudolabel/ensemble_bert_unimol.py \
        --bert_labels pseudolabel/pi1m_pseudolabels_bert.csv \
        --unimol_labels pseudolabel/pi1m_pseudolabels_unimol.csv \
        --output_path pseudolabel/pi1m_pseudolabels_ensemble_2models.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def main(args):
    print("\n" + "=" * 80)
    print("ENSEMBLE PSEUDO-LABELS: BERT + UNI-MOL")
    print("=" * 80 + "\n")
    
    # Load both label sets
    print(f"📂 Loading BERT labels from {args.bert_labels}...")
    if not Path(args.bert_labels).exists():
        print(f"❌ File not found: {args.bert_labels}")
        return
    bert_df = pd.read_csv(args.bert_labels)
    print(f"   ✅ Loaded {len(bert_df)} samples\n")
    
    print(f"📂 Loading Uni-Mol labels from {args.unimol_labels}...")
    if not Path(args.unimol_labels).exists():
        print(f"❌ File not found: {args.unimol_labels}")
        return
    unimol_df = pd.read_csv(args.unimol_labels)
    print(f"   ✅ Loaded {len(unimol_df)} samples\n")
    
    # Verify same SMILES
    if len(bert_df) != len(unimol_df):
        print(f"⚠️  Warning: Different number of samples ({len(bert_df)} vs {len(unimol_df)})")
    
    if not bert_df['SMILES'].equals(unimol_df['SMILES']):
        print("⚠️  Warning: SMILES don't match perfectly, will align by position")
    
    # Create ensemble
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    ensemble_df = pd.DataFrame({'SMILES': bert_df['SMILES']})
    
    print("🔀 Averaging predictions...\n")
    for prop in properties:
        if prop in bert_df.columns and prop in unimol_df.columns:
            # Average BERT and Uni-Mol
            bert_vals = bert_df[prop].values
            unimol_vals = unimol_df[prop].values
            
            ensemble_vals = np.nanmean(
                np.array([bert_vals, unimol_vals]),
                axis=0
            )
            
            ensemble_df[prop] = ensemble_vals
            
            print(f"   {prop}:")
            print(f"      BERT:    Mean={np.nanmean(bert_vals):.4f}, Std={np.nanstd(bert_vals):.4f}")
            print(f"      Uni-Mol: Mean={np.nanmean(unimol_vals):.4f}, Std={np.nanstd(unimol_vals):.4f}")
            print(f"      Ensemble: Mean={np.nanmean(ensemble_vals):.4f}, Std={np.nanstd(ensemble_vals):.4f}\n")
        else:
            print(f"   ⚠️  Property {prop} missing in one of the models")
            ensemble_df[prop] = np.nan
    
    # Save
    print(f"💾 Saving ensemble labels to {args.output_path}...")
    ensemble_df.to_csv(args.output_path, index=False)
    print(f"✅ Saved {len(ensemble_df)} ensemble pseudo-labeled samples\n")
    
    print("=" * 80)
    print("✅ ENSEMBLE COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Output: {args.output_path}")
    print(f"   Model 1: BERT")
    print(f"   Model 2: Uni-Mol")
    print(f"   Ensemble: Average")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble BERT and Uni-Mol pseudo-labels.")
    parser.add_argument("--bert_labels", type=str, default="pseudolabel/pi1m_pseudolabels_bert.csv",
                        help="Path to BERT pseudo-labels")
    parser.add_argument("--unimol_labels", type=str, default="pseudolabel/pi1m_pseudolabels_unimol.csv",
                        help="Path to Uni-Mol pseudo-labels")
    parser.add_argument("--output_path", type=str, default="pseudolabel/pi1m_pseudolabels_ensemble_2models.csv",
                        help="Path to save ensemble labels")
    args = parser.parse_args()
    main(args)

