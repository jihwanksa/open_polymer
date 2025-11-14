"""
Simplified pseudo-label generation script that works without numpy/pandas in memory.

This version processes data in streaming fashion to avoid memory issues
and can work with or without heavy dependencies.

Usage:
    python pseudolabel/generate_pseudolabels_simple.py --help
"""

import csv
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Iterator
import json

def extract_features_from_smiles(smiles: str) -> Dict[str, float]:
    """
    Extract 21 chemistry-based features from SMILES string.
    Uses only string operations - no numpy/pandas needed.
    """
    smiles_str = str(smiles).strip()
    
    if not smiles_str:
        return None
    
    try:
        # Basic counts (10 features)
        basic = {
            'smiles_length': float(len(smiles_str)),
            'carbon_count': float(smiles_str.count('C')),
            'nitrogen_count': float(smiles_str.count('N')),
            'oxygen_count': float(smiles_str.count('O')),
            'sulfur_count': float(smiles_str.count('S')),
            'fluorine_count': float(smiles_str.count('F')),
            'ring_count': float(smiles_str.count('c') + smiles_str.count('C1')),
            'double_bond_count': float(smiles_str.count('=')),
            'triple_bond_count': float(smiles_str.count('#')),
            'branch_count': float(smiles_str.count('(')),
        }
        
        # Chemistry-based features (11 additional)
        num_side_chains = float(smiles_str.count('('))
        backbone_carbons = float(smiles_str.count('C') - smiles_str.count('C('))
        aromatic_count = float(smiles_str.count('c'))
        h_bond_donors = float(smiles_str.count('O') + smiles_str.count('N'))
        h_bond_acceptors = float(smiles_str.count('O') + smiles_str.count('N'))
        num_rings = float(smiles_str.count('1') + smiles_str.count('2'))
        single_bonds = float(len(smiles_str) - smiles_str.count('=') - smiles_str.count('#') - aromatic_count)
        halogen_count = float(smiles_str.count('F') + smiles_str.count('Cl') + smiles_str.count('Br'))
        heteroatom_count = float(smiles_str.count('N') + smiles_str.count('O') + smiles_str.count('S'))
        mw_estimate = float(
            smiles_str.count('C') * 12 + smiles_str.count('O') * 16 +
            smiles_str.count('N') * 14 + smiles_str.count('S') * 32 + smiles_str.count('F') * 19
        )
        branching_ratio = num_side_chains / max(backbone_carbons, 1.0)
        
        # Combine all features (21 total)
        features = {
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
        
        return features
        
    except Exception as e:
        print(f"Warning: Failed to extract features from SMILES '{smiles}': {e}", file=sys.stderr)
        return None


def read_smiles_csv(input_path: str) -> Iterator[tuple]:
    """
    Read SMILES from CSV file in streaming fashion.
    Yields (row_number, smiles) tuples.
    """
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if 'SMILES' in row:
                yield i + 1, row['SMILES']
            else:
                print(f"Error: 'SMILES' column not found. Available columns: {list(row.keys())}")
                sys.exit(1)


def generate_dummy_predictions(n_samples: int) -> List[List[float]]:
    """
    Generate dummy predictions for testing.
    In real use, these would come from BERT, AutoGluon, Uni-Mol models.
    
    Returns: List of [Tg, FFV, Tc, Density, Rg] predictions
    """
    import random
    random.seed(42)
    
    predictions = []
    for _ in range(n_samples):
        # Realistic ranges based on data
        tg = random.uniform(0, 400)  # Tg range
        ffv = random.uniform(0.25, 0.5)  # FFV range
        tc = random.uniform(0.05, 0.35)  # Tc range
        density = random.uniform(0.8, 1.4)  # Density range
        rg = random.uniform(5, 30)  # Rg range
        
        predictions.append([tg, ffv, tc, density, rg])
    
    return predictions


def apply_tg_transformation(predictions: List[List[float]]) -> List[List[float]]:
    """Apply Tg transformation: (9/5) × Tg + 45"""
    transformed = []
    for pred in predictions:
        tg_transformed = (9/5) * pred[0] + 45
        transformed.append([tg_transformed, pred[1], pred[2], pred[3], pred[4]])
    return transformed


def save_pseudolabels_csv(smiles_list: List[str], predictions: List[List[float]], 
                         output_path: str, target_names: List[str]):
    """Save pseudo-labels to CSV file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving pseudo-labels to {output_path}...")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SMILES'] + target_names)
        writer.writeheader()
        
        for smiles, pred in zip(smiles_list, predictions):
            row = {'SMILES': smiles}
            for i, target in enumerate(target_names):
                row[target] = f"{pred[i]:.6f}"
            writer.writerow(row)
    
    print(f"   ✅ Saved {len(smiles_list)} pseudo-labeled samples")
    print(f"   Output: {output_path}")


def main(args):
    print("\n" + "="*80)
    print("PSEUDO-LABEL GENERATION (Simplified Version)")
    print("="*80)
    print("\nThis simplified version:")
    print("1. Reads SMILES from CSV without loading full dataset into memory")
    print("2. Extracts features using only string operations")
    print("3. Generates dummy predictions (replace with real models)")
    print("4. Saves pseudo-labels for training")
    
    target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Check input file
    if not Path(args.input_data).exists():
        print(f"❌ Input file not found: {args.input_data}")
        sys.exit(1)
    
    print(f"\n📂 Reading SMILES from {args.input_data}...")
    
    # Read SMILES and extract features in streaming fashion
    smiles_list = []
    features_list = []
    
    for row_num, smiles in read_smiles_csv(args.input_data):
        if row_num % 1000 == 0:
            print(f"   Processing row {row_num}...", end='\r')
        
        features = extract_features_from_smiles(smiles)
        if features is not None:
            smiles_list.append(smiles)
            features_list.append(features)
    
    print(f"\n   ✅ Extracted features for {len(smiles_list)} SMILES")
    
    # Generate predictions (dummy for now)
    print(f"\n🔮 Generating pseudo-label predictions...")
    print(f"   ⚠️  Using dummy predictions (replace with real BERT/AutoGluon/Uni-Mol)")
    
    predictions = generate_dummy_predictions(len(smiles_list))
    
    # Apply Tg transformation (optional)
    if args.apply_tg_transform:
        print(f"\n🔧 Applying Tg transformation: (9/5) × Tg + 45...")
        predictions = apply_tg_transformation(predictions)
    
    # Save results
    save_pseudolabels_csv(smiles_list, predictions, args.output_path, target_names)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ PSEUDO-LABEL GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📋 Summary:")
    print(f"   Input: {args.input_data}")
    print(f"   Samples processed: {len(smiles_list)}")
    print(f"   Output: {args.output_path}")
    print(f"\n📝 Next steps:")
    print(f"   1. Integrate real BERT predictions")
    print(f"   2. Integrate real AutoGluon predictions")
    print(f"   3. Integrate real Uni-Mol predictions")
    print(f"   4. Use setup_pretrained_models.py to download models")
    print(f"   5. Update generate_pseudolabels.py with actual model predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels (simplified version without heavy dependencies)"
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
        default="pseudolabel/pi1m_pseudolabels_dummy.csv",
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

