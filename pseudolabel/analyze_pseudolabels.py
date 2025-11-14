"""
Analyze and compare generated pseudo-labels with reference dataset.

This script helps validate the quality of generated pseudo-labels by comparing
them with the reference pseudo-label dataset (from 1st place solution's ensemble).
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_datasets(generated_path: str, reference_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both generated and reference pseudo-label datasets"""
    print(f"📂 Loading generated pseudo-labels from {generated_path}...")
    generated = pd.read_csv(generated_path)
    print(f"   ✅ Loaded {len(generated)} samples")
    
    print(f"📂 Loading reference pseudo-labels from {reference_path}...")
    reference = pd.read_csv(reference_path)
    print(f"   ✅ Loaded {len(reference)} samples")
    
    return generated, reference


def align_datasets(generated: pd.DataFrame, reference: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align datasets by SMILES to ensure we're comparing the same molecules"""
    print("\n🔄 Aligning datasets by SMILES...")
    
    # Merge on SMILES
    merged = generated.merge(
        reference[['SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']],
        on='SMILES',
        suffixes=('_gen', '_ref'),
        how='inner'
    )
    
    print(f"   ✅ Aligned {len(merged)} samples with matching SMILES")
    
    return merged


def compute_statistics(merged: pd.DataFrame, target_names: list) -> Dict:
    """Compute comparison statistics for each property"""
    stats = {}
    
    for target in target_names:
        gen_col = f'{target}_gen'
        ref_col = f'{target}_ref'
        
        if gen_col not in merged.columns or ref_col not in merged.columns:
            continue
        
        gen = merged[gen_col].values
        ref = merged[ref_col].values
        
        # Filter out NaN values
        mask = ~(np.isnan(gen) | np.isnan(ref))
        gen = gen[mask]
        ref = ref[mask]
        
        diff = gen - ref
        
        # Compute metrics
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        mape = np.mean(np.abs(diff / (np.abs(ref) + 1e-6)))
        correlation = np.corrcoef(gen, ref)[0, 1] if len(gen) > 1 else np.nan
        
        stats[target] = {
            'n_samples': len(gen),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'gen_mean': gen.mean(),
            'ref_mean': ref.mean(),
            'gen_std': gen.std(),
            'ref_std': ref.std(),
            'gen_min': gen.min(),
            'gen_max': gen.max(),
            'ref_min': ref.min(),
            'ref_max': ref.max(),
        }
    
    return stats


def print_statistics(stats: Dict):
    """Print comparison statistics in formatted table"""
    print("\n" + "="*100)
    print("PSEUDO-LABEL COMPARISON STATISTICS")
    print("="*100)
    
    print(f"\n{'Property':<10} {'Samples':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<10} {'Correlation':<12}")
    print("-" * 100)
    
    for target, s in stats.items():
        print(f"{target:<10} {s['n_samples']:<10} {s['mae']:>10.4f}  {s['rmse']:>10.4f}  "
              f"{s['mape']:>8.1%}  {s['correlation']:>10.3f}")
    
    print("\n" + "="*100)
    print("DETAILED STATISTICS PER PROPERTY")
    print("="*100)
    
    for target, s in stats.items():
        print(f"\n📊 {target}:")
        print(f"   Samples: {s['n_samples']}")
        print(f"   Generated: mean={s['gen_mean']:>10.4f}, std={s['gen_std']:>8.4f}, "
              f"range=[{s['gen_min']:>8.2f}, {s['gen_max']:>8.2f}]")
        print(f"   Reference: mean={s['ref_mean']:>10.4f}, std={s['ref_std']:>8.4f}, "
              f"range=[{s['ref_min']:>8.2f}, {s['ref_max']:>8.2f}]")
        print(f"   Error:     MAE={s['mae']:>10.4f}, RMSE={s['rmse']:>8.4f}, MAPE={s['mape']:>8.1%}")
        print(f"   Correlation: {s['correlation']:>8.3f}")


def plot_comparison(merged: pd.DataFrame, target_names: list, output_dir: str = "pseudolabel"):
    """Generate comparison plots"""
    try:
        import matplotlib.pyplot as plt
        
        print(f"\n📈 Generating comparison plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, target in enumerate(target_names[:6]):
            gen_col = f'{target}_gen'
            ref_col = f'{target}_ref'
            
            if gen_col not in merged.columns or ref_col not in merged.columns:
                continue
            
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(merged[ref_col], merged[gen_col], alpha=0.3, s=1)
            
            # Perfect prediction line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
            
            ax.set_xlabel('Reference')
            ax.set_ylabel('Generated')
            ax.set_title(f'{target}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'pseudolabel_comparison.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"   ✅ Saved plot to {output_path}")
        plt.close()
        
    except ImportError:
        print("   ⚠️  Matplotlib not installed, skipping plots")


def identify_outliers(merged: pd.DataFrame, target_names: list, threshold: float = 2.0) -> Dict:
    """Identify outlier predictions (difference > threshold * std)"""
    outliers = {}
    
    print(f"\n🔍 Identifying outliers (threshold: {threshold} std)...")
    
    for target in target_names:
        gen_col = f'{target}_gen'
        ref_col = f'{target}_ref'
        
        if gen_col not in merged.columns or ref_col not in merged.columns:
            continue
        
        diff = merged[gen_col] - merged[ref_col]
        std = diff.std()
        mean = diff.mean()
        
        outlier_mask = np.abs(diff - mean) > threshold * std
        outlier_indices = merged.index[outlier_mask]
        
        outliers[target] = {
            'n_outliers': outlier_mask.sum(),
            'pct_outliers': outlier_mask.sum() / len(merged) * 100,
            'examples': merged.iloc[outlier_indices.tolist()[:5]].to_dict('records')
        }
    
    return outliers


def print_outliers(outliers: Dict):
    """Print outlier analysis"""
    print("\n" + "="*100)
    print("OUTLIER ANALYSIS")
    print("="*100)
    
    for target, data in outliers.items():
        print(f"\n{target}:")
        print(f"   Outliers: {data['n_outliers']} ({data['pct_outliers']:.2f}%)")
        
        if data['examples']:
            print(f"   Example outliers:")
            for i, example in enumerate(data['examples'][:3]):
                smiles = example.get('SMILES', 'N/A')[:50]
                print(f"      {i+1}. SMILES: {smiles}...")


def save_report(stats: Dict, outliers: Dict, output_path: str):
    """Save analysis report to file"""
    print(f"\n💾 Saving report to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("PSEUDO-LABEL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Statistics
        f.write("STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Property':<10} {'Samples':<10} {'MAE':<12} {'RMSE':<12} {'Correlation':<12}\n")
        f.write("-"*80 + "\n")
        
        for target, s in stats.items():
            f.write(f"{target:<10} {s['n_samples']:<10} {s['mae']:>10.4f}  {s['rmse']:>10.4f}  "
                   f"{s['correlation']:>10.3f}\n")
        
        # Outliers
        f.write("\n\nOUTLIER ANALYSIS\n")
        f.write("-"*80 + "\n")
        
        for target, data in outliers.items():
            f.write(f"\n{target}:\n")
            f.write(f"   Outliers: {data['n_outliers']} ({data['pct_outliers']:.2f}%)\n")
    
    print(f"   ✅ Report saved to {output_path}")


def main(args):
    print("\n" + "="*100)
    print("PSEUDO-LABEL ANALYSIS AND COMPARISON")
    print("="*100)
    
    target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Load datasets
    generated, reference = load_datasets(args.generated, args.reference)
    
    # Align datasets
    merged = align_datasets(generated, reference)
    
    # Compute statistics
    stats = compute_statistics(merged, target_names)
    
    # Print statistics
    print_statistics(stats)
    
    # Identify outliers
    outliers = identify_outliers(merged, target_names, threshold=args.outlier_threshold)
    print_outliers(outliers)
    
    # Generate plots
    if args.plot:
        plot_comparison(merged, target_names, os.path.dirname(args.generated))
    
    # Save report
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    save_report(stats, outliers, args.output_report)
    
    print(f"\n{'='*100}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*100}")
    
    # Summary
    print("\n📋 SUMMARY:")
    print(f"   Generated pseudo-labels are {'GOOD' if np.mean([s['correlation'] for s in stats.values()]) > 0.8 else 'NEEDS REVIEW'}")
    print(f"   Average correlation: {np.mean([s['correlation'] for s in stats.values()]):.3f}")
    print(f"   Total samples compared: {len(merged)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze generated pseudo-labels")
    parser.add_argument(
        "--generated",
        type=str,
        required=True,
        help="Path to generated pseudo-labels CSV"
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference pseudo-labels CSV"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="pseudolabel/analysis_report.txt",
        help="Path to save analysis report"
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=2.0,
        help="Threshold for outlier detection (in standard deviations)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate comparison plots"
    )
    
    args = parser.parse_args()
    main(args)

