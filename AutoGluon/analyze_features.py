"""
Analyze AutoGluon feature importance and generate insights.

This script:
1. Loads trained AutoGluon models
2. Extracts feature importance for each property
3. Identifies most/least important features
4. Generates summary report
5. Creates visualizations

Usage:
    python AutoGluon/analyze_features.py \
        --model_dir models/autogluon_production \
        --output_report AutoGluon/feature_analysis.txt
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def analyze_features(model_dir, output_report):
    """Analyze feature importance across all properties"""
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80 + "\n")
    
    # Load feature importance
    importance_path = os.path.join(model_dir, 'feature_importance.json')
    
    if not Path(importance_path).exists():
        print(f"❌ Feature importance file not found: {importance_path}")
        print("   Run training first: python AutoGluon/train_autogluon_production.py")
        return
    
    with open(importance_path, 'r') as f:
        all_importance = json.load(f)
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Generate report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("AUTOGLUON FEATURE IMPORTANCE ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("21 Chemistry Features:")
    report_lines.append("-"*80)
    report_lines.append("Simple Features (10):")
    report_lines.append("  1. smiles_length - Number of characters in SMILES")
    report_lines.append("  2. carbon_count - Number of carbon atoms")
    report_lines.append("  3. nitrogen_count - Number of nitrogen atoms")
    report_lines.append("  4. oxygen_count - Number of oxygen atoms")
    report_lines.append("  5. sulfur_count - Number of sulfur atoms")
    report_lines.append("  6. fluorine_count - Number of fluorine atoms")
    report_lines.append("  7. ring_count - Number of rings")
    report_lines.append("  8. double_bond_count - Number of double bonds")
    report_lines.append("  9. triple_bond_count - Number of triple bonds")
    report_lines.append(" 10. branch_count - Number of branches")
    report_lines.append("")
    report_lines.append("Polymer-Specific Features (11):")
    report_lines.append(" 11. num_side_chains - Number of side chains")
    report_lines.append(" 12. backbone_carbons - Carbons in main chain")
    report_lines.append(" 13. aromatic_count - Number of aromatic atoms")
    report_lines.append(" 14. h_bond_donors - Hydrogen bond donor count")
    report_lines.append(" 15. h_bond_acceptors - Hydrogen bond acceptor count")
    report_lines.append(" 16. num_rings - Number of ring structures")
    report_lines.append(" 17. single_bonds - Number of single bonds")
    report_lines.append(" 18. halogen_count - Number of halogens")
    report_lines.append(" 19. heteroatom_count - Number of non-carbon atoms")
    report_lines.append(" 20. mw_estimate - Estimated molecular weight")
    report_lines.append(" 21. branching_ratio - Ratio of branches to backbone")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("FEATURE IMPORTANCE PER PROPERTY")
    report_lines.append("="*80)
    report_lines.append("")
    
    all_features_importance = {}
    
    for target in target_cols:
        if target in all_importance and all_importance[target]:
            report_lines.append(f"\n{target}:")
            report_lines.append("-"*80)
            
            importance_dict = all_importance[target]
            
            # Convert to DataFrame for sorting
            if isinstance(importance_dict, dict):
                importance_df = pd.DataFrame(
                    list(importance_dict.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
            else:
                continue
            
            report_lines.append(f"Top 10 Most Important Features:")
            for i, row in enumerate(importance_df.head(10).itertuples(), 1):
                report_lines.append(f"  {i:2d}. {row.Feature:25s} {row.Importance:8.4f}")
            
            report_lines.append(f"\nBottom 5 Least Important Features:")
            for i, row in enumerate(importance_df.tail(5).itertuples(), 1):
                report_lines.append(f"  {i:2d}. {row.Feature:25s} {row.Importance:8.4f}")
            
            all_features_importance[target] = importance_df
        else:
            report_lines.append(f"\n{target}:")
            report_lines.append("  ⚠️  No feature importance data available")
    
    # Overall summary
    report_lines.append("\n" + "="*80)
    report_lines.append("FEATURE USAGE SUMMARY")
    report_lines.append("="*80 + "\n")
    
    # Count how many properties use each feature
    all_features = set()
    feature_usage = {}
    
    for target, importance_df in all_features_importance.items():
        for feat in importance_df['Feature']:
            all_features.add(feat)
            if feat not in feature_usage:
                feature_usage[feat] = 0
            feature_usage[feat] += 1
    
    report_lines.append("Features used in multiple properties (more universal):")
    for feat, count in sorted(feature_usage.items(), key=lambda x: x[1], reverse=True):
        if count >= 3:
            report_lines.append(f"  {feat:25s} - Used in {count}/5 properties")
    
    report_lines.append("\nFeatures used rarely (might be redundant):")
    for feat, count in sorted(feature_usage.items(), key=lambda x: x[1]):
        if count == 1:
            report_lines.append(f"  {feat:25s} - Used in {count}/5 properties only")
    
    # Key insights
    report_lines.append("\n" + "="*80)
    report_lines.append("KEY INSIGHTS & RECOMMENDATIONS")
    report_lines.append("="*80 + "\n")
    
    # Identify consistently important features
    top_features_per_target = {}
    for target, importance_df in all_features_importance.items():
        top_5 = importance_df.head(5)['Feature'].tolist()
        top_features_per_target[target] = top_5
    
    # Find features in top 5 for multiple properties
    universal_features = {}
    for feat in all_features:
        count = sum(1 for target in top_features_per_target if feat in top_features_per_target.get(target, []))
        if count >= 3:
            universal_features[feat] = count
    
    if universal_features:
        report_lines.append("🌟 UNIVERSAL FEATURES (top 5 for multiple properties):")
        for feat, count in sorted(universal_features.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  ✓ {feat:25s} - Top 5 in {count}/5 properties")
    
    report_lines.append("\n💡 INTERPRETATION:")
    report_lines.append("  - Features appearing in top 5 for 3+ properties are highly important")
    report_lines.append("  - Features used only once might be redundant or property-specific")
    report_lines.append("  - Consider simplifying model by removing least important features")
    report_lines.append("  - AutoGluon already handles feature interaction internally")
    
    # Print and save
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, 'w') as f:
        f.write(report_text)
    
    print(f"\n✅ Report saved to: {output_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AutoGluon feature importance")
    parser.add_argument("--model_dir", type=str, default="models/autogluon_production",
                       help="Directory with trained AutoGluon models")
    parser.add_argument("--output_report", type=str, default="AutoGluon/feature_analysis.txt",
                       help="Output file for analysis report")
    
    args = parser.parse_args()
    analyze_features(args.model_dir, args.output_report)

