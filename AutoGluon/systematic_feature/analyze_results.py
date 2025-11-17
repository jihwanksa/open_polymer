"""
Analyze results from systematic feature analysis

Generates:
1. Feature efficiency metrics (redundancy analysis)
2. Performance comparison across configurations
3. Feature selection patterns
4. Recommendations for optimal feature subset
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_config_results(results_dir):
    """Load all configuration results"""
    all_results = {}
    
    for config_dir in sorted(Path(results_dir).glob('*_*')):
        if config_dir.is_dir():
            config_name = config_dir.name
            results_file = config_dir / 'config_results.json'
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    all_results[config_name] = json.load(f)
    
    return all_results


def calculate_metrics(all_results):
    """Calculate key metrics for each configuration"""
    metrics = []
    
    for config_name, results in all_results.items():
        input_features = results['input_features']
        models = results['models']
        
        if not models:
            continue
        
        selected_features = [m['selected_features'] for m in models.values()]
        avg_selected = np.mean(selected_features)
        
        metrics.append({
            'Configuration': config_name,
            'Input Features': input_features,
            'Avg Selected Features': f"{avg_selected:.1f}",
            'Feature Reduction %': f"{(1 - avg_selected/input_features)*100:.1f}%",
            'Models Trained': len(models),
            'Training Time (s)': f"{results.get('training_time', 0):.1f}"
        })
    
    return pd.DataFrame(metrics)


def generate_feature_efficiency_report(results_dir):
    """Generate feature efficiency analysis"""
    
    print("\n" + "="*80)
    print("SYSTEMATIC FEATURE ANALYSIS - RESULTS REPORT")
    print("="*80 + "\n")
    
    all_results = load_config_results(results_dir)
    
    if not all_results:
        print("❌ No results found in", results_dir)
        return
    
    # Metrics comparison
    print("📊 FEATURE EFFICIENCY METRICS\n")
    metrics_df = calculate_metrics(all_results)
    print(metrics_df.to_string(index=False))
    
    # Key insights
    print("\n\n" + "="*80)
    print("🔍 KEY INSIGHTS")
    print("="*80 + "\n")
    
    # Find configuration with best feature reduction
    best_reduction = metrics_df.loc[metrics_df['Feature Reduction %'].str.rstrip('%').astype(float).idxmax()]
    print(f"✅ Best Feature Reduction: {best_reduction['Configuration']}")
    print(f"   {best_reduction['Feature Reduction %']} reduction")
    print(f"   {best_reduction['Avg Selected Features']} features selected from {best_reduction['Input Features']}")
    
    # Feature categories analysis
    print("\n\n" + "="*80)
    print("📈 DETAILED CONFIGURATION ANALYSIS")
    print("="*80 + "\n")
    
    configs_info = {
        'A_simple_only': {
            'Type': 'Baseline',
            'Components': 'Simple (10)',
            'Expected': 'Weakest - just SMILES counting'
        },
        'B_hand_crafted_only': {
            'Type': 'Domain Knowledge',
            'Components': 'Hand-crafted (11)',
            'Expected': 'Good - pure domain expertise'
        },
        'C_current_baseline': {
            'Type': 'Current System',
            'Components': '10 Simple + 11 HC + 13 RDKit',
            'Expected': 'Good baseline for comparison'
        },
        'D_expanded_rdkit': {
            'Type': 'RDKit Expansion',
            'Components': '10 Simple + 11 HC + 35 RDKit',
            'Expected': 'May find additional useful descriptors'
        },
        'E_all_rdkit': {
            'Type': 'Full Feature Space',
            'Components': '10 Simple + 11 HC + 200 RDKit',
            'Expected': 'Highest dimensional - best feature selection test'
        },
        'F_rdkit_only_expanded': {
            'Type': 'Isolated RDKit',
            'Components': '35 RDKit descriptors',
            'Expected': 'Test RDKit performance alone'
        },
        'G_no_simple': {
            'Type': 'Ablation Study',
            'Components': '11 HC + 13 RDKit (no simple)',
            'Expected': 'Assess simple feature value'
        },
        'H_no_hand_crafted': {
            'Type': 'Ablation Study',
            'Components': '10 Simple + 13 RDKit (no HC)',
            'Expected': 'Assess domain knowledge value'
        }
    }
    
    for config_name in all_results.keys():
        if config_name in configs_info:
            info = configs_info[config_name]
            results = all_results[config_name]
            
            print(f"\n{config_name.upper()}")
            print(f"  Type: {info['Type']}")
            print(f"  Components: {info['Components']}")
            print(f"  Expected: {info['Expected']}")
            print(f"  Input Features: {results['input_features']}")
            print(f"  Models Trained: {len(results['models'])}")
            if results['models']:
                avg_selected = np.mean([m['selected_features'] for m in results['models'].values()])
                print(f"  Avg Selected: {avg_selected:.1f} features")
    
    # Recommendations
    print("\n\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80 + "\n")
    
    print("1. FEATURE EFFICIENCY:")
    print("   - Check if configurations with fewer input features (A, B) still")
    print("     achieve good performance - simpler is better!")
    print("   - If E_all_rdkit selects only 15-20 features, we're missing important ones")
    print("     in our manual selection (C_current)")
    
    print("\n2. DOMAIN KNOWLEDGE VALUE:")
    print("   - Compare B vs F: Does hand-crafted outperform expanded RDKit alone?")
    print("   - Compare G vs H: Which ablation hurts performance more?")
    print("   - This shows whether domain knowledge or RDKit is more important")
    
    print("\n3. OPTIMAL FEATURE SET:")
    print("   - Compare C vs D vs E: Does more RDKit help?")
    print("   - If D and E perform similarly, expanded RDKit (56) is sufficient")
    print("   - If E is significantly better, the full RDKit space is valuable")
    
    print("\n4. COMPUTATIONAL EFFICIENCY:")
    print("   - Balance between input features and performance improvement")
    print("   - If A performs 95% as well as C with 3x fewer features, use A!")
    
    # Save report
    report_path = Path(results_dir) / 'ANALYSIS_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write("SYSTEMATIC FEATURE ANALYSIS - RESULTS REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(metrics_df.to_string(index=False))
    
    print(f"\n\n✅ Analysis saved to: {report_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to current directory's results
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(project_root, 'AutoGluon/systematic_feature/results')
    
    generate_feature_efficiency_report(results_dir)

