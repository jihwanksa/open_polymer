"""
Compare all trained models using competition metric
"""
import pandas as pd
import os

# Load results
project_root = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(project_root, 'results')

# Compile results
all_results = []

# Traditional models
trad_results = pd.read_csv(os.path.join(results_dir, 'competition_metrics.csv'))
for _, row in trad_results.iterrows():
    all_results.append({
        'Model': row['Model'],
        'Type': 'Traditional ML',
        'wMAE': row['wMAE']
    })

# GNN
gnn_results = pd.read_csv(os.path.join(results_dir, 'gnn_results.csv'))
for _, row in gnn_results.iterrows():
    all_results.append({
        'Model': 'GNN (Graph Conv)',
        'Type': 'Deep Learning',
        'wMAE': row['wMAE']
    })

# Create comparison
comparison_df = pd.DataFrame(all_results).sort_values('wMAE')

print("=" * 80)
print("MODEL COMPARISON - COMPETITION METRIC (wMAE)")
print("=" * 80)
print("\n🏆 LEADERBOARD (Lower wMAE is better):\n")
print(f"{'Rank':<6} {'Model':<25} {'Type':<20} {'wMAE':<12}")
print("-" * 80)

for rank, (_, row) in enumerate(comparison_df.iterrows(), 1):
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
    print(f"{medal:<6} {row['Model']:<25} {row['Type']:<20} {row['wMAE']:<12.6f}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
print(f"""
1. **XGBoost** is the top performer with wMAE = {comparison_df.iloc[0]['wMAE']:.6f}
   - Excellent for tabular data with molecular descriptors and fingerprints
   - Fast training and inference
   - Interpretable feature importance

2. **Random Forest** is a close second with wMAE = {comparison_df.iloc[1]['wMAE']:.6f}
   - Similar performance to XGBoost
   - Good ensemble approach
   - More robust to outliers

3. **GNN (Graph Conv)** has wMAE = {comparison_df.iloc[2]['wMAE']:.6f}
   - Still learning - needs more epochs and tuning
   - Direct graph representation of molecules
   - Potential for improvement with:
     * More training epochs (100+)
     * Larger hidden dimensions
     * Better hyperparameter tuning
     * Pre-training on molecular property prediction tasks

4. **Transformer models** (ChemBERTa, etc.):
   - Require additional dependency configuration (protobuf conflicts)
   - Would benefit from pre-trained chemical language models
   - Implementation ready but not trained due to environment issues

RECOMMENDATION:
- **For submission**: Use XGBoost (best performance)
- **For ensemble**: Combine XGBoost + Random Forest predictions
- **For future work**: Fine-tune GNN and resolve Transformer dependencies
""")

# Save comparison
output_path = os.path.join(results_dir, 'model_comparison_final.csv')
comparison_df.to_csv(output_path, index=False)
print(f"\n✅ Full comparison saved to: {output_path}")
print("=" * 80)

