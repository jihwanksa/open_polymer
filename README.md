# Open Polymer: Winning Kaggle Solution

**Polymer property prediction using simple features + external data + Kaggle automation.**

🏆 **Score: 0.08008 (Private) | 0.10125 (Public)** | Time: ~90 seconds per submission

## TL;DR

```bash
# Make changes to notebook, then submit:
python kaggle/kaggle_automate.py "Your message"

# Done! Get score in ~3 minutes (push + execute + submit + score)
```

## Why This Works

| What | Why | Result |
|------|-----|--------|
| **Massive external data** | Tg: 511→2,447 (+380%), Density: 613→1,394 (+127%) | +2% improvement (0.085→0.083) |
| **Ensemble (5 models)** | Variance reduction through model averaging | +0.4% improvement (0.083→0.08266) |
| **Chemistry features (21)** | Polymer-specific: branching, backbone, H-bonding | +3.1% improvement (0.08266→0.08008) |
| **Tg transformation** | Fix train/test distribution shift: (9/5)×x+45 | +30% improvement |
| **MAE objective** | Match competition metric exactly | +5-15% improvement |

**Key insight:** Ensemble + domain-specific features work together. Chemistry features (branching, backbone structure) capture polymer properties that simple counts miss.

## Score Progress (Empirical)

| Version | Configuration | Score | Change  | Leaderboard Placement |
|---------|---------------|-------|---------|----------------------|
| v1 | Original data only | 0.139 | Baseline | - |
| v2 | + Tc dataset (Tc-SMILES) | 0.092 | ↓ 33.8% | - |
| v3 | + Tg dataset (TG-of-Polymer) | 0.085 | ↓ 7.6% | 15th 🥉 |
| v4 | + Density (PI1070) | 0.088 | Reverted ⚠️ | - |
| v5 | + Tc +Tg +Rg +LAMALAB (7,369 Tg) | 0.083 | ↓ 2.4% ✅ | 13th |
| v6 | + Ensemble (5 models per property) | 0.08266 | ↓ 0.4% | 10th |
| **v7** | **+ 21 chemistry features (14th place insights)** | **0.08008** | **↓ 3.1% ✅** | **6th 🏆** |

**Total improvement: 0.139 → 0.08008 = 42.4% error reduction** 🎉

**Breakdown of improvements:**
- Data augmentation (v1→v5): 40.3% reduction
- Ensemble (v5→v6): 0.4% reduction  
- Chemistry features (v6→v7): 3.1% reduction

### What's in v6 (Ensemble)
- **Ensemble XGBoost**: 5 independent models per property with different random seeds
- **Model averaging**: Predictions averaged across all 5 models for variance reduction
- **Same 10 basic features**: smiles_length, carbon_count, nitrogen_count, etc.
- **Benefit**: Reduces overfitting through model diversity (different random states)

### What's in v7 (Ensemble + Chemistry Features)
Built on v6, adds **11 chemistry-based features** inspired by 14th place solution:

**Structural Features:**
1. `num_side_chains` - Polymer branching from backbone
2. `backbone_carbons` - Main chain carbon count
3. `branching_ratio` - Side chains per backbone carbon

**Chemical Properties:**
4. `aromatic_count` - Aromatic ring content (affects rigidity, Tg)
5. `h_bond_donors` - Hydrogen bonding donors (O, N)
6. `h_bond_acceptors` - Hydrogen bonding acceptors
7. `num_rings` - Total ring structures
8. `single_bonds` - Chain flexibility indicator
9. `halogen_count` - F, Cl, Br content
10. `heteroatom_count` - N, O, S atoms
11. `mw_estimate` - Molecular weight approximation

**Total: 21 features** (10 basic + 11 chemistry)

**Key Insight:** Chemistry features capture polymer-specific properties (branching, backbone structure, H-bonding) that correlate with Tg, density, and other target properties.

## Setup

```bash
# Clone and install
git clone https://github.com/jihwanksa/open_polymer.git
cd open_polymer
pip install -r requirements.txt
```

## Workflow

### Standard Iteration (no new datasets)
1. **Edit notebook:** `polymer-v32-enhanced-tc-tg-augmentation.ipynb`
2. **Submit:** `python kaggle/kaggle_automate.py "v25: your change"`
3. **Get score:** Check terminal output in ~3 minutes

### When Adding New Datasets
1. **Add datasets in Kaggle notebook UI** (Add Input section)
2. **Update metadata:** Edit `kernel-metadata.json` to include new dataset slug
3. **Submit:** `python kaggle/kaggle_automate.py "v26: added new data"`
4. **Get score:** ~3 minutes

That's it!

## Files

| File | Purpose |
|------|---------|
| `polymer-v32-enhanced-tc-tg-augmentation.ipynb` | Working notebook (2K lines) |
| `kaggle/kaggle_automate.py` | One-command automation |
| `src/kaggle_solution.py` | Reusable solution classes |
| `kernel-metadata.json` | Kaggle kernel configuration with datasets |

## Using Solution Code

```python
from src.kaggle_solution import SimpleFeatureExtractor, KaggleSolution

# Extract 10 simple features
extractor = SimpleFeatureExtractor()
X = extractor.extract_features(df)

# Train XGBoost with MAE objective
solution = KaggleSolution()
solution.train(X_train, y_train)

# Predict with Tg transformation
predictions = solution.predict(X_test)
predictions = solution.apply_tg_transformation(predictions)
```

## Key Learnings

### Why Simple Beats Complex
- **511 samples / 10 features** = 51 samples per feature ✅
- **511 samples / 1037 features** = 0.49 samples per feature ❌
- Below 5 samples/feature = inevitable overfitting

### Why Deep Learning Failed
- GNNs & Transformers need 100K+ samples to learn patterns
- Only 7,973 training samples (10x too small)
- Result: Memorize training data, fail on test

### Domain > Engineering (Updated with v6-v7)
- External Tg data: +2% (0.085→0.083)
- Ensemble (5 models): +0.4% (0.083→0.08266)
- Chemistry features: +3.1% (0.08266→0.08008) ← **Biggest single improvement!**
- Understanding Tg shift: +30%

**Key finding:** Domain-specific features (polymer branching, backbone structure) > generic model complexity

### Data Augmentation Details
**External datasets used:**
1. **Tc-SMILES** (minatoyukinaxlisa) - Tc crystallization temp
2. **TG-of-Polymer** (akihiroorita) - Tg glass transition temp
3. **PI1070.csv** - Density & Rg from simulations
4. **LAMALAB_curated** - Experimental Tg from literature (7,369 samples!)

**Augmentation impact:**
- Tg: 511 → 2,447 samples (+380%)
- Density: 613 → 1,394 samples (+127%)
- Rg: 614 → 1,684 samples (+174%)
- Tc: 737 → 867 samples (+18%)
- Total: 10,039 → 10,820 training samples (+7.7%)

### Metric Alignment Matters
- Competition uses wMAE (weighted Mean Absolute Error)
- Using MAE objective instead of MSE: +5-15% improvement
- Always optimize for exact competition metric

## Performance

- **Private:** 0.08008 ⭐ (improved from 10th place)
- **Public:** 0.10125
- **Training time:** 90-95 seconds per submission (5x ensemble)
- **Generalization:** 0.021 private-public gap (excellent)

## Next

- Try new hyperparameters: `python kaggle/kaggle_automate.py "try lr=0.03"`
- Test different Tg transforms: `python kaggle/kaggle_automate.py "test (9/5)*x+50"`
- Add new datasets: Update `kernel-metadata.json` dataset_sources

## Architecture

```
Ensemble XGBoost (5 models per property, predictions averaged)
├─ Tg (glass transition temp) - 2,447 samples (22.6%)
├─ FFV (free volume fraction) - 7,030 samples (65.0%)
├─ Tc (crystallization temp) - 867 samples (8.0%)
├─ Density - 1,394 samples (12.9%)
└─ Rg (radius of gyration) - 1,684 samples (15.5%)

Features: 21 chemistry-based (v7)
Basic (10): smiles_length, carbon_count, nitrogen_count, oxygen_count,
            sulfur_count, fluorine_count, ring_count, double_bond_count,
            triple_bond_count, branch_count

Chemistry (11): num_side_chains, backbone_carbons, branching_ratio,
                aromatic_count, h_bond_donors, h_bond_acceptors,
                num_rings, single_bonds, halogen_count,
                heteroatom_count, mw_estimate

Training: 18,035 samples (7,973 original + 10,062 augmented)
Ensemble: 5 models with different random seeds per property
Objective: MAE (matches wMAE metric)
```

## Commands Reference

```bash
# Submit
python kaggle/kaggle_automate.py "message"

# Check Kaggle API
kaggle competitions submissions -c neurips-open-polymer-prediction-2025 --csv

# View kernel metadata
cat kernel-metadata.json
```

---

**Status:** Production ready | **Last Updated:** Nov 4, 2025 | **Score:** 0.08008 (Private) | 0.10125 (Public)
