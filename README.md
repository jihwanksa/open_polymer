# Open Polymer: Winning Kaggle Solution

**Polymer property prediction using simple features + external data + Kaggle automation.**

🏆 **Score: 0.07874 (Private) | 0.10354 (Public) | v53: Random Forest Ensemble 4th place in leaderboard** | Time: ~50 seconds per submission

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

### Phase 1: Data Augmentation (v1-v7)
| Version | Configuration | Score | Change  | Leaderboard Placement |
|---------|---------------|-------|---------|----------------------|
| v1 | Original data only | 0.139 | Baseline | 2000th |
| v2 | + Tc dataset (Tc-SMILES) | 0.09228 | ↓ 33.8% | 453th |
| v3 | + Tg dataset (TG-of-Polymer) | 0.08548 | ↓ 7.6% | 15th 🥉 |
| v4 | + Density (PI1070) | 0.08871 | Reverted ⚠️ | 66th |
| v5 | + Tc +Tg +Rg +LAMALAB (7,369 Tg) | 0.08334 | ↓ 2.4% ✅ | 13th |
| v6 | + Ensemble (5 models per property) | 0.08266 | ↓ 0.4% | 10th |
| v7 | + 21 chemistry features (14th place insights) | 0.08008 | ↓ 3.1% ✅ | 6th 🏆 |

### Phase 2: Systematic Hyperparameter & Model Testing (v48-v54)

| Version | Model | Hyperparameters | Private / Public | Change | Status |
|---------|-------|----------------|------------------|--------|--------|
| **v48** | **XGBoost Ensemble** | n=500, lr=0.05, depth=8, subsample=0.8, colsample=0.8 | **0.08008 / 0.10125** | Baseline | ✅ |
| v50 | XGBoost | n=500, lr=0.05, **depth=10**, **reg_lambda=3.0**, **min_child_weight=3** | 0.08081 / 0.09915 | ↑ +0.9% ❌ | Worse |
| v51 | XGBoost | **n=800**, **lr=0.03**, depth=8, subsample=0.8, colsample=0.8 | 0.07977 / 0.10186 | ↓ 0.4% ✅ | Better! |
| v52 | XGBoost | n=500, lr=0.05, depth=8, **subsample=0.9**, **colsample=0.7**, **colsample_bylevel=0.7** | 0.07965 / 0.10102 | ↓ 0.5% ✅ | **Better! 5th place in leaderboard 🥇** |
| **v53** | **Random Forest** | **n=500**, **depth=15**, **min_split=5**, **min_leaf=2**, **max_features='sqrt'** | **0.07874 / 0.10354** | **↓ 1.7% ✅** | **BEST! 4th place in leaderboard 🥇** |
| v54 | LightGBM | n=500, depth=8, num_leaves=63, objective='mae' | 0.08011 / 0.09492 | ↑ +0.1% ❌ | Slightly worse |

**Total improvement: 0.139 → 0.07874 = 43.4% error reduction** 🎉

**Key Findings:**

1. **🏆 Random Forest (v53) wins!** 
   - Simple decision tree ensemble with depth=15, max_features='sqrt'
   - **Best private score: 0.07874** (1.7% improvement over v48, 1.1% better than XGBoost v52)
   - Validation: Tg MAE=36.18 (R²=0.80), Density MAE=0.045 (R²=0.82)
   - **Why it works**: Random Forest's variance reduction through bagging complements the chemistry features well

2. **Feature sampling XGBoost (v52) strong runner-up**
   - Subsample 0.9 (more data per tree) + colsample 0.7 (feature diversity)
   - Private score: 0.07965 (0.5% improvement over v48)
   - Improves generalization by reducing feature correlation

3. **LightGBM (v54) slightly worse than XGBoost**
   - Score: 0.08011 / 0.09492 (0.1% worse than baseline)
   - Fast training but slightly overfits compared to XGBoost
   - Best validation on Density (MAE=0.039, R²=0.84)

4. **More trees + slower learning (v51) helps**
   - 800 trees with lr=0.03 achieves 0.07977 (0.4% improvement)
   - Nearly as good as v52, but 60% longer training time

5. **Deeper trees + regularization (v50) slightly worse**
   - depth=10 + L2 regularization causes slight overfitting
   - Private: 0.08081 (+0.9% worse than baseline)
   - Public: 0.09915 (better than baseline, but doesn't generalize)

**Key insight**: Random Forest's bagging approach (bootstrap + feature randomness) works better than XGBoost's boosting for this polymer dataset with 21 chemistry features. The simpler averaging strategy is more robust than gradient boosting's sequential error correction.

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

### 1. Clone Repository
```bash
git clone https://github.com/jihwanksa/open_polymer.git
cd open_polymer
```

### 2. Create Conda Environment (Recommended)
```bash
# Create environment with Python 3.10
conda create -n polymer python=3.10 -y

# Activate environment
conda activate polymer

# Install dependencies
pip install -r requirements.txt
```

**Note:** Using conda is recommended because:
- `torch_geometric` and `torch` require compatible versions
- Avoids system Python conflicts
- Isolated dependencies per project

### 3. Verify Installation
```bash
python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}, PyG: {torch_geometric.__version__}')"
```

## 🏆 Best Kaggle Submission

**`best.ipynb`** - The best performing Kaggle notebook (4th place leaderboard)
- **Model:** Random Forest Ensemble with 21 chemistry features
- **Private Score:** 0.07874 ⭐
- **Public Score:** 0.10354
- **Key Feature:** Tg transformation from 2nd place winner's discovery
- **Status:** Ready to submit - your colleague can fork and run this directly!

This notebook is **production-ready** and achieved the best competition results locally.

## Local Training

Train models locally to validate performance before pushing to Kaggle:

### 1. Train Best Random Forest Model (v53)
```bash
conda activate polymer
python src/train_v53_best.py
```
**Output:** `models/random_forest_v53_best.pkl` (the 4th place leaderboard model)
**Score:** Private 0.07874 | Public 0.10354
**Time:** ~50 seconds
**Features:** 21 chemistry-based features + data augmentation

### 2. Train Graph Neural Network (GNN)
```bash
conda activate polymer
python src/train_gnn_tuned.py
```
**Output:** `models/gnn_best_tuned.pt`
**Current Best:** GNN_Wider with 0.173055 wMAE (validation)
**Features:** RDKit-enhanced node features (16 dims) + edge features (6 dims)
**GPU Acceleration:** 
- NVIDIA CUDA: Automatic ✅
- Apple Silicon (M1/M2/M3): MPS enabled ✅ (2-5x faster than CPU!)
- Intel Mac: CPU fallback ⚠️

**Verify your device:**
```bash
conda activate polymer
python test_mps_device.py
```

### 3. Train Transformer Model (Optional)
```bash
conda activate polymer
python src/train_transformer.py
```
**Output:** `models/transformer_model.pt`
**Note:** Requires 20+ minutes on GPU
**Architecture:** DistilBERT-based SMILES encoder

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

### Graph Neural Networks (GNNs) - With Chemistry Enhancement
- **Previous GNN best (basic features)**: 0.177712 wMAE (GNN_Deeper)
- **Current GNN best (RDKit-enhanced)**: 0.173055 wMAE (GNN_Wider) ✅ **+2.6% improvement!**
- **Overall validation wMAE**: 0.189640 (validates RDKit enhancement working)
- **Enhancement**: Richer node features (16 dims) + edge features (6 dims) from RDKit
- Node features now include: atom valence, hybridization (SP/SP2/SP3), ring membership
- Edge features now include: bond type, aromaticity, ring membership

**Property-wise Performance (GNN_Wider):**
| Property | n | MAE | wMAE | R² |
|----------|---|-----|------|-----|
| Tg (glass transition) | 87 | 88.82 | 0.259 | -0.41 |
| FFV (free volume) | 1419 | 0.039 | 0.051 | -4.12 |
| Tc (crystallization) | 144 | 0.156 | 0.403 | -3.61 |
| Density | 123 | 0.673 | 0.705 | -26.68 |
| Rg (radius) | 124 | 11.03 | 0.505 | -5.80 |

- **Key insight**: Intrinsic chemistry encoding (from RDKit) helps GNNs learn better than artificial graph-level summaries
- **Limitation**: Still below traditional ML (0.07874), but shows promise with proper feature engineering

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

- **Best Model:** v53 (Random Forest Ensemble)
- **Private:** 0.07874 🥇 (1.7% improvement over v48, 43.4% total from baseline)
- **Public:** 0.10354
- **Training time:** 50 seconds per submission (5x ensemble, faster than XGBoost)
- **Generalization:** 0.026 private-public gap
- **Key insight:** Random Forest's bagging (bootstrap + feature randomness) beats gradient boosting for this chemistry feature set

## Next

- Try new hyperparameters: `python kaggle/kaggle_automate.py "try lr=0.03"`
- Test different Tg transforms: `python kaggle/kaggle_automate.py "test (9/5)*x+50"`
- Add new datasets: Update `kernel-metadata.json` dataset_sources

## Architecture

```
v53: Ensemble Random Forest (5 models per property, predictions averaged) 🥇
├─ Tg (glass transition temp) - 9,814 samples (54.4%)
├─ FFV (free volume fraction) - 7,030 samples (39.0%)
├─ Tc (crystallization temp) - 867 samples (4.8%)
├─ Density - 1,242 samples (6.9%)
└─ Rg (radius of gyration) - 1,243 samples (6.9%)

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
Algorithm: Bootstrap aggregating (bagging) with feature randomness

Hyperparameters (v53 - BEST):
  n_estimators: 500
  max_depth: 15         ← Deeper trees than XGBoost
  min_samples_split: 5
  min_samples_leaf: 2
  max_features: 'sqrt'  ← √21 ≈ 4.6 features per split (strong randomness)
  bootstrap: True       ← Default: each tree sees different data
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

**Status:** Production ready | **Last Updated:** Nov 10, 2025 | **Best Model:** v53 Random Forest Ensemble | **Score:** 0.07874 (Private) | 0.10354 (Public)
