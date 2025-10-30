# Open Polymer: Winning Kaggle Solution

**Polymer property prediction using simple features + external data + Kaggle automation.**

Score: **0.085 (Private) | 0.100 (Public)** | Time: ~50 seconds per submission

## TL;DR

```bash
# Make changes to notebook, then submit:
python kaggle/kaggle_automate.py "Your message"

# Done! Get score in ~3 minutes (push + execute + submit + score)
```

## Why This Works

| What | Why | Result |
|------|-----|--------|
| **10 simple features** | Prevent overfitting on 500-700 samples | 2.35x better than 1037 features |
| **External data** | 511→2,447 Tg samples, 737→867 Tc samples | +7.7% improvement |
| **Tg transformation** | Fix train/test distribution shift: (9/5)×x+45 | +30% improvement |
| **MAE objective** | Match competition metric exactly | +5-15% improvement |

**Key insight:** On small datasets, simplicity wins. 10 features > 1037 features (proved empirically).

## Setup

```bash
# Clone and install
git clone https://github.com/jihwanksa/open_polymer.git
cd open_polymer
pip install -r requirements.txt
```

## Workflow

### Standard Iteration (no new datasets)
1. **Edit notebook:** `polymer-v2-enhanced-tc-tg-augmentation.ipynb`
2. **Submit:** `python kaggle/kaggle_automate.py "v25: your change"`
3. **Get score:** Check terminal output in ~3 minutes

### When Adding New Datasets
1. **Add datasets in Kaggle notebook UI** (Add Input section)
2. **Update metadata:** `python kaggle/sync_metadata.py`
3. **Submit:** `python kaggle/kaggle_automate.py "v26: added new data"`
4. **Get score:** ~3 minutes

That's it!

## Files

| File | Purpose |
|------|---------|
| `polymer-v2-enhanced-tc-tg-augmentation.ipynb` | Working notebook (2K lines) |
| `kaggle/kaggle_automate.py` | One-command automation |
| `src/kaggle_solution.py` | Reusable solution classes |
| `kaggle/sync_metadata.py` | Auto-update kernel-metadata.json |

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

## Auto-Update Kernel Metadata

Instead of manually editing `kernel-metadata.json`:

```bash
# Update datasets list (after adding new inputs to Kaggle notebook)
python kaggle/sync_metadata.py
```

This script:
- Lists the datasets used by the kernel
- Updates kernel-metadata.json with one command
- Minimal manual work needed

## Key Learnings

### Why Simple Beats Complex
- **511 samples / 10 features** = 51 samples per feature ✅
- **511 samples / 1037 features** = 0.49 samples per feature ❌
- Below 5 samples/feature = inevitable overfitting

### Why Deep Learning Failed
- GNNs & Transformers need 100K+ samples to learn patterns
- Only 7,973 training samples (10x too small)
- Result: Memorize training data, fail on test

### Domain > Engineering
- External data: +7.7%
- Better features: +0%
- Understanding Tg shift: +30% ← **This won the competition**

### Metric Alignment Matters
- Competition uses wMAE (weighted Mean Absolute Error)
- Using MAE objective instead of MSE: +5-15% improvement
- Always optimize for exact competition metric

## Performance

- **Private:** 0.085 ⭐
- **Public:** 0.100
- **Training time:** 48-64 seconds per submission
- **Generalization:** 0.1% CV-test gap (excellent)

## Next

- Try new hyperparameters: `python kaggle/kaggle_automate.py "try lr=0.03"`
- Test different Tg transforms: `python kaggle/kaggle_automate.py "test (9/5)*x+50"`
- Add new datasets: Use `sync_metadata.py` to auto-update

## Architecture

```
XGBoost (separate models for each property)
├─ Tg (glass transition temp)
├─ FFV (free volume fraction)
├─ Tc (crystallization temp)
├─ Density
└─ Rg (radius of gyration)

Features: 10 simple SMILES-based
- smiles_length, carbon_count, nitrogen_count, oxygen_count
- sulfur_count, fluorine_count, ring_count, double_bond_count
- triple_bond_count, branch_count

Training: 10,039 samples (7,973 original + 2,066 augmented)
Objective: MAE (matches wMAE metric)
```

## Commands Reference

```bash
# Submit
python kaggle/kaggle_automate.py "message"

# Auto-update kernel metadata
python kaggle/sync_metadata.py

# Check Kaggle API
kaggle competitions submissions -c neurips-open-polymer-prediction-2025 --csv
```

---

**Status:** Production ready | **Last Updated:** Oct 30, 2025 | **Score:** 0.085 (Private)
