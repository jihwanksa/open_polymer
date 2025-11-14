## AutoGluon Integration for Pseudo-Label Ensemble

### Overview

AutoGluon is an **AutoML framework** that complements BERT and Uni-Mol for robust pseudo-label generation:

```
┌─────────────────────────────────────────────────────────────────┐
│ THREE-MODEL ENSEMBLE FOR PSEUDO-LABELING                        │
└─────────────────────────────────────────────────────────────────┘

BERT (Deep Learning)        Uni-Mol (Molecular GNN)   AutoGluon (Tabular ML)
       ↓                             ↓                         ↓
SMILES → Embeddings      SMILES → Embeddings      SMILES → Features (21-dim)
(768-dim)                (512-dim)                        ↓
   ↓                         ↓                    ┌─────────────┐
Train heads             Train heads               │ RF          │
(10 epochs)             (50 epochs)               │ XGBoost     │
   ↓                         ↓                    │ LightGBM    │
50K predictions         50K predictions           │ NN ensemble │
   ↓                         ↓                    └─────────────┘
                                                        ↓
                                                   50K predictions
                                                        ↓
                           ENSEMBLE AVERAGE
                                ↓
                        Balanced, Robust
                        50K Pseudo-Labels 🎯
```

### Why Three Models?

| Aspect | BERT | Uni-Mol | AutoGluon |
|--------|------|---------|-----------|
| **Input** | SMILES (sequence) | SMILES (sequence) | Features (vector) |
| **Architecture** | Pre-trained transformer | Molecule-specific GNN | AutoML (RF+XGB+LGB+NN) |
| **Strength** | Language understanding | Molecular structure | Tabular pattern learning |
| **Diversity** | ✅ Different from ML | ✅ Different from DL | ✅ Different from embeddings |
| **Use Case** | General sequences | Molecular graphs | Structured features |

**Key Insight:** Each model learns from a different representation:
- BERT/Uni-Mol: Learn from raw SMILES (high-level patterns)
- AutoGluon: Learns from engineered features (explicit domain knowledge)

### AutoGluon Workflow

```
Step 1: Train AutoGluon Models
┌─────────────────────────────┐
│ 7,973 labeled samples       │
├─────────────────────────────┤
│ Extract 21 features         │
│ (smiles_length,             │
│  carbon_count,              │
│  branching_ratio, ...)      │
├─────────────────────────────┤
│ For each property:          │
│  AutoGluon tries:           │
│  - Random Forest            │
│  - XGBoost                  │
│  - LightGBM                 │
│  - Neural Network           │
│  - Ensemble of above        │
├─────────────────────────────┤
│ Selects best                │
│ Saves trained model         │
└─────────────────────────────┘

Step 2: Generate Pseudo-Labels
┌─────────────────────────────┐
│ 50,000 unlabeled SMILES     │
├─────────────────────────────┤
│ Extract 21 features         │
├─────────────────────────────┤
│ Feed to trained AutoGluon   │
├─────────────────────────────┤
│ Get predictions             │
│ Apply Tg transformation     │
└─────────────────────────────┘
```

### Installation

AutoGluon requires additional dependencies:

```bash
# Activate pseudolabel environment
conda activate pseudolabel_env

# Install AutoGluon (if not already installed)
pip install autogluon
pip install lightgbm xgboost  # For better ensemble
```

### Quick Start

**1. Train AutoGluon Models**
```bash
python pseudolabel/train_autogluon_models.py \
    --time_limit 600 \
    --preset medium
```

Options:
- `--time_limit`: Seconds per model (default: 600 = 10 min)
  - Fast: 120-300 seconds
  - Medium: 600 seconds (recommended)
  - High: 1200+ seconds (very slow)
- `--preset`: One of `fast`, `medium`, `high`, `best`
  - `fast`: Quick training, may be less accurate
  - `medium`: Good balance (recommended) ✅
  - `high`: Tries more models
  - `best`: Exhaustive search (very slow)

**Time Expectation:**
```
Tg:      ~10 min (500 samples)
FFV:     ~10 min (7030 samples)
Tc:      ~10 min (737 samples)
Density: ~10 min (613 samples)
Rg:      ~10 min (614 samples)
─────────────────────────
Total:   ~50 minutes (5 properties × 10 min each)
```

**2. Generate Pseudo-Labels**
```bash
python pseudolabel/generate_with_autogluon.py
```

**Output:**
```
pseudolabel/pi1m_pseudolabels_autogluon.csv (6.5 MB)
50,001 rows (header + 50K samples)
Columns: SMILES, Tg, FFV, Tc, Density, Rg
```

### Key Differences from BERT/Uni-Mol

| Aspect | BERT/Uni-Mol | AutoGluon |
|--------|--------------|-----------|
| **Model type** | Neural Network | Machine Learning Ensemble |
| **Input** | Raw SMILES | Extracted features |
| **Features** | Learned embeddings (768/512 dims) | Hand-crafted (21 dims) |
| **Training** | Supervised learning | AutoML + hyperparameter tuning |
| **Speed** | Fast (~5-10 min) | Slower (~50 min) but automatic |
| **Output** | Embeddings then predictions | Direct predictions |
| **Control** | Fixed architecture | AutoGluon chooses best model |

### Ensemble Strategy

**Option 1: 2-Model Ensemble (BERT + Uni-Mol) ✅ CURRENT**
```python
# Already implemented
ensemble = (bert_preds + unimol_preds) / 2
# Used in v85 to get 0.07533 private score
```

**Option 2: 3-Model Ensemble (BERT + Uni-Mol + AutoGluon) 🚀 PROPOSED**
```python
ensemble = (bert_preds + unimol_preds + autogluon_preds) / 3
# Even more robust, captures different perspectives
```

**Option 3: Weighted Ensemble**
```python
# Give different weights based on model quality
ensemble = (0.4 * bert_preds + 0.4 * unimol_preds + 0.2 * autogluon_preds)
# Adjust weights based on validation performance
```

### When to Use Each Model

**Use BERT when:**
- ✅ You want stable, conservative predictions
- ✅ You need fast inference (~5 min for 50K)
- ✅ You want pre-trained model (no training needed)

**Use Uni-Mol when:**
- ✅ You want molecular-specific patterns
- ✅ You can tolerate higher variance
- ✅ You want to capture edge cases

**Use AutoGluon when:**
- ✅ You want to leverage domain features (21 chemistry features)
- ✅ You're willing to spend 50 minutes training
- ✅ You want automatic hyperparameter tuning
- ✅ You prefer interpretability (tabular models)

**Use All Three when:**
- ✅ You want maximum robustness
- ✅ Total time: ~1 hour setup + 10 min inference
- ✅ Expected improvement: More balanced predictions

### Example: Running Full Pipeline

```bash
conda activate pseudolabel_env

# Step 1: BERT (5 min)
python pseudolabel/train_bert_heads.py
python pseudolabel/generate_with_bert.py

# Step 2: Uni-Mol (6 min)
python pseudolabel/train_unimol_heads.py
python pseudolabel/generate_with_unimol.py

# Step 3: AutoGluon (50 min)
python pseudolabel/train_autogluon_models.py --time_limit 600
python pseudolabel/generate_with_autogluon.py

# Step 4: Ensemble all three (2 min)
python pseudolabel/ensemble_three_models.py

# Result: pi1m_pseudolabels_ensemble_3models.csv
```

### Performance Expectations

**Individual Models:**
```
BERT:      Tg_mean=160.19, Tg_std=7.27   (conservative)
Uni-Mol:   Tg_mean=224.51, Tg_std=112.02 (diverse)
AutoGluon: Tg_mean=???,    Tg_std=???    (tabular ML)
```

**2-Model Ensemble (Current v85):**
```
Tg_mean=192.35, Tg_std=56.13
Score: 0.07533 Private / 0.08139 Public 🥇
```

**3-Model Ensemble (Proposed):**
```
Tg_mean=????, Tg_std=????
Expected: Even more balanced and robust ✨
```

### Troubleshooting

**Issue: AutoGluon takes too long**
```bash
# Use faster preset
python pseudolabel/train_autogluon_models.py \
    --time_limit 300 \
    --preset fast
```

**Issue: AutoGluon runs out of memory**
```bash
# Reduce batch size or run one property at a time
# Edit train_autogluon_models.py to train specific properties
```

**Issue: Models not saved**
```bash
# Check models/autogluon_models/ directory exists
mkdir -p models/autogluon_models

# Verify training completed
ls -la models/autogluon_models/
```

### Next Steps

1. ✅ **BERT Pseudo-Labels** - Generated (pi1m_pseudolabels_bert.csv)
2. ✅ **Uni-Mol Pseudo-Labels** - Generated (pi1m_pseudolabels_unimol.csv)
3. ✅ **2-Model Ensemble** - Generated (pi1m_pseudolabels_ensemble_2models.csv)
4. 🚀 **AutoGluon Training** - Ready to implement
5. 🚀 **3-Model Ensemble** - Ready after AutoGluon
6. 🎯 **Final Model** - Train RF on best ensemble labels

### References

- AutoGluon: https://auto.gluon.ai/
- AutoML concept: Tuning hyperparameters automatically
- Ensemble learning: Combining multiple models for robustness

---

**Status:** Ready to implement | **Expected Benefit:** +0.5-1.5% score improvement over 2-model ensemble

