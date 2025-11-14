# Generated Pseudo-Label Files Summary

## Overview

All pseudo-label files have been successfully generated using BERT and Uni-Mol models.

## Generated Files

### 1. BERT Predictions
**File:** `pseudolabel/pi1m_pseudolabels_bert.csv`
- **Size:** 6.9 MB
- **Samples:** 50,000 polymers
- **Properties:** Tg, FFV, Tc, Density, Rg
- **Model:** BERT (unikei/bert-base-smiles) with trained prediction heads
- **Generation Time:** ~5 minutes
- **Device:** MPS (Apple Silicon acceleration)

**Statistics:**
```
Tg:      Mean=160.19, Std=7.27  (conservative, stable)
FFV:     Mean=0.3609, Std=0.0232
Tc:      Mean=0.2230, Std=0.0481
Density: Mean=1.0548, Std=0.0864
Rg:      Mean=16.1305, Std=0.9381
```

**Characteristics:**
- ✅ Lower variance (more conservative)
- ✅ Stable predictions
- ✅ Fast generation

---

### 2. Uni-Mol Predictions
**File:** `pseudolabel/pi1m_pseudolabels_unimol.csv`
- **Size:** 5.1 MB
- **Samples:** 50,000 polymers
- **Properties:** Tg, FFV, Tc, Density, Rg
- **Model:** Uni-Mol (dptech/Uni-Mol2) with trained prediction heads
- **Generation Time:** ~2 minutes
- **Device:** MPS (Apple Silicon acceleration)

**Statistics:**
```
Tg:      Mean=224.51, Std=112.02  (diverse, captures variation)
FFV:     Mean=0.3677, Std=0.0112
Tc:      Mean=0.2539, Std=0.0517
Density: Mean=0.9845, Std=0.0794
Rg:      Mean=16.3080, Std=2.5086
```

**Characteristics:**
- ✅ Higher variance (captures diversity)
- ✅ Molecule-specific predictions
- ✅ Deterministic embeddings

---

### 3. Ensemble (BERT + Uni-Mol Average) ⭐ RECOMMENDED
**File:** `pseudolabel/pi1m_pseudolabels_ensemble_2models.csv`
- **Size:** 6.8 MB
- **Samples:** 50,000 polymers
- **Properties:** Tg, FFV, Tc, Density, Rg
- **Model:** Average of BERT and Uni-Mol predictions
- **Generation Time:** ~1 minute
- **Aggregation:** Simple mean of both model outputs

**Statistics:**
```
Tg:      Mean=192.35, Std=56.13  (balanced - between BERT and Uni-Mol)
FFV:     Mean=0.3643, Std=0.0129
Tc:      Mean=0.2384, Std=0.0354
Density: Mean=1.0197, Std=0.0587
Rg:      Mean=16.2193, Std=1.3376
```

**Why use ensemble?**
- ✅ **Balanced variance:** Not too conservative (BERT), not too diverse (Uni-Mol)
- ✅ **Robust:** Reduces individual model biases
- ✅ **Better generalization:** Combines BERT stability with Uni-Mol diversity
- ✅ **Used in v85:** Achieved 0.07533 private score (🥇 Tied 1st Place!)

---

## Comparison Summary

| Metric | BERT | Uni-Mol | Ensemble |
|--------|------|---------|----------|
| **File Size** | 6.9 MB | 5.1 MB | 6.8 MB |
| **Tg Mean** | 160.19 | 224.51 | 192.35 ✨ |
| **Tg Std** | 7.27 | 112.02 | 56.13 ✨ |
| **FFV Stability** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Diversity** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Gen. Time** | ~5 min | ~2 min | ~1 min |
| **Recommended** | ❌ | ❌ | ✅ **YES!** |

---

## Usage in Training

### Recommended: Use Ensemble Labels

```bash
# In train_v85_best.py or your training script:
pseudo_labels = pd.read_csv('pseudolabel/pi1m_pseudolabels_ensemble_2models.csv')

# Combine with original training data
train_data = pd.concat([original_train, pseudo_labels], ignore_index=True)

# Train Random Forest on augmented data (57,973 samples)
model.train(train_data, labels)

# Kaggle Result: 0.07533 Private / 0.08139 Public 🥇
```

### Individual Labels (Advanced)

Use BERT only:
```python
bert_labels = pd.read_csv('pseudolabel/pi1m_pseudolabels_bert.csv')
# For stable, conservative predictions
```

Use Uni-Mol only:
```python
unimol_labels = pd.read_csv('pseudolabel/pi1m_pseudolabels_unimol.csv')
# For capturing diversity
```

---

## Data Augmentation Impact

### Before Pseudo-Labels (v53)
```
Training samples: 7,973 (original only)
Private Score:   0.07874
Public Score:    0.10354
Samples/property ratio: ~10-800 (varies)
```

### After Pseudo-Labels (v85)
```
Training samples: 57,973 (7,973 original + 50,000 pseudo)
Private Score:   0.07533  ← 4.3% improvement! 🎉
Public Score:    0.08139
Samples/property ratio: ~1,200-11,500 (much better)
Sample increase: +626% total
```

---

## Quality Assurance

### Validation Approach
✅ Models trained on original 7,973 labeled samples  
✅ Predictions generated for 50,000 unlabeled SMILES  
✅ Ensemble aggregation for robustness  
✅ Tg transformation applied: (9/5) × Tg + 45  
✅ Final validation on test set shows improvement  

### Checks Performed
- [x] All 50,000 SMILES processed
- [x] No NaN values in critical properties
- [x] Tg transformation applied correctly
- [x] Distribution reasonably matches training data
- [x] Ensemble variance balanced between models

---

## Technical Details

### BERT Model
- **Source:** Hugging Face `unikei/bert-base-smiles`
- **Embedding Dim:** 768
- **Heads:** 5 (one per property)
- **Training:** 10 epochs, batch size 32, Adam (lr=0.001)
- **Device:** MPS (Apple Silicon)

### Uni-Mol Model
- **Source:** Hugging Face `dptech/Uni-Mol2` (84M variant)
- **Embedding Dim:** 512
- **Heads:** 5 (one per property)
- **Training:** 50 epochs, batch size 32, Adam (lr=0.001)
- **Device:** MPS (Apple Silicon)

### Ensemble
- **Method:** Element-wise mean of predictions
- **Normalization:** Uses StandardScaler (fitted on training data)
- **Tg Transform:** (9/5) × Tg + 45 (2nd place solution)

---

## Next Steps

1. ✅ **Pseudo-labels generated** (you are here!)
2. 📂 **Use in training:** See `src/train_v85_best.py`
3. 🚀 **Submit to Kaggle:** Expected score ~0.075 private
4. 🎯 **Iterate:** Different ensemble weights, add more models

---

## Commands to Regenerate

If you need to regenerate pseudo-labels:

```bash
conda activate pseudolabel_env

# Full pipeline
python pseudolabel/train_bert_heads.py
python pseudolabel/generate_with_bert.py
python pseudolabel/train_unimol_heads.py
python pseudolabel/generate_with_unimol.py
python pseudolabel/ensemble_bert_unimol.py

# Result: All CSV files regenerated
```

---

**Generated:** Nov 14, 2025 | **Status:** ✅ Complete | **Score Impact:** +4.3% (v85)
