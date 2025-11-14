# Quick Start: AutoGluon Production Training

## ⚠️ Important Update

**The training script now uses the FULL data pipeline from `best.ipynb`:**
- Original training: 7,973 samples
- + External Tc dataset
- + External Tg dataset
- + PI1070 (Density + Rg)
- + LAMALAB Tg dataset
- + 50K pseudo-labels (BERT + Uni-Mol + AutoGluon ensemble)
- **Total: ~60K+ training samples**

This matches exactly what `best.ipynb` uses, so AutoGluon can compete fairly with our current best model (v85).

---

## The Simplest Path

### 1. Run AutoGluon Training (~30 minutes to 2 hours)
```bash
cd /Users/jihwan/Downloads/open_polymer
conda activate pseudolabel_env
python AutoGluon/train_autogluon_production.py --preset medium_quality --time_limit 1800
```

That's it! This will:
- ✅ Load all augmented training data (matching best.ipynb)
- ✅ Extract 21 chemistry features from 60K+ SMILES
- ✅ Train AutoGluon models for each of 5 properties
- ✅ Save models to `models/autogluon_production/`
- ✅ Output feature importance rankings
- ✅ **Use CPU-only mode (no MPS hanging!)**

### 2. What You'll See
```
AUTOGLUON PRODUCTION TRAINING WITH FULL DATA AUGMENTATION
================================================================================

STEP 1: Loading original training data...
✅ Loaded 7973 original samples

STEP 2: Canonicalizing SMILES...
✅ 7973 samples after canonicalization

STEP 3: Loading external Tc dataset...
✅ Added 875 Tc samples

[... more augmentation steps ...]

STEP 8: Extracting 21 chemistry features...
✅ Final training data: 60012 samples with 21 features

STEP 9: Training AutoGluon models...
🔧 Training AutoGluon for Tg...
  Training samples: 8500+
  ✅ Model trained and saved to models/autogluon_production/Tg

[... for each of 5 properties ...]

✅ AUTOGLUON PRODUCTION TRAINING COMPLETE!
```

---

## Understanding the Data Pipeline

### Match best.ipynb Exactly

The training script replicates **all** data augmentation from `best.ipynb`:

```
Original Data (7,973)
    ↓
+ External Tc (875)
+ External Tg (7,000+)
+ PI1070 Density/Rg (1,000+)
+ LAMALAB Tg (2,000+)
+ Pseudo-Labels (50,000)
    ↓
Total: ~60,000+ training samples
```

Each sample has:
- SMILES string
- Up to 5 properties (some sparse)
- 21 chemistry-based features

### Why This Matters

**Before:** AutoGluon trained only on 7,973 samples (unfair comparison)
**Now:** AutoGluon trains on 60K+ samples (fair comparison with v85)

This means AutoGluon now has the same advantages as our best Random Forest model!

---

## Expected Performance

| Component | v85 (Current) | AutoGluon (New) |
|-----------|---------------|-----------------|
| Training samples | 60K+ | 60K+ (same!) |
| Features | 21 chemistry | 21 chemistry (same!) |
| Optimization | Manual tuning | Automated search |
| Ensemble | 5× RF | Mixed models (AutoML) |
| Expected score | 0.07533 | ? (We'll find out!) |

---

## Troubleshooting

### "Process hanging or slow"
This is normal! AutoGluon is searching:
- Multiple algorithms (RF, XGBoost, LightGBM, NN)
- Hyperparameter combinations
- Ensemble strategies

**Expected times:**
- `--preset fast` (300 sec): ~5-10 min total
- `--preset medium_quality` (1800 sec): ~30 min - 1 hour
- `--preset high_quality` (3600+ sec): 1-3 hours

### "Out of memory"
Reduce time limit:
```bash
python AutoGluon/train_autogluon_production.py --preset fast --time_limit 300
```

### "ModuleNotFoundError: autogluon"
Install it:
```bash
conda activate pseudolabel_env
pip install autogluon
```

---

## Next Steps

1. **Run training** (takes 30 min - 2 hours):
   ```bash
   python AutoGluon/train_autogluon_production.py
   ```

2. **Check feature importance**:
   ```bash
   cat models/autogluon_production/feature_importance.json
   ```

3. **Compare with v85**:
   - v85 score: **0.07533** (Private)
   - AutoGluon score: TBD (will know after Kaggle submission)

4. **Decision**:
   - If AutoGluon ≥ 0.075 → Use it!
   - If AutoGluon < 0.075 → Stick with v85

---

**Estimated time to completion:** 1-2 hours | **Difficulty:** Easy (just run the script!)
