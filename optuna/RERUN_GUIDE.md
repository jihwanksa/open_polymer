# Optuna Re-run Guide - Fixed Metric

## ✅ What Was Fixed

Changed from **complex wMAE (wrong)** to **simple MAE (correct)**

### Before (v56, v57 - FAILED):
```python
# Used wMAE with VALIDATION statistics
w_i = (1/r_i) * (K * sqrt(1/n_i)) / sum(sqrt(1/n_j))
wmae = sum(w_i * MAE_i)

Problem: validation stats ≠ test stats!
Result: v56 scored 0.08001 (worse than v53's 0.07874)
```

### After (NEW - CORRECT):
```python
# Simple unweighted MAE (what v53 used!)
simple_mae = mean(MAE_Tg, MAE_FFV, MAE_Tc, MAE_Density, MAE_Rg)

Benefit: Matches what successful manual configs optimized
Expected: Should beat or match v53 (0.07874)
```

## 🚀 How to Run

### Random Forest (150 trials, ~30-45 min)
```bash
cd /Users/jihwan/Downloads/open_polymer/optuna
python optuna_tune_rf.py
```

**Output:** `optuna_polymer_rf_simple_mae.db`

### XGBoost (150 trials, ~30-45 min)
```bash
cd /Users/jihwan/Downloads/open_polymer/optuna
python optuna_tune_xgb.py
```

**Output:** `optuna_polymer_xgb_simple_mae.db`

## 📊 What to Expect

### Old Results (wMAE - WRONG):
| Version | Local Metric | Kaggle | Status |
|---------|--------------|--------|--------|
| v56 (RF) | wMAE=0.0252 | 0.08001 | ❌ Worse |
| v57 (XGB) | wMAE=0.0239 | 0.08??? | ❌ Worse |

### New Results (Simple MAE - CORRECT):
| Version | Local Metric | Kaggle | Expected |
|---------|--------------|--------|----------|
| v58 (RF) | Simple MAE | TBD | ✅ Better! |
| v59 (XGB) | Simple MAE | TBD | ✅ Better! |

**Prediction:** Should beat or match v53 (0.07874) because we're now optimizing the same metric that made v53 successful!

## 📁 New Files

- `optuna_polymer_rf_simple_mae.db` - RF optimization with correct metric
- `optuna_polymer_xgb_simple_mae.db` - XGB optimization with correct metric
- Old `optuna_polymer_rf.db` and `optuna_polymer_xgb.db` are preserved for comparison

## ⏱️ Time Estimate

- **RF**: 150 trials × 15-20 sec/trial = ~35-45 minutes
- **XGB**: 150 trials × 15-20 sec/trial = ~35-45 minutes
- **Total**: ~1-1.5 hours for both

## 📝 After Optimization

1. Check best configs from new database
2. Update notebook with best hyperparameters
3. Push to Kaggle as v58 (RF) or v59 (XGB)
4. **Compare to v53 baseline (0.07874)**

## 🎯 Success Criteria

- **Local Simple MAE** should correlate with Kaggle score
- **Kaggle score ≤ 0.07874** (beats or matches v53)
- If successful: Confirms simple MAE is the right metric!
- If fails: Something else is different between local/Kaggle

---

**Ready to run!** Just execute the scripts above. 🚀

