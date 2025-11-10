# Optuna Hyperparameter Optimization

This folder contains Optuna optimization scripts and results for systematic hyperparameter tuning.

## Quick Start

```bash
cd /Users/jihwan/Downloads/open_polymer/optuna

# Random Forest optimization
python optuna_tune_rf.py

# XGBoost optimization  
python optuna_tune_xgb.py
```

## Files

| File | Purpose |
|------|---------|
| `optuna_tune_rf.py` | Random Forest optimization script |
| `optuna_tune_xgb.py` | XGBoost optimization script |
| `OPTUNA_GUIDE.md` | Complete usage guide |
| `OPTUNA_COMPARISON.md` | RF vs XGB comparison |
| `RF_BEST_RESULTS.md` | Random Forest optimization results |
| `XGB_BEST_RESULTS.md` | **XGBoost optimization results** ⭐ |
| `optuna_polymer_rf.db` | RF optimization database (150 trials) |
| `optuna_polymer_xgb.db` | XGB optimization database (86 trials) |
| `test_optuna_env.py` | Environment test script |

## Latest Results

### Random Forest (97/100 trials completed)
- **Best Local Validation wMAE**: 0.0252
- **Config**: n=800, depth=25, max_features=4
- **Kaggle Test Result (v56)**: 0.08001 (Private), 0.10388 (Public)
- **Status**: ❌ Worse than v53 baseline (0.07874) - Local validation didn't translate!

⚠️ **Key Lesson:** Local validation wMAE optimized on training data did NOT improve competition score. The Optuna hyperparameters likely overfit to the training distribution.

See `RF_BEST_RESULTS.md` for full details.

### XGBoost (86/100 trials completed)
- **Best Local Validation wMAE**: 0.0239 (better than RF!)
- **Config**: n=650, depth=12, lr=0.065, subsample=0.69
- **Kaggle Test Result (v57)**: Pending submission
- **Status**: 🔄 Testing on Kaggle

See `XGB_BEST_RESULTS.md` for full details.

## Live Dashboard

View optimization progress in real-time:

```bash
# For Random Forest
optuna-dashboard sqlite:///optuna_polymer_rf.db

# For XGBoost (once started)
optuna-dashboard sqlite:///optuna_polymer_xgb.db --port 8081
```

Then open: http://127.0.0.1:8080

## Lessons Learned & Fix

### Problem Identified ❌

**v56 & v57 used WRONG metric:**
- Optimized wMAE with **validation** statistics (n_i, r_i from validation set)
- Competition uses wMAE with **test** statistics (unknown!)
- Result: Local optimization didn't translate to Kaggle performance

### Solution Implemented ✅

**Now using SIMPLE MAE** (what v53 actually optimized):
```python
# Simple unweighted average of per-property MAEs
simple_mae = mean(MAE_Tg, MAE_FFV, MAE_Tc, MAE_Density, MAE_Rg)
```

**Why this works:**
1. No dependency on unknown test statistics
2. Matches what successful manual configs (v53) optimized
3. Treats all properties equally (simpler is better)

### Ready to Re-run 🚀

**New optimization scripts:**
- `optuna_tune_rf.py` - Fixed to use simple MAE
- `optuna_tune_xgb.py` - Fixed to use simple MAE

**Commands:**
```bash
cd /Users/jihwan/Downloads/open_polymer/optuna
python optuna_tune_rf.py    # RF: 150 trials, ~35-45 min
python optuna_tune_xgb.py   # XGB: 150 trials, ~35-45 min
```

**Expected:** Should beat or match v53 (0.07874) ✅

See `RERUN_GUIDE.md` and `OPTUNA_FIX.md` for details.

---

**Last Updated**: 2025-11-10

