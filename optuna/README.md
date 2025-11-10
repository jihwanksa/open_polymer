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
| `RF_BEST_RESULTS.md` | **Current best RF results** ⭐ |
| `optuna_polymer_rf.db` | RF optimization database |
| `optuna_polymer_xgb.db` | XGB optimization database |
| `test_optuna_env.py` | Environment test script |

## Latest Results

### Random Forest (97/100 trials completed)
- **Best wMAE**: 0.0252
- **Config**: n=800, depth=25, max_features=4
- **Improvement**: 68% better than v53 (0.07874)
- **Status**: Ready to test on Kaggle! 🚀

See `RF_BEST_RESULTS.md` for full details.

### XGBoost
- **Status**: Ready to run
- **Command**: `python optuna_tune_xgb.py`

## Live Dashboard

View optimization progress in real-time:

```bash
# For Random Forest
optuna-dashboard sqlite:///optuna_polymer_rf.db

# For XGBoost (once started)
optuna-dashboard sqlite:///optuna_polymer_xgb.db --port 8081
```

Then open: http://127.0.0.1:8080

## Next Steps

1. **Test Optuna RF config on Kaggle** (v56)
   - See if wMAE=0.0252 translates to better Kaggle score
   
2. **Run XGBoost optimization** (optional)
   - Compare which model architecture works better

3. **Iterate based on results**
   - If RF Optuna config wins, try variations
   - If not, analyze what local wMAE doesn't capture

---

**Last Updated**: 2025-11-10

