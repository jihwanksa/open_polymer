# Optuna Hyperparameter Optimization Guide

## Quick Start

```bash
cd /Users/jihwan/Downloads/open_polymer

# Install dependencies
pip install optuna plotly kaleido scikit-learn pandas tqdm

# Run optimization (takes 20-40 minutes)
python optuna_tune_rf.py
```

## What It Does

1. **Loads local data** from `data/raw/` (no Kaggle needed!)
2. **Creates 21 chemistry features** (same as your notebook)
3. **Uses correct competition metric (wMAE)** with proper weights:
   - Tc (rarest): weight = 3.01 (highest priority!)
   - FFV: weight = 1.51
   - Density: weight = 1.57
   - Rg: weight = 0.03
   - Tg (most common): weight = 0.0007 (lowest priority)
4. **Runs 100 trials** testing different hyperparameter combinations
5. **Outputs top 5 configs** for Kaggle testing

## Example Output

```
OPTIMIZATION COMPLETE!
======================================================================

Best wMAE: 0.082456

Best hyperparameters:
  n_estimators: 650
  max_depth: 18
  min_samples_split: 4
  min_samples_leaf: 2
  max_features: sqrt

TOP 5 CONFIGURATIONS FOR KAGGLE TESTING
======================================================================

[Rank 1] Trial #47
  wMAE: 0.082456
  Hyperparameters:
    n_estimators: 650
    max_depth: 18
    min_samples_split: 4
    min_samples_leaf: 2
    max_features: sqrt
  Per-property MAE:
    Tg: 36.15
    FFV: 0.0101
    Tc: 0.0268  ← Most important!
    Density: 0.0447
    Rg: 2.25

[Rank 2] Trial #82
  wMAE: 0.083127
  ...
```

## Workflow

### Stage 1: Run Optuna Locally (Once)

```bash
# Takes 20-40 minutes
python optuna_tune_rf.py

# Creates:
# - optuna_best_configs_<timestamp>.json
# - optuna_optimization_history.html
# - optuna_param_importances.html
# - optuna_polymer_rf.db
```

### Stage 2: Test on Kaggle (3-5 times)

**For each top config:**

1. **Update notebook** with hyperparameters:
   ```python
   # v56: Optuna Rank 1
   model = RandomForestRegressor(
       n_estimators=650,      # From Optuna
       max_depth=18,          # From Optuna
       min_samples_split=4,   # From Optuna
       ...
   )
   ```

2. **Push and submit:**
   ```bash
   python kaggle/kaggle_automate.py "v56: Optuna Rank 1"
   ```

3. **Record score** and compare with local wMAE

4. **Try next config** if improvement found

## Understanding Results

### Optimization History
Open `optuna_optimization_history.html` to see:
- How wMAE improved over 100 trials
- Early trials explore randomly
- Later trials exploit best regions

### Hyperparameter Importance
Open `optuna_param_importances.html` to see:
- Which hyperparameters matter most
- Example: If `max_depth` has high importance, focus tuning there
- Low importance params can use default values

## Key Differences from Current Approach

| Aspect | Current (v53) | With Optuna |
|--------|--------------|-------------|
| **Trials** | 1 config tested | 100+ configs tested |
| **Selection** | Manual guess | Smart Bayesian search |
| **Metric** | Unweighted MAE | **Competition wMAE** ✅ |
| **Feedback** | Kaggle only (slow) | Local validation (fast) |
| **Time** | 3 min per test | 20 min for 100 tests |

## Expected Improvements

**Current best (v53):** 0.07874

**With Optuna optimizing for wMAE:**
- **Rank 1:** Likely 0.073-0.077 (better Tc/FFV focus)
- **Rank 2-3:** Close alternatives
- **Insight:** See which hyperparameters actually matter

## Troubleshooting

**If data not found:**
```bash
# Check data locations
ls data/raw/train.csv
ls data/raw/train_supplement/Tc_SMILES.csv
```

**If slow:**
- Reduce `n_trials` from 100 to 50 in `main()`
- Reduce `n_ensemble` from 3 to 1 in `objective()`

**If out of memory:**
- Reduce `n_estimators` upper bound from 800 to 500

## Advanced: Resume Optimization

```python
# If interrupted, resume from where it stopped
study = optuna.load_study(
    study_name='polymer_rf_optimization',
    storage='sqlite:///optuna_polymer_rf.db'
)
study.optimize(objective, n_trials=50)  # Continue for 50 more trials
```

## Advanced: Try Different Models

Want to optimize XGBoost or LightGBM too? Easy to modify:

```python
# In objective(), replace RandomForestRegressor with:
model = xgb.XGBRegressor(
    n_estimators=trial.suggest_int('n_estimators', 100, 1000),
    max_depth=trial.suggest_int('max_depth', 3, 10),
    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
    ...
)
```

---

**Ready to start?** Just run `python optuna_tune_rf.py` and let it optimize for 20-40 minutes! ⚡

