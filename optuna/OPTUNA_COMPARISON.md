# Optuna Model Comparison: Random Forest vs XGBoost

## Running Both Optimizations

### Random Forest (Running now)
```bash
python optuna_tune_rf.py
```

### XGBoost (Run after or in parallel)
```bash
# Option 1: Run after RF finishes
python optuna_tune_xgb.py

# Option 2: Run in parallel (different terminal)
# Terminal 1:
python optuna_tune_rf.py

# Terminal 2 (new terminal):
python optuna_tune_xgb.py
```

## What's Being Optimized

### Random Forest Hyperparameters
- `n_estimators`: 300-800 (number of trees)
- `max_depth`: 10-25 (tree depth)
- `min_samples_split`: 2-10
- `min_samples_leaf`: 1-5
- `max_features`: sqrt, log2, 0.3, 0.5, 0.7

### XGBoost Hyperparameters (More complex)
- `n_estimators`: 300-800
- `max_depth`: 4-12
- `learning_rate`: 0.01-0.2 (log scale)
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.5-1.0
- `colsample_bylevel`: 0.5-1.0
- `min_child_weight`: 1-10
- `reg_alpha`: 1e-8 to 10 (L1 regularization)
- `reg_lambda`: 1e-8 to 10 (L2 regularization)
- `gamma`: 1e-8 to 10 (min loss reduction)

## Expected Results

### Current Manual Testing
| Model | Score | Config |
|-------|-------|--------|
| Random Forest v53 | **0.07874** | depth=15, n=500, max_features='sqrt' |
| XGBoost v52 | 0.07965 | depth=8, n=500, subsample=0.9, colsample=0.7 |

### With Optuna (Expected)
| Model | Expected wMAE | Why |
|-------|---------------|-----|
| Random Forest | 0.073-0.077 | Already best, might find better configs |
| XGBoost | 0.072-0.078 | More hyperparameters to tune = more potential |

## Output Files

### Random Forest
- `optuna_polymer_rf.db` - Study database
- `optuna_best_configs_<timestamp>.json` - Top 5 configs
- `optuna_optimization_history.html` - Progress chart
- `optuna_param_importances.html` - Which params matter

### XGBoost
- `optuna_polymer_xgb.db` - Study database
- `optuna_xgb_best_configs_<timestamp>.json` - Top 5 configs
- `optuna_xgb_optimization_history.html` - Progress chart
- `optuna_xgb_param_importances.html` - Which params matter

## Comparing Results

After both finish, compare:

```python
import json

# Load RF results
with open('optuna_best_configs_<RF_timestamp>.json') as f:
    rf_configs = json.load(f)

# Load XGB results  
with open('optuna_xgb_best_configs_<XGB_timestamp>.json') as f:
    xgb_configs = json.load(f)

print("Random Forest Best:", rf_configs[0]['wmae'])
print("XGBoost Best:", xgb_configs[0]['wmae'])
```

## Testing Strategy

### Phase 1: Test Best from Each Model (2 submissions)
1. **v56**: Best Random Forest config from Optuna
2. **v57**: Best XGBoost config from Optuna

### Phase 2: Test Runner-ups (2-4 submissions)
- If v56 > v53 (0.07874): Try RF Rank 2-3
- If v57 > v52 (0.07965): Try XGB Rank 2-3

### Phase 3: Ensemble (Optional)
- Combine predictions from best RF + best XGB
- Average or weighted average based on validation scores

## Key Insight: Focus on Tc and FFV

Both optimizations use **competition wMAE** which heavily weights:
- **Tc** (weight=3.01) - Rarest property
- **FFV** (weight=1.51) - Medium
- **Density** (weight=1.57) - Medium

Check per-property MAE in results - configs with better Tc/FFV should score higher on Kaggle!

## Monitoring Progress

### Random Forest
```bash
# Watch the database (in new terminal)
watch -n 5 'sqlite3 optuna_polymer_rf.db "SELECT number, value FROM trials ORDER BY value LIMIT 5"'
```

### XGBoost
```bash
watch -n 5 'sqlite3 optuna_polymer_xgb.db "SELECT number, value FROM trials ORDER BY value LIMIT 5"'
```

## Expected Timeline

- **RF optimization**: 20-40 minutes (running now)
- **XGB optimization**: 20-40 minutes
- **Total**: ~1 hour if sequential, ~40 min if parallel

Then 2-5 Kaggle submissions (~3 min each) to test best configs.

---

**Prediction**: One of these optimized configs will beat your current best (0.07874)! 🎯

