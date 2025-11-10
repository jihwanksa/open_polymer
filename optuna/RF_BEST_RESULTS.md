# Random Forest Optuna Optimization Results

## Summary
- **Trials Completed**: 97/100
- **Best wMAE**: 0.025234
- **Improvement over v53**: 68% better (0.07874 → 0.0252)

## Best Configuration (Trial #95, #96, #94)

```python
model = RandomForestRegressor(
    n_estimators=800,        # ← Increased from 500
    max_depth=25,            # ← Increased from 15
    min_samples_split=3,     # ← Decreased from 5
    min_samples_leaf=1,      # ← Decreased from 2
    max_features=4,          # ← ~19% of 21 features (was 'sqrt' ≈ 4.6)
    random_state=42,
    n_jobs=-1
)
```

## Key Findings

1. **Deeper trees work better** (25 vs 15)
   - More capacity to learn polymer structure relationships
   
2. **More trees help** (800 vs 500)
   - Better ensemble averaging

3. **Less restrictive leaf requirements**
   - min_samples_leaf=1 (was 2)
   - Allows finer-grained predictions

4. **Similar feature sampling** 
   - max_features=4 ≈ 19% of features
   - Close to 'sqrt'(21) ≈ 4.6

## Comparison with Manual Testing

| Method | Config | wMAE (Local) |
|--------|--------|--------------|
| **Optuna** | depth=25, n=800, max_feat=4 | **0.0252** |
| Manual v53 | depth=15, n=500, max_feat='sqrt' | 0.0787 |

**68% improvement with systematic hyperparameter search!**

## Next Steps for Kaggle

### v56: Test Optuna Best Config

Update notebook cell with:

```python
class RobustRandomForestModel:
    ...
    model = RandomForestRegressor(
        n_estimators=800,        # Optuna best
        max_depth=25,            # Optuna best
        min_samples_split=3,     # Optuna best
        min_samples_leaf=1,      # Optuna best
        max_features=4,          # Optuna best (integer, not 'sqrt')
        random_state=42 + i * 10 + j,
        n_jobs=-1
    )
```

Then:
```bash
python kaggle/kaggle_automate.py "v56: Optuna RF best (depth=25, n=800)"
```

## Expected Kaggle Score

If local wMAE correlates with Kaggle:
- Current best: 0.07874 (v53)
- Expected: ~0.02-0.03 (if correlation holds)
- **Could be a massive improvement!** 🎯

---

**Generated**: 2025-11-10  
**Database**: optuna_polymer_rf.db  
**Study**: polymer_rf_optimization

