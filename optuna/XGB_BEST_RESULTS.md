# XGBoost Optuna Optimization Results

## Summary
- **Trials Completed**: 86/100
- **Best wMAE**: 0.023861 (Trial #82)
- **Better than RF Optuna**: 0.0239 vs 0.0252 (5.4% better locally)

## Best Configuration (Trial #82)

```python
model = xgb.XGBRegressor(
    n_estimators=650,
    max_depth=12,
    learning_rate=0.0646,
    subsample=0.6861,
    colsample_bytree=0.9526,
    colsample_bylevel=0.5477,
    min_child_weight=5,
    gamma=0.000114,
    reg_alpha=9.03e-06,
    reg_lambda=2.48e-08,
    objective='reg:absoluteerror',
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
```

## Key Findings

1. **Deeper but narrower trees**
   - max_depth=12 (vs default 6)
   - Moderate learning_rate=0.065
   
2. **Aggressive subsampling**
   - subsample=0.686 (68.6% of samples per tree)
   - colsample_bytree=0.953 (95.3% of features per tree)
   - colsample_bylevel=0.548 (54.8% of features per level)

3. **Minimal regularization**
   - reg_alpha=9.03e-06 (very small L1)
   - reg_lambda=2.48e-08 (very small L2)
   - gamma=0.000114 (minimal split loss)

4. **Conservative splits**
   - min_child_weight=5 (requires 5+ samples per leaf)

## Comparison: XGBoost vs Random Forest (Optuna)

| Metric | XGBoost (Trial #82) | Random Forest (Trial #145) |
|--------|---------------------|----------------------------|
| **Local wMAE** | **0.0239** ✅ | 0.0252 |
| **n_estimators** | 650 | 600-800 |
| **max_depth** | 12 | 25 |
| **Kaggle Score** | TBD (v57) | 0.08001 (v56) ❌ |

## Kaggle Submission

### v57: XGBoost with Optuna Config
- **Pushed:** 2025-11-10
- **Configuration:** Trial #82 (wMAE=0.0239)
- **Status:** Awaiting score

**Prediction:** Given that RF Optuna (v56) scored worse than baseline v53 despite better local wMAE, XGBoost Optuna may also disappoint. The local validation improvement may not generalize to the competition test set.

## Lessons from RF Optuna (v56)

⚠️ **Warning:** RF Optuna achieved wMAE=0.0252 locally but scored **0.08001** on Kaggle (worse than v53's 0.07874).

**Why local validation failed:**
1. Training data distribution ≠ Competition test set
2. Aggressive hyperparameters may overfit to training patterns
3. Simpler, more conservative configs often generalize better

**Implications for v57:**
- XGBoost Optuna has even better local wMAE (0.0239)
- But this may also be overfitting to training distribution
- Conservative expectation: Similar or slightly worse than v53

## Next Steps

1. **Check v57 Kaggle score**
   - Compare to v53 baseline (0.07874)
   - If worse: Confirms local validation overfitting
   - If better: Local validation sometimes works!

2. **Analyze what local wMAE doesn't capture**
   - Distribution shift between train and test
   - Feature interactions that don't generalize
   - Why simpler configs (v53) beat optimized ones

3. **Alternative approach**
   - Manual hyperparameter tuning with intuition
   - Focus on robust, conservative settings
   - Test fewer, more principled configurations

---

**Generated**: 2025-11-10  
**Database**: optuna_polymer_xgb.db  
**Study**: polymer_xgb_optimization  
**Status**: v57 submitted, awaiting results 🤞

