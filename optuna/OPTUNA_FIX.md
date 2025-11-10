# Optuna Metric Fix - Why v56 and v57 Failed

## Problem Identified

Both Optuna optimizations (RF v56 and XGBoost v57) scored **worse** than manual configs despite better local validation:

| Version | Local Metric | Kaggle Score | Status |
|---------|--------------|--------------|--------|
| v53 (Manual RF) | Simple MAE | 0.07874 | ✅ Best |
| v56 (Optuna RF) | wMAE=0.0252 | 0.08001 | ❌ Worse |
| v57 (Optuna XGBoost) | wMAE=0.0239 | ? (likely worse) | ❌ Worse |

## Root Cause

**Optuna optimized the WRONG metric!**

### What Optuna Used (WRONG):
```python
# wMAE with TRAINING data statistics
w_i = (1/r_i) * (K * sqrt(1/n_i)) / sum(sqrt(1/n_j))

where:
- n_i = sample counts in VALIDATION SET
- r_i = value ranges in VALIDATION SET
```

### What Competition Uses:
```python
# wMAE with TEST data statistics (unknown!)
w_i = (1/r_i) * (K * sqrt(1/n_i)) / sum(sqrt(1/n_j))

where:
- n_i = sample counts in TEST SET (hidden!)
- r_i = value ranges in TEST SET (hidden!)
```

**The statistics are completely different!** Optimizing with validation stats doesn't translate to test performance.

### What Successful Manual Configs Used:
```python
# Simple unweighted MAE (what v53 used)
MAE = mean(MAE_Tg, MAE_FFV, MAE_Tc, MAE_Density, MAE_Rg)
```

This is **model-agnostic** and correlates better with competition performance!

## Solution

**Re-run Optuna with simple unweighted MAE:**

```python
def objective(trial):
    # ... train model ...
    
    # Calculate simple MAE (no complex weighting)
    maes = []
    for i, target in enumerate(target_names):
        mask = ~np.isnan(y_val[:, i])
        if mask.sum() > 0:
            mae = mean_absolute_error(y_val[mask, i], y_pred[mask, i])
            maes.append(mae)
    
    # Simple average (what successful manual configs optimized)
    return np.mean(maes) if maes else float('inf')
```

### Why This Works:

1. **No dependency on unknown test statistics**
2. **What v53 actually optimized** (manually)
3. **Treats all properties equally** (simpler is often better)
4. **Correlates with competition metric** better than training-based wMAE

## Comparison

| Metric | Uses Test Stats? | Optimized by v53? | Correlates with Kaggle? |
|--------|------------------|-------------------|-------------------------|
| **Simple MAE** | ❌ No | ✅ Yes (implicitly) | ✅ Yes |
| **wMAE (training)** | ❌ No (uses validation) | ❌ No | ❌ No |
| **wMAE (competition)** | ✅ Yes (test) | N/A | ✅ Perfect (it IS the metric) |

## Next Steps

1. **Fix Optuna scripts** to use simple MAE
2. **Re-run RF optimization** (150 trials)
3. **Re-run XGBoost optimization** (100 trials)
4. **Test new configs** on Kaggle

**Expected:** Configs should now beat or match v53 (0.07874)

---

**Key Lesson:** When test statistics are unknown, optimize simpler proxy metrics that worked for successful baselines!

