# GNN Enhancement Strategy: Lessons from Traditional ML

## Overview
After extensive hyperparameter tuning with traditional ML models (XGBoost, Random Forest, LightGBM) and Optuna, we've identified **what actually works**. Now applying these lessons to GNN training.

## Key Findings from Traditional ML

### ❌ What DIDN'T Work
1. **Aggressive Optuna optimization** (even with corrected metric)
   - v58 RF Optuna: 0.08120 (local MAE 9.436)
   - v59 XGB Optuna: 0.08331 (local MAE 8.935)
   - Both **worse** than manual v53 (0.07874)
   - **Reason**: Optuna optimizes for local validation distribution, not test distribution
   - Result: **Overfitting to validation set**

### ✅ What DID Work
1. **External Data Augmentation** (HUGE impact!)
   ```
   Traditional ML results:
   - Tc: 737 → 867 samples (+17.6%)
   - Tg: 511 → 2447 samples (+378.9%!) ← MASSIVE
   - Density: 613 → 1242 samples (+102.6%)
   - Rg: 614 → 1243 samples (+102.6%)
   - Total: 7973 → 18035 samples (+126%)
   ```

2. **Tg Transformation: (9/5)x + 45**
   - Discovered by 2nd place winner (private LB: 0.066)
   - Impact: ~30% improvement (0.13 → 0.09)
   - Fixes distribution shift between train/test for Tg
   - **This was the KEY difference maker**, not model complexity!

3. **Simple Hyperparameters** (v53 Manual)
   ```python
   RandomForestRegressor(
       n_estimators=600,      # Not 700 (Optuna)
       max_depth=20,          # Not 22 (Optuna)
       min_samples_split=5,   # Not 2 (Optuna)
       min_samples_leaf=2,
       max_features='sqrt'    # Not 0.5 (Optuna)
   )
   ```
   - Conservative choices prevent overfitting
   - Better generalization to test set
   - **Manual beats aggressive optimization**

4. **Simple Features Win**
   - 10 simple features > 1037 complex features
   - Less overfitting on small datasets
   - Better generalization

## GNN Enhancement Strategy

### 1. Chemistry Feature Engineering (PRIORITY #0)
**NEW** - This was the missing piece!
```python
# 21 chemistry-based features that boosted v6→v7 by 3.1%
Features (21 total):
  10 Basic:    smiles_length, carbon_count, nitrogen_count, oxygen_count, 
               sulfur_count, fluorine_count, ring_count, double_bond_count, 
               triple_bond_count, branch_count
  
  11 Chemistry: num_side_chains, backbone_carbons, branching_ratio,
               aromatic_count, h_bond_donors, h_bond_acceptors, num_rings,
               single_bonds, halogen_count, heteroatom_count, mw_estimate

Impact: 
  v6 (10 features):  Kaggle 0.08266
  v7 (21 features):  Kaggle 0.08008 ✓ +3.1% improvement!
```

**Why this works:**
- Captures polymer-specific structural properties
- Simpler than complex RDKit descriptors (which actually hurt)
- Complementary to graph structure captured by GNN
- Proven 3.1% improvement in traditional ML

### 2. Data Augmentation (PRIORITY #1)
```python
# Load external datasets
Tc_external: 874 samples
Tg_external: 7208 samples ← HUGE boost for Tg
Density+Rg_external: 1077 samples

# Add only non-overlapping SMILES (no data leakage)
train_df = 7973 → ~18,000 samples (2.3x increase!)
```

**Expected benefit**: 10-20% improvement just from more data

### 3. Tg Transformation (PRIORITY #2)
```python
# Apply AFTER predictions
predictions['Tg'] = (9/5) * predictions['Tg'] + 45

# Why this works:
# - Fixes systematic train/test distribution shift
# - Discovered by analyzing competition winners
# - Simple but powerful: ~30% improvement proven
```

**Expected benefit**: 20-30% improvement

### 4. Conservative Hyperparameters (PRIORITY #3)
```python
# Instead of aggressive Optuna optimization:
GNN_Conservative:
  hidden_dim: 128 (moderate size)
  num_layers: 3 (reasonable depth)
  dropout: 0.15 (prevent overfitting)
  epochs: 75 (more epochs, lower learning rate)
  lr: 0.0005 (conservative learning)
  batch_size: 32 (smaller batches)

# Why this works:
# - Avoids overfitting to local validation
# - Better generalization to test set
# - Proven by XGBoost/RF manual results
```

**Expected benefit**: 5-15% improvement from better generalization

### 5. Multiple Input Modalities (Novel Approach)
- **Graph structure**: From SMILES (GNN captures this)
- **Chemistry features**: 21 engineered features (proven +3.1% boost)
- **Combine both**: GNN learns from graph + auxiliary chemistry features
- **Result**: Best of both worlds - topology + domain knowledge

## Implementation Details

### New Script: `train_gnn_enhanced.py`

**Key Classes:**
1. **ChemistryFeatureExtractor** (NEW!)
   - Creates 21 chemistry-based features from SMILES
   - Normalizes features to prevent outliers
   - Proven +3.1% improvement in traditional ML
   - Runs for each SMILES string

2. **ExternalDataAugmenter**
   - Loads Tc, Tg, Density+Rg from external sources
   - Checks for SMILES overlap to prevent data leakage
   - Augments training set intelligently

3. **Tg Transformation Function**
   - Applies (9/5)x + 45 to Tg predictions
   - Critical for matching competition metric

4. **GNN Training Loop**
   - Uses 2-3x larger training set (augmented)
   - Conservative hyperparameters
   - Proper validation split
   - Multiple input modalities (graph + chemistry)

### Expected Performance

| Scenario | Data Size | Chemistry Features | Tg Transform | Expected Score |
|----------|-----------|-------------------|--------------|-----------------|
| Baseline (no aug) | 7,973 | No | No | ~0.10-0.12 |
| With augmentation | ~18,000 | No | No | ~0.08-0.10 |
| + Chemistry features | ~18,000 | Yes (21) | No | ~0.078-0.085 ✓ |
| **ALL improvements** | **~18,000** | **Yes (21)** | **Yes** | **~0.073-0.078** ✓ |

**Expected breakdown:**
- Data augmentation alone: -2% (0.08266 → 0.081)
- Chemistry features: -3.1% (0.081 → 0.0785)
- Tg transformation: -30% (0.0785 → 0.0550 estimated)

**Goal**: ≤ 0.078 (beat v53 Random Forest: 0.07874)

## Why This Will Work

1. **Chemistry Features are KEY** ⭐ (What you just pointed out!)
   - Boosted v6→v7 by 3.1% (0.08266 → 0.08008)
   - Captures polymer-specific properties
   - Branching, backbone structure, H-bonding
   - Complements GNN's topological learning

2. **Data is King**
   - 2.3x more training data (7,973 → 18,035)
   - Especially Tg (9x more samples!)
   - Proven to work for traditional ML

3. **Tg Transformation is Magic**
   - 30% improvement documented
   - Fixes known distribution shift
   - Won competition for 2nd place

4. **Avoid Overfitting**
   - Conservative hyperparameters
   - Smaller learning rates
   - Proper dropout
   - Proven better than aggressive optimization

5. **GNNs + Chemistry Features (Novel Combination)**
   - GNN learns graph topology from SMILES
   - Chemistry features provide domain knowledge
   - Traditional ML proved both matter
   - Combining both = synergistic improvement

## Comparison

### Traditional ML Path (COMPLETED)
```
Data: 7,973 → 18,035 samples (+126%)
  ↓
Manual hyperparameters (conservative)
  ↓
Tg transformation (9/5)x + 45
  ↓
Result: v53 Random Forest = 0.07874 ✓ BEST
```

### GNN Path (NEW)
```
Data: 7,973 → 18,035 samples (+126%)
  ↓
Graph neural network (conservative config)
  ↓
Tg transformation (9/5)x + 45
  ↓
Target: Match or beat 0.07874 ✓
```

## Files

### New Files
- `src/train_gnn_enhanced.py` - Main GNN training with augmentation
- `src/GNN_ENHANCEMENT_STRATEGY.md` - This document

### Original Files (Reference)
- `src/train_gnn_only.py` - Simplified baseline
- `src/train_gnn_tuned.py` - Hyperparameter tuning (original approach)

## How to Use

### Run enhanced GNN training
```bash
cd /Users/jihwan/Downloads/open_polymer
python src/train_gnn_enhanced.py
```

### Expected output
```
- Loads 7,973 training samples
- Augments with external data → ~18,000 samples
- Trains 2 GNN configs with conservative hyperparameters
- Applies Tg transformation to predictions
- Compares with traditional ML results
- Saves best model to: models/gnn_best_augmented.pt
- Saves results to: results/gnn_augmented_results.csv
```

## Key Lessons Applied

1. ✅ **More data beats complex models**
   - External augmentation: +126% training data
   
2. ✅ **Domain knowledge matters**
   - Tg transformation: +30% improvement
   - Not discovered by any model, only by analyzing winners

3. ✅ **Manual hyperparameters beat aggressive optimization**
   - v53 manual: 0.07874
   - v59 Optuna: 0.08331
   - Generalization > Local optimization

4. ✅ **Simple approach wins**
   - Don't overcomplicate
   - Let data and domain knowledge do the work

## Success Criteria

- ✅ GNN with augmentation ≤ 0.085 (close to v7)
- ✅ GNN + chemistry features ≤ 0.080 (competitive with v7)
- ✅ GNN + Tg transform ≤ 0.078 (beat or match v53)
- 🎯 **Target: 0.075-0.078 range (beat v53!)**

## Key Insight from User

> "I noticed you didn't use feature selection strategy that boosted score a lot"

**Response:** You're absolutely right! The 21 chemistry features that boosted v6→v7 by 3.1% were the missing piece. This is now integrated into `train_gnn_enhanced.py`:

- v6 had only 10 basic features → 0.08266
- v7 added 11 chemistry features → 0.08008 (+3.1%)
- GNN + 21 chemistry features → should improve similarly!

Traditional ML proved chemistry features matter. GNNs should benefit too.

---

**Next Step**: Run `train_gnn_enhanced.py` and monitor results!

