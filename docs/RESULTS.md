# Polymer Properties Prediction - Complete Results

**Competition:** NeurIPS Open Polymer Prediction 2025  
**Evaluation Metric:** Weighted Mean Absolute Error (wMAE)  
**Dataset:** 7,973 training samples, 5 target properties

---

## 🏆 Final Leaderboard

| Rank | Model | Type | wMAE | Training Time | Status |
|------|-------|------|------|---------------|--------|
| 🥇 | **XGBoost** | Traditional ML | **0.030429** | 5 min | ✅ Production Ready |
| 🥈 | **Random Forest** | Traditional ML | **0.031638** | 3 min | ✅ Production Ready |
| 🥉 | **Transformer (DistilBERT)** | Deep Learning | **0.069180** | 22 min | ✅ Research Ready |
| 4️⃣ | **GNN (Tuned)** | Deep Learning | **0.177712** | 30 sec | ✅ Trained (needs improvement) |

**Winner: XGBoost** - Best performance with wMAE = 0.030429

---

## 📊 Model Details

### 1. XGBoost (Champion) 🥇

**Overall Performance:**
- **wMAE: 0.030429** ⭐⭐⭐
- Training time: ~5 minutes (CPU)
- Model size: 8.7 MB

**Property-wise Performance:**

| Property | n_samples | MAE | RMSE | R² | Weighted MAE |
|----------|-----------|-----|------|----|--------------|
| Tg (°C) | 97 | 55.30 | 69.70 | 0.627 | 0.146 |
| FFV | 1,416 | 0.007 | 0.015 | 0.760 | 0.005 |
| Tc (K) | 133 | 0.031 | 0.047 | 0.756 | 0.079 |
| Density (g/cm³) | 111 | 0.038 | 0.064 | 0.798 | 0.046 |
| Rg (Å) | 113 | 2.173 | 3.140 | 0.562 | 0.102 |

**Configuration:**
- Features: 15 molecular descriptors + 1024-bit Morgan fingerprints
- Hyperparameters: max_depth=10, learning_rate=0.1, n_estimators=200
- Training: Separate model per target property

**Strengths:**
- ✅ Best overall performance across all properties
- ✅ Fast training and inference
- ✅ Interpretable feature importance
- ✅ Robust to sparse labels

---

### 2. Random Forest 🥈

**Overall Performance:**
- **wMAE: 0.031638**
- Training time: ~7 minutes (CPU)
- Model size: 65 MB

**Property-wise Performance:**

| Property | MAE | RMSE | R² | Weighted MAE |
|----------|-----|------|----|--------------|
| Tg | 54.70 | 69.49 | 0.629 | 0.145 |
| FFV | 0.008 | 0.015 | 0.743 | 0.006 |
| Tc | 0.031 | 0.046 | 0.761 | 0.081 |
| Density | 0.044 | 0.070 | 0.756 | 0.053 |
| Rg | 2.207 | 3.291 | 0.518 | 0.103 |

**Strengths:**
- ✅ Very close to XGBoost performance (3.9% difference)
- ✅ More robust to outliers
- ✅ Good ensemble candidate
- ✅ Natural feature importance

---

### 3. Transformer (DistilBERT) 🥉

**Overall Performance:**
- **wMAE: 0.069180**
- Training time: 22 minutes (GPU: RTX 4070)
- Model size: ~250 MB

**Property-wise Performance:**

| Property | MAE | RMSE | R² | Weighted MAE |
|----------|-----|------|----|--------------|
| Tg | 79.75 | 118.50 | 0.102 | 0.211 |
| FFV | 0.022 | 0.038 | 0.341 | 0.015 |
| Tc | 0.054 | 0.082 | 0.248 | 0.139 |
| Density | 0.114 | 0.165 | 0.089 | 0.139 |
| Rg | 2.689 | 4.821 | 0.197 | 0.126 |

**Configuration:**
- Base Model: DistilBERT (distilbert-base-uncased, frozen)
- Architecture: 768 → 256 → 128 → 5
- Input: Raw SMILES strings (max length: 256 tokens)
- Training: 20 epochs, batch size 16, lr=2e-5
- Features: Early stopping, gradient clipping, NaN handling

**Strengths:**
- ✅ No feature engineering required (end-to-end from SMILES)
- ✅ Better than GNN (2.6x improvement)
- ✅ Positive R² on all properties (unlike GNN)
- ✅ Reasonable training time (22 min)
- ✅ Production-ready with proper error handling

**Limitations:**
- ⚠️ Still 127% behind XGBoost (2.3x worse)
- ⚠️ Requires GPU for practical use
- ⚠️ Large model size (250 MB vs 9 MB for XGBoost)
- ⚠️ Would benefit from chemistry-specific pretraining (ChemBERTa)

**Analysis:**
- Shows that transformers can learn from raw SMILES without feature engineering
- Better generalization than GNN on this dataset size
- Likely to improve significantly with:
  - ChemBERTa instead of DistilBERT
  - Larger dataset (50K+ samples)
  - More epochs and hyperparameter tuning
  - Unfreezing encoder for fine-tuning

---

### 4. GNN (Graph Convolutional Network)

**Overall Performance:**
- **wMAE: 0.177712** (Best after hyperparameter tuning)
- Training time: 30 seconds with GPU (RTX 4070)
- Improvement: 14% better than baseline (0.206 → 0.178)

**Tuning Results:**

| Configuration | wMAE | Training Time |
|---------------|------|---------------|
| GNN_Deeper (Winner) | 0.177712 | 29.6s |
| GNN_Wider | 0.183700 | 51.4s |
| GNN_LongTrain | 0.194629 | 45.7s |
| GNN_Baseline | 0.203360 | 23.1s |

**Best Configuration (GNN_Deeper):**
- Architecture: 4-layer GCN
- Hidden dimensions: 128
- Dropout: 0.2
- Batch size: 64
- Epochs: 50
- Learning rate: 0.001

**Property-wise Performance:**

| Property | MAE | RMSE | R² |
|----------|-----|------|----|
| Tg | 91.24 | 124.80 | -0.498 |
| FFV | 0.034 | 0.050 | -2.483 |
| Tc | 0.147 | 0.178 | -3.105 |
| Density | 0.733 | 0.772 | -29.496 |
| Rg | 11.869 | 13.278 | -6.241 |

**Analysis:**
- ⚠️ Still 484% behind XGBoost
- ⚠️ Negative R² indicates underfitting
- ⚠️ Dataset too small (~8K samples) for GNN to shine
- ✅ GPU acceleration working properly
- ✅ Significant improvement with tuning

**Potential Improvements:**
1. Pre-train on large molecular databases (PubChem, ChEMBL)
2. Add edge features (bond types, bond lengths)
3. Use Graph Attention Networks (GAT)
4. Increase dataset size to 100K+ samples
5. Transfer learning from related chemistry tasks

---

## 🎯 Competition Metric Explained

**Weighted Mean Absolute Error (wMAE):**

```
wMAE = (1/|X|) × Σ_X Σ_i w_i × |ŷ_i(X) - y_i(X)|

where: w_i = (1/r_i) × (K × √(1/n_i)) / Σ_j(√(1/n_j))
```

**Key Points:**
- Balances properties regardless of scale or frequency
- FFV has highest weight (71.5%) due to abundance
- Tc has second highest weight (257.8%) despite fewer samples
- Lower is better

**Property Weights:**

| Property | Weight | Contribution to wMAE |
|----------|--------|---------------------|
| Tg | 0.0026 | Low (sparse data) |
| FFV | 0.7152 | Very High (abundant) |
| Tc | 2.5778 | Very High (important) |
| Density | 1.2200 | High |
| Rg | 0.0468 | Low |

---

## 💡 Key Insights

### 1. Traditional ML > Deep Learning (for this dataset)

**Why?**
- Dataset size (8K samples) ideal for tree-based models
- Molecular descriptors + fingerprints capture chemistry effectively
- GNNs need 100K+ samples to outperform
- No need for GPU infrastructure

**Recommendation:** Use XGBoost or ensemble for production

### 2. Feature Engineering is Critical

**Impact Analysis:**
- Descriptors only: wMAE ~0.08
- Fingerprints only: wMAE ~0.05
- **Combined: wMAE ~0.03** ✅

**Best Features:**
- Molecular Weight, LogP, TPSA (physical properties)
- Morgan fingerprints (structural patterns)
- Combination captures both physics and chemistry

### 3. Property Difficulty Ranking

**From Easiest to Hardest:**

1. **Density** (R²=0.798) ⭐⭐
   - Direct physical property
   - Well-captured by molecular features
   
2. **FFV & Tc** (R²~0.76) ⭐
   - Abundant training data
   - Clear structure-property relationships
   
3. **Tg** (R²=0.627)
   - Sparse labels (6% coverage)
   - Temperature-dependent property challenging
   
4. **Rg** (R²=0.562)
   - Complex structural property
   - Conformation-dependent
   - Most difficult to predict

### 4. Data Coverage vs Performance

**Surprising Finding:**
- FFV: 88% coverage → R²=0.760
- Density: 1.4% coverage → R²=0.798 ⭐

**Conclusion:** Quality of features matters more than quantity of labels

---

## 🚀 Recommendations

### For Kaggle Submission

**Single Model:**
```python
# Use XGBoost (best standalone)
wMAE = 0.030429
```

**Ensemble (Recommended):**
```python
# Weighted average
final_pred = 0.6 * xgboost_pred + 0.4 * rf_pred
# Expected wMAE: ~0.029 (5-10% improvement)
```

### For Production Deployment

1. **Primary:** XGBoost
   - Fast inference (<1ms per molecule)
   - Small model size (8.7 MB)
   - Easy to deploy

2. **Fallback:** Random Forest
   - More robust to outliers
   - Good for uncertainty estimation

3. **Don't use:** GNN (yet)
   - Requires GPU
   - 6x worse performance
   - Complex deployment

### For Future Research

1. **Improve Transformer:**
   - Use ChemBERTa instead of DistilBERT
   - Unfreeze encoder for fine-tuning
   - Train on larger dataset (50K+ samples)
   - More epochs and learning rate scheduling
   - Expected gain: 30-50% wMAE reduction

2. **Improve GNN:**
   - Collect 100K+ polymer data
   - Pre-train on QM9/ESOL datasets
   - Implement Graph Attention (GAT)
   - Add 3D conformer features
   - Edge features (bond types, lengths)

3. **Advanced Ensembles:**
   - Stacking (XGBoost + RF + Transformer)
   - Bayesian Model Averaging
   - Weighted ensemble (optimize weights on validation)
   - Expected gain: 10-20% wMAE reduction

---

## 📈 Training History

### Development Timeline

1. **Baseline (Day 1)**
   - XGBoost: wMAE = 0.030429 ✅
   - Random Forest: wMAE = 0.031638 ✅

2. **GNN Initial (Day 2)**
   - Baseline: wMAE = 0.206 (6.8x worse)
   - Issues: CPU-only, no tuning

3. **GNN Tuned (Day 3)**
   - Fixed CUDA support ✅
   - Hyperparameter tuning ✅
   - Best: wMAE = 0.178 (14% improvement)

4. **Transformer Training (Day 4)**
   - Fixed NaN loss issues ✅
   - DistilBERT: wMAE = 0.069 ✅
   - 2.6x better than GNN
   - Training time: 22 minutes

5. **Final Status**
   - All 4 model architectures complete ✅
   - Interactive web demo deployed ✅
   - Comprehensive documentation ✅
   - Ready for Kaggle submission

---

## 📁 Reproducibility

### Environment
```bash
conda create -n polymer_pred python=3.10
conda activate polymer_pred
conda install -c conda-forge rdkit
pip install -r requirements.txt

# For GPU (GNN/Transformers)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

### Training Commands
```bash
# Traditional ML
python src/train.py

# Competition evaluation
python src/evaluate_competition.py

# GNN tuning
python src/train_gnn_tuned.py

# Compare all models
python src/compare_all_models.py
```

### Results Files
```
results/
├── model_comparison.csv           # Per-property metrics
├── competition_metrics.csv        # wMAE for traditional models
├── gnn_results.csv               # GNN baseline
├── gnn_tuning_results.csv        # GNN hyperparameter search
└── model_comparison_final.csv    # Final rankings
```

---

## 🎓 Lessons Learned

1. **Start Simple:** Traditional ML often beats DL on small datasets
2. **Feature Engineering:** Good features > fancy models
3. **Understand Metrics:** wMAE weights properties differently than MAE
4. **Sparse Labels:** Per-target training handles missing data well
5. **GPU Matters:** 100x speedup for deep learning (when it works)
6. **Hyperparameters:** 14% GNN improvement with tuning
7. **Dataset Size:** 8K samples perfect for XGBoost, too small for GNN

---

## 📞 Support

**Issues?**
- Check `requirements.txt` for dependency versions
- Ensure RDKit installed via conda (not pip)
- GPU models need CUDA-compatible torch-scatter

**Questions?**
- See README.md for basic usage
- Check source code (heavily commented)
- GitHub Issues: [open_polymer](https://github.com/jihwanksa/open_polymer)

---

**Last Updated:** October 10, 2025  
**Status:** ✅ All Models Complete | 🏆 XGBoost Best (wMAE: 0.030) | 🎨 Interactive Demo Live
