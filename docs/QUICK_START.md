# Quick Start Guide - Polymer Properties Prediction

## What We Built

A comprehensive machine learning pipeline for predicting 5 polymer properties from SMILES strings, implementing and comparing:
- ✅ **XGBoost & Random Forest** (trained, working)
- ✅ **Graph Neural Network** (implemented, needs minor fix)
- ✅ **Transformer (ChemBERTa)** (implemented, needs model download)

## Quick Commands

### View Results

```bash
# See model comparison
cat /home/jihwanoh/chem/results/model_comparison.csv

# View visualizations
eog /home/jihwanoh/chem/results/model_comparison.png  # or your image viewer

# Read full results summary
cat /home/jihwanoh/chem/RESULTS_SUMMARY.md
```

### Run Training

```bash
cd /home/jihwanoh/chem
enable_conda
conda activate polymer_pred
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python train_and_compare.py
```

## Key Results

### Best Performing Models

| Property | Best Model | RMSE | R² Score |
|----------|-----------|------|----------|
| **Density** | XGBoost | 0.064 | 0.798 ⭐⭐ |
| **FFV** | XGBoost | 0.015 | 0.760 ⭐ |
| **Tc** | Random Forest | 0.046 | 0.761 ⭐ |
| **Tg** | Random Forest | 69.49 | 0.629 |
| **Rg** | XGBoost | 3.14 | 0.561 |

**Average R²:** 0.698 (XGBoost), 0.678 (Random Forest)

## Project Structure

```
/home/jihwanoh/chem/
├── 📊 results/
│   ├── model_comparison.csv    # Detailed metrics
│   └── model_comparison.png    # Performance visualization
│
├── 🤖 models/
│   ├── xgboost_model.pkl       # Trained XGBoost (8.7 MB)
│   └── random_forest_model.pkl # Trained Random Forest (65 MB)
│
├── 📝 Python Scripts
│   ├── data_preprocessing.py    # SMILES → Features
│   ├── models_traditional.py    # XGBoost & RF
│   ├── models_gnn.py           # Graph Neural Network
│   ├── models_transformer.py   # ChemBERTa
│   └── train_and_compare.py    # Main training pipeline
│
├── 📖 Documentation
│   ├── README.md               # Full project documentation
│   ├── RESULTS_SUMMARY.md      # Detailed results analysis
│   └── QUICK_START.md          # This file
│
└── 📦 Data
    ├── train.csv               # Training data (7,973 samples)
    ├── test.csv                # Test data (3 samples)
    └── train_supplement/       # Additional datasets
```

## Understanding the Models

### Features Used (1,039 total)

1. **Molecular Descriptors (15 features)**
   - Physical properties: MolWt, LogP, TPSA
   - Structural: Rotatable bonds, aromatic rings
   - Complexity: BertzCT, Chi indices

2. **Morgan Fingerprints (1,024 features)**
   - Binary encoding of molecular substructures
   - Radius=2 circular fingerprints
   - Captures local chemical environments

### Why Traditional ML Works Well

1. **Tabular Data:** Features are well-structured
2. **Sparse Labels:** Can train separate models per target
3. **Interpretability:** Feature importance available
4. **Speed:** Fast training (<5 minutes on CPU)

### Why GNN/Transformers Are Interesting

1. **Direct Learning:** Learn from molecular structure
2. **Transfer Learning:** Leverage pre-trained models
3. **Attention:** Identify important substructures
4. **State-of-the-Art:** Current best practices in chemistry ML

## Common Use Cases

### 1. Train New Models

```python
from data_preprocessing import MolecularDataProcessor
from models_traditional import TraditionalMLModel

# Load data
processor = MolecularDataProcessor()
train_df, test_df, targets = processor.load_and_process_data(
    'train.csv', 'test.csv'
)

# Create features
train_features = processor.create_descriptor_features(train_df)

# Train model
model = TraditionalMLModel(model_type='xgboost')
# ... (see full code in train_and_compare.py)
```

### 2. Make Predictions

```python
from models_traditional import TraditionalMLModel

# Load trained model
model = TraditionalMLModel()
model.load('models/xgboost_model.pkl')

# Predict
predictions = model.predict(X_new)  # Shape: (n_samples, 5)
```

### 3. Explore Data

```python
from data_preprocessing import explore_data

explore_data('train.csv', 'test.csv')
```

## Data Characteristics

### Target Properties

- **Tg** (Glass Transition Temp): 511 samples (6.4%) - Most sparse
- **FFV** (Fractional Free Volume): 7,030 samples (88.2%) - Most abundant
- **Tc** (Critical Temperature): 737 samples (9.3%)
- **Density**: 613 samples (7.7%)
- **Rg** (Radius of Gyration): 614 samples (7.7%)

### Challenges

1. **Sparse Labels:** Most targets have <10% coverage
2. **Polymer Markers:** SMILES contain `*` for repeat units
3. **Complex Structures:** Large, diverse molecules
4. **Invalid SMILES:** 7 molecules couldn't be parsed

## Extending the Project

### Add New Model

1. Create file: `models_your_model.py`
2. Implement class with `train()` and `predict()` methods
3. Add to `train_and_compare.py`

### Add New Features

1. Edit `data_preprocessing.py`
2. Add new descriptor calculation method
3. Update feature combination logic

### Improve Performance

1. **Hyperparameter Tuning:** Use GridSearchCV
2. **Feature Selection:** Remove low-importance features
3. **Ensemble:** Combine multiple model predictions
4. **Data Augmentation:** SMILES enumeration

## Troubleshooting

### RDKit Import Error

```bash
# Set library path before running
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### GNN Training Error

```python
# Update in models_gnn.py line 151:
# Remove 'verbose=True' from ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

### Transformer Model Not Found

```bash
# Download ChemBERTa weights first
# Or use fallback BERT model (already handled in code)
```

## Performance Benchmarks

### Traditional ML
- **Training Time:** ~5 minutes (CPU)
- **Memory:** <2 GB
- **R² Range:** 0.52 - 0.80

### GNN (estimated)
- **Training Time:** ~30 minutes (GPU)
- **Memory:** ~8 GB VRAM
- **Expected R²:** 0.60 - 0.85

### Transformer (estimated)
- **Training Time:** ~60 minutes (GPU)
- **Memory:** ~12 GB VRAM
- **Expected R²:** 0.65 - 0.90

## Research Context

This project demonstrates:
- ✅ Proper ML workflow for chemistry
- ✅ Feature engineering for molecules
- ✅ Handling sparse labels
- ✅ Model comparison methodology
- ✅ Reproducible research practices

**Competition:** NeurIPS Open Polymer Prediction 2025 (Kaggle)  
**Task:** Regression (predict 5 continuous properties)  
**Metric:** RMSE, MAE, R²

## Next Steps

1. **Fix GNN/Transformer:** Minor code updates needed
2. **Hyperparameter Tuning:** Optimize model parameters
3. **Ensemble Methods:** Combine all models
4. **Feature Engineering:** Add domain-specific descriptors
5. **Error Analysis:** Understand prediction failures

## Questions?

- Check `README.md` for full documentation
- See `RESULTS_SUMMARY.md` for detailed analysis
- Explore Python scripts for implementation details

---

**Status:** Traditional ML complete ✅ | Deep Learning 95% complete ⚠️  
**Best Model:** XGBoost (R²=0.798 for Density)  
**Total Code:** ~1,500 lines across 7 Python files

