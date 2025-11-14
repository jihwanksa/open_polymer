# Source Code Directory

This directory contains all source code for training and evaluating polymer property prediction models.

## 🏆 Best Kaggle Notebook & Local Training

**`best.ipynb`** (in root directory)
- **Status:** 🥇 **TIED WITH 1ST PLACE on Kaggle!**
- **Score:** Private 0.07533 | Public 0.08139 (v85)
- **Model:** Random Forest Ensemble + 21 chemistry features + 50K pseudo-labels
- **Ready to use:** Your colleague can fork and run directly on Kaggle!

**`train_v85_best.py`** (this directory - local training)
- **Status:** Exactly replicates best.ipynb locally (v85 = 1st place!)
- **Score:** Trains model with 60K samples (including pseudo-labels)
- **Time:** ~50 seconds to train
- **Usage:** `python src/train_v85_best.py`

## 📁 Directory Structure

```
src/
├── models/                      # Model implementations
│   ├── __init__.py
│   ├── traditional.py          # XGBoost & Random Forest
│   ├── gnn.py                  # Graph Neural Networks
│   └── transformer.py          # DistilBERT-based model
│
├── data_preprocessing.py       # Feature extraction & data processing
├── train.py                    # Traditional ML training pipeline
├── train_v85_best.py          # Best RF model (1st place! replicates best.ipynb locally)
├── train_gnn_tuned.py         # GNN training with hyperparameters
└── train_transformer.py       # Transformer training pipeline
```

## 🔧 Core Modules

### 1. Data Processing (`data_preprocessing.py`)

**Purpose**: Load raw data, parse SMILES, and extract molecular features

**Key Classes:**
- `MolecularDataProcessor` - Main data processing pipeline

**Key Functions:**
```python
# Load and process datasets
load_and_process_data(train_path, test_path)

# Extract molecular descriptors
compute_molecular_descriptors(smiles)
# Returns: MolWt, LogP, TPSA, NumRotatableBonds, etc. (15 features)

# Generate Morgan fingerprints
compute_morgan_fingerprint(smiles, radius=2, n_bits=2048)
# Returns: Binary fingerprint vector

# Create combined features
create_descriptor_features(df)  # Descriptors
create_fingerprint_features(df) # Fingerprints
```

**Features Generated:**
- **Molecular Descriptors** (15): MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, etc.
- **Morgan Fingerprints** (2048): Circular molecular fingerprints (radius=2)

### 2. Model Implementations (`models/`)

#### Traditional ML (`models/traditional.py`)

**Classes:**
- `TraditionalMLModel` - Wrapper for XGBoost and Random Forest

**Key Methods:**
```python
model = TraditionalMLModel(model_type='xgboost')

# Train on data with sparse labels
metrics = model.train(X_train, y_train, X_val, y_val, target_names)

# Make predictions
predictions = model.predict(X_test)  # Shape: (n_samples, 5)

# Save/load models
model.save('models/xgboost_model.pkl')
model.load('models/xgboost_model.pkl')
```

**Hyperparameters:**
- **XGBoost**: n_estimators=500, max_depth=8, learning_rate=0.05
- **Random Forest**: n_estimators=300, max_depth=20

**Features:**
- Handles sparse labels (trains separate model per target)
- Filters NaN/Inf values automatically
- Returns comprehensive metrics (RMSE, MAE, R²)

#### Graph Neural Networks (`models/gnn.py`)

**Classes:**
- `GNNModel` - Graph convolutional network for molecular graphs
- `MoleculeGNN` - PyTorch model architecture

**Architecture:**
```python
Input: Molecular graph (atoms=nodes, bonds=edges)
├── GCNConv(in=9, out=128)
├── GCNConv(128, 256)
├── GCNConv(256, 256)
├── GCNConv(256, 128)
├── Global pooling (mean + max)
└── MLP(256 → 128 → 64 → 5)
```

**Node Features** (16) - Enhanced with RDKit Chemistry:
- Atom type (C, N, O, F, P, S, H) - 7 binary features
- Connectivity (degree / 4.0)
- Formal charge (normalized / 4.0)
- Aromaticity (float)
- Hydrogen count (/ 4.0)
- Explicit valence (/ 8.0)
- Ring membership (binary)
- Hybridization: SP, SP2, SP3 (3 binary features)
- **Improvement**: 9 → 16 dimensions captures richer chemistry

**Edge Features** (6) - NEW! Bond properties:
- Bond type: Single, Double, Triple (3 binary features)
- Bond aromaticity (binary)
- Ring membership (binary)
- **Impact**: Helps GNN learn bond chemistry patterns

**Performance Improvement:**
- Previous GNN best (basic): 0.177712 wMAE (GNN_Deeper)
- Current GNN best (RDKit-enhanced): 0.173055 wMAE (GNN_Wider) ✅ +2.6%
- Overall validation wMAE: 0.189640

**Property-wise Breakdown (GNN_Wider):**
| Property | Samples | MAE | wMAE | R² | Notes |
|----------|---------|-----|------|-----|-------|
| Tg (glass transition) | 87 | 88.82 | 0.259 | -0.41 | Low R² indicates room for improvement |
| FFV (free volume) | 1419 | 0.039 | 0.051 | -4.12 | Most abundant property, still challenging |
| Tc (crystallization) | 144 | 0.156 | 0.403 | -3.61 | Moderate performance |
| Density | 123 | 0.673 | 0.705 | -26.68 | Worst property, needs attention |
| Rg (radius) | 124 | 11.03 | 0.505 | -5.80 | Scale issue (Rg values are large) |

- **Key insight**: Intrinsic chemistry features > artificial graph-level summaries
- **Challenge**: Negative R² values suggest GNN may be better at ranking than absolute prediction

**Key Methods:**
```python
model = GNNModel(hidden_channels=128, dropout=0.3)

# Train with PyTorch Geometric
val_loss = model.train(train_loader, val_loader, epochs=30)

# Evaluate
val_loss, metrics = model.evaluate(val_loader)
```

**GPU Acceleration:**
- Automatically uses CUDA if available
- ~100x faster than CPU for graph operations

#### Transformer (`models/transformer.py`)

**Classes:**
- `TransformerModel` - DistilBERT-based SMILES encoder
- `TransformerMoleculeModel` - PyTorch architecture

**Architecture:**
```python
Input: SMILES string
├── DistilBERT tokenizer + encoder (frozen)
├── [CLS] token embedding → 768-dim
├── Linear(768 → 256) + ReLU + Dropout
├── Linear(256 → 128) + ReLU + Dropout
└── Linear(128 → 5)
```

**Key Methods:**
```python
model = TransformerModel(
    model_name='distilbert-base-uncased',
    num_targets=5,
    hidden_dim=256,
    dropout=0.2
)

# Prepare data
dataset = model.prepare_data(smiles_list, targets)

# Train
val_loss = model.train(
    train_dataset, val_dataset,
    epochs=20, batch_size=16, lr=2e-5
)
```

**Features:**
- Automatic NaN handling in sparse labels
- Gradient clipping (max_norm=1.0)
- Early stopping (patience=15)
- Learning rate scheduling (ReduceLROnPlateau)

### 3. Training Scripts

#### Traditional ML Training (`train.py`)

**Purpose**: Train XGBoost and Random Forest models

**Workflow:**
1. Load and preprocess data
2. Extract molecular descriptors + fingerprints
3. Train both models with cross-validation
4. Evaluate with competition metric (wMAE)
5. Save models and results

**Usage:**
```bash
python src/train.py
```

**Outputs:**
- `models/xgboost_model.pkl`
- `models/random_forest_model.pkl`
- `results/competition_metrics.csv`
- `results/model_comparison.png`

#### GNN Training (`train_gnn_tuned.py`)

**Purpose**: Train tuned Graph Neural Network

**Hyperparameters:**
```python
hidden_channels = 128
num_layers = 4
dropout = 0.3
batch_size = 32
learning_rate = 0.001
epochs = 30
```

**Usage:**
```bash
python src/train_gnn_tuned.py
```

**Outputs:**
- `models/gnn_tuned_model.pt`
- `results/gnn_results.csv`

**GPU Requirements:**
- Recommended: NVIDIA GPU with 4GB+ VRAM
- Falls back to CPU if GPU unavailable (slower)

#### Transformer Training (`train_transformer.py`)

**Purpose**: Train DistilBERT-based transformer

**Configuration:**
```python
model_name = 'distilbert-base-uncased'
epochs = 20
batch_size = 16
learning_rate = 2e-5
max_length = 256  # SMILES tokenization
```

**Usage:**
```bash
python src/train_transformer.py
```

**Outputs:**
- `models/transformer_model.pt`
- `results/transformer_results.csv`

**Note**: Training takes ~22 minutes on GPU (RTX 4070)

## 🚀 Quick Start

### 0. Setup Environment (First Time Only)

```bash
# Create conda environment with Python 3.10
conda create -n polymer python=3.10 -y
conda activate polymer

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import torch_geometric; print('✅ All dependencies OK')"
```

### 1. Train All Models

```bash
# Activate environment
conda activate polymer

# ⭐ BEST MODEL - Random Forest v85 (1st Place!) - 50 seconds
python src/train_v85_best.py

# GNN with RDKit-enhanced features - 5-10 minutes
python src/train_gnn_tuned.py

# Transformer Model - 20+ minutes on GPU
python src/train_transformer.py
```

**Recommended Order:**
1. **`train_v85_best.py`** ⭐ (fastest, BEST 1st place performance: 0.07533!)
2. `train_gnn_tuned.py` (medium time, interesting results)
3. `train_transformer.py` (longest, optional)

### Training Scripts Reference

| Script | Model | Time | Output | Features |
|--------|-------|------|--------|----------|
| `train_v85_best.py` | Random Forest Ensemble | ~50s | `models/random_forest_v85_best.pkl` | 21 chemistry + canon + pseudo |
| `train_gnn_tuned.py` | Graph Neural Networks | ~5-10m | `models/gnn_best_tuned.pt` | 16 node + 6 edge features (RDKit) |
| `train_transformer.py` | DistilBERT Transformer | ~20m | `models/transformer_model.pt` | SMILES tokenization (768-dim) |
| `train.py` | XGBoost/Random Forest | ~2m | `models/{xgb,rf}_model.pkl` | Molecular descriptors + fingerprints |

**When to use each:**
- **`train_v85_best.py`** ⭐ **BEST - 1st Place!** 🥇
  - Achieves 0.07533 private score (tied 1st place on Kaggle!)
  - 60K training samples with pseudo-labels
  - SMILES canonicalization for consistency
  - 21 chemistry features + Random Forest ensemble
  - **Use for all Kaggle submissions**
  
- **`train_gnn_tuned.py`**: Research/comparison, validates GNN feature engineering
  - RDKit-enhanced node/edge features show improvement
  
- **`train_transformer.py`**: Experimental, slow but good for ensemble
  - Can be combined with RF for better ensemble predictions
  
- **`train.py`**: Quick baseline, good for debugging
  - Fast to run, useful for CI/CD validation

### 2. Use Trained Models

```python
from src.models import TraditionalMLModel, GNNModel, TransformerModel
import pandas as pd

# Load XGBoost
xgb = TraditionalMLModel()
xgb.load('models/xgboost_model.pkl')

# Predict on new data
predictions = xgb.predict(X_new)
# Returns shape: (n_samples, 5) for [Tg, FFV, Tc, Density, Rg]
```

### 3. Custom Feature Extraction

```python
from src.data_preprocessing import MolecularDataProcessor

processor = MolecularDataProcessor()

# Single molecule
smiles = "*CC(*)CCCC"
descriptors = processor.compute_molecular_descriptors(smiles)
fingerprint = processor.compute_morgan_fingerprint(smiles)

# Batch processing
df = pd.read_csv('data.csv')
features = processor.create_descriptor_features(df)
```

## 📊 Evaluation Metrics

All models use the competition metric:

**Weighted Mean Absolute Error (wMAE)**:
```
wMAE = Σ(weight_i * MAE_i) / Σ(weight_i)

Weights:
- FFV: 1.3036 (highest)
- Tc:  2.5755
- Density: 1.0486
- Rg: 0.0458
- Tg: 0.0029
```

Additional metrics reported:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## 🔬 Data Handling

### Sparse Labels
- Not all samples have all 5 properties
- Coverage: Tg (6%), FFV (88%), Tc (10%), Density (8%), Rg (8%)
- **Strategy**: Train separate model per property, skip NaN values

### Train/Val Split
- **Train**: 80% (6,378 samples)
- **Val**: 20% (1,595 samples)
- Stratified by property availability

### Feature Preprocessing
- Remove NaN/Inf from features
- StandardScaler for descriptors
- Binary features for fingerprints

## 🐛 Debugging Tips

### Import Errors
```python
# If models don't import:
import sys
sys.path.append('/path/to/chem/src')
from models import TraditionalMLModel
```

### CUDA/GPU Issues
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Force CPU if needed
model = GNNModel(device='cpu')
```

### Memory Issues
```python
# Reduce batch size
model.train(..., batch_size=8)  # instead of 32

# Use gradient accumulation
# (not implemented, would need code modification)
```

## 📚 Dependencies

### Required
```
# Core
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0

# Chemistry
rdkit>=2022.09.1

# Deep Learning
torch>=2.0.0
torch-geometric>=2.3.0
transformers>=4.30.0
```

### Optional
```
matplotlib>=3.7.0  # For visualizations
seaborn>=0.12.0    # For plots
tqdm>=4.65.0       # Progress bars
```

## 🔧 Configuration

### Modify Hyperparameters

**Traditional ML**:
Edit `train.py`, lines 50-70

**GNN**:
Edit `train_gnn_tuned.py`, lines 20-30

**Transformer**:
Edit `train_transformer.py`, lines 60-90

### Change Model Architecture

**GNN layers**:
Modify `models/gnn.py`, `MoleculeGNN` class

**Transformer head**:
Modify `models/transformer.py`, `TransformerMoleculeModel` class

## 📝 Code Style

- **Format**: PEP 8
- **Docstrings**: Google style
- **Type Hints**: Used where helpful
- **Comments**: Explain "why", not "what"

## 🤝 Contributing

To add a new model:

1. Create `models/your_model.py`
2. Implement `train()` and `predict()` methods
3. Add to `models/__init__.py`
4. Create `train_your_model.py`
5. Update this README

## 📄 License

MIT License - See project root LICENSE file

---

**Last Updated**: October 2025  
**Maintainer**: Jihwan Oh

