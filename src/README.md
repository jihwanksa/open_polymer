# Source Code Directory

This directory contains all source code for training and evaluating polymer property prediction models.

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

**Node Features** (9):
- Atom type (C, N, O, etc.)
- Atomic number
- Degree
- Formal charge
- Hybridization
- Aromaticity
- Ring membership

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

### 1. Train All Models

```bash
# Setup environment
conda activate polymer_pred

# Traditional ML (fastest)
python src/train.py

# Deep Learning
python src/train_gnn_tuned.py      # Requires: PyTorch Geometric
python src/train_transformer.py    # Requires: Transformers
```

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

