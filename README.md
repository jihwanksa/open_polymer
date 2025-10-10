# Open Polymer: Polymer Properties Prediction 🧪

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning framework for predicting polymer properties from SMILES strings. Implements and compares multiple model architectures including traditional ML (XGBoost, Random Forest), Graph Neural Networks, and Transformers.

> **Competition:** [NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) (Kaggle)

## 🎯 Quick Results

**Competition Metric (wMAE - Lower is Better):**

| Rank | Model | Type | wMAE | Training Time | Status |
|------|-------|------|------|---------------|--------|
| 🥇 | **XGBoost** | Traditional ML | **0.030429** | 5 min | ✅ Best |
| 🥈 | **Random Forest** | Traditional ML | **0.031638** | 3 min | ✅ |
| 🥉 | **Transformer** | Deep Learning | **0.069180** | 22 min | ✅ |
| 4️⃣ | **GNN (Tuned)** | Deep Learning | **0.177712** | 30 sec | ✅ |

**Property-wise (Best Models):**

| Property | Best Model | RMSE | R² | MAE |
|----------|-----------|------|-----|-----|
| **Density** | XGBoost | 0.064 | **0.798** ⭐⭐ | 0.038 |
| **FFV** | XGBoost | 0.015 | **0.760** ⭐ | 0.007 |
| **Tc** | Random Forest | 0.046 | **0.761** ⭐ | 0.031 |
| **Tg** | Random Forest | 69.49 °C | 0.629 | 54.70 |
| **Rg** | XGBoost | 3.14 Å | 0.562 | 2.173 |

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Model Architectures](#-model-architectures)
- [Usage Examples](#-usage-examples)
- [Results](#-results)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

## ✨ Features

- **Multiple Model Architectures**
  - ✅ Traditional ML: XGBoost, Random Forest (wMAE: 0.030)
  - ✅ Transformers: DistilBERT-based (wMAE: 0.069)
  - ✅ Graph Neural Networks: 4-layer GCN with GPU acceleration (wMAE: 0.178)

- **Comprehensive Feature Engineering**
  - 15 molecular descriptors (MolWt, LogP, TPSA, etc.)
  - 1,024-bit Morgan fingerprints
  - Direct graph representation for GNNs

- **Robust Data Pipeline**
  - SMILES parsing with RDKit
  - Handles sparse labels (6-88% coverage per property)
  - Automated feature extraction and preprocessing

- **Extensive Evaluation**
  - Multiple metrics (RMSE, MAE, R²)
  - Cross-validation support
  - Detailed performance visualization

- **Interactive Web Demo**
  - Beautiful Gradio interface
  - Real-time molecule visualization
  - Instant property predictions
  - Application guidance for each polymer

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Anaconda or Miniconda (recommended for RDKit)
- CUDA-capable GPU (optional, for deep learning models)

### Setup by Platform

#### Linux / macOS

```bash
# Clone the repository
git clone https://github.com/jihwanksa/open_polymer.git
cd open_polymer

# Create conda environment
conda create -n polymer_pred python=3.10 -y
conda activate polymer_pred

# Install RDKit (must use conda)
conda install -c conda-forge rdkit -y

# Install other dependencies
pip install -r requirements.txt

# Set library path (Linux/macOS only)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

#### Windows

```bash
# Clone the repository
git clone https://github.com/jihwanksa/open_polymer.git
cd open_polymer

# Open Anaconda Prompt and run:
conda create -n polymer_pred python=3.10 -y
conda activate polymer_pred
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

**Note:** Windows users don't need to set `LD_LIBRARY_PATH`. All Python code is cross-platform compatible!

### Dependencies

- **Chemistry:** RDKit
- **Traditional ML:** scikit-learn, XGBoost
- **Deep Learning:** PyTorch, PyTorch Geometric, Transformers
- **Utilities:** pandas, numpy, matplotlib, seaborn

See [`requirements.txt`](requirements.txt) for full list.

## 🎓 Quick Start

### 1. Try the Interactive Demo 🎨

```bash
# Launch web interface
python app/app.py

# Opens at http://localhost:7861
# - Visualize molecules
# - Predict properties instantly
# - Get application guidance
```

**See [app/README.md](app/README.md) for detailed demo guide**

### 2. Train Models

```bash
# Traditional ML (5-7 min on CPU)
python src/train.py

# GNN (30 sec with GPU)
python src/train_gnn_tuned.py

# Transformer (22 min with GPU)
python src/train_transformer.py
```

### 3. View Results

```bash
# See comparison metrics
cat results/all_models_comparison.csv

# View visualizations
# results/model_comparison.png
```

### 4. Use Trained Models

```python
from src.models import TraditionalMLModel

# Load trained model
model = TraditionalMLModel(model_type='xgboost')
model.load('models/xgboost_model.pkl')

# Make predictions
predictions = model.predict(X_new)  # Shape: (n_samples, 5)
```

## 📁 Project Structure

```
open_polymer/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── app/                         # 🎨 Interactive Web Demo
│   ├── app.py                   # Gradio web interface
│   ├── README.md                # Demo usage guide
│   ├── LAUNCH_DEMO.md           # Detailed deployment guide
│   └── screenshot_*.png         # Interface screenshots
│
├── src/                         # Source code
│   ├── README.md                # Code documentation
│   ├── data_preprocessing.py    # Feature extraction
│   ├── train.py                 # Traditional ML training
│   ├── train_gnn_tuned.py       # GNN training
│   ├── train_transformer.py     # Transformer training
│   └── models/                  # Model implementations
│       ├── __init__.py
│       ├── traditional.py       # XGBoost & Random Forest
│       ├── gnn.py              # Graph Neural Network
│       └── transformer.py      # DistilBERT-based model
│
├── data/                        # Data directory
│   └── raw/                    # Original Kaggle data
│       ├── train.csv           # Training data (7,973 samples)
│       ├── test.csv            # Test data (3 samples)
│       ├── sample_submission.csv
│       └── train_supplement/   # Additional datasets
│
├── models/                      # Saved model checkpoints
│   ├── README.md                # Model specifications
│   ├── xgboost_model.pkl       # (9 MB)
│   ├── random_forest_model.pkl # (65 MB)
│   ├── gnn_tuned_model.pt      # (2 MB)
│   └── transformer_model.pt    # (250 MB)
│
├── results/                     # Training results & analysis
│   ├── README.md                # Results documentation
│   ├── all_models_comparison.csv  # Complete metrics
│   ├── model_comparison.png     # Performance plots
│   └── *_results.csv            # Individual model results
│
└── docs/                        # Documentation & presentations
    ├── RESULTS.md               # Detailed analysis
    ├── VC_PITCH.md              # Investor presentation
    ├── EXECUTIVE_SUMMARY.md     # One-page overview
    └── COMPLETION_SUMMARY.md    # Project summary
```

## 🤖 Model Architectures

### 1. Traditional Machine Learning

**XGBoost** (Gradient Boosting)
- 500 trees, depth=8, lr=0.05
- Separate model per target property
- Features: Combined descriptors + Morgan fingerprints
- **Best for:** Density (R²=0.798)

**Random Forest** (Ensemble)
- 300 trees, depth=20
- Strong baseline performance
- Good interpretability via feature importance
- **Best for:** Tc (R²=0.761)

### 2. Graph Neural Networks (GPU-Accelerated)

- **Architecture:** 4-layer GCN with batch normalization (tuned)
- **Pooling:** Global mean + max pooling
- **Input:** Molecular graphs (atoms as nodes, bonds as edges)
- **Node Features:** Atom type, degree, charge, aromaticity (9 features)
- **Performance:** wMAE = 0.178 (14% improvement over baseline)
- **Training:** 30 seconds on RTX 4070 GPU

### 3. Transformer (DistilBERT)

- **Base:** Pre-trained DistilBERT encoder (frozen)
- **Head:** 3-layer MLP regression head (768 → 256 → 128 → 5)
- **Input:** Raw SMILES strings (max length: 256)
- **Training:** 20 epochs, batch size 16, lr=2e-5
- **Performance:** wMAE = 0.069 (3rd place, 21.8 min training)
- **Features:** Early stopping, gradient clipping, NaN handling

## 💡 Usage Examples

### Training a New Model

```python
from src.data_preprocessing import MolecularDataProcessor
from src.models import TraditionalMLModel

# Load and process data
processor = MolecularDataProcessor()
train_df, test_df, targets = processor.load_and_process_data(
    'data/raw/train.csv',
    'data/raw/test.csv'
)

# Extract features
train_features = processor.create_descriptor_features(train_df)
train_fp = processor.create_fingerprint_features(train_df)
X = pd.concat([train_features, train_fp], axis=1)

# Prepare targets
y = train_df[targets].values

# Train model
model = TraditionalMLModel(model_type='xgboost')
metrics = model.train(X_train, y_train, X_val, y_val, targets)

# Save model
model.save('models/my_model.pkl')
```

### Making Predictions

```python
from src.models import TraditionalMLModel
import pandas as pd

# Load model
model = TraditionalMLModel()
model.load('models/xgboost_model.pkl')

# Prepare new data
new_smiles = ["*CC(*)CCCC", "*c1ccccc1*"]
# ... (extract features as above)

# Predict
predictions = model.predict(X_new)
results = pd.DataFrame(
    predictions,
    columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg']
)
```

### Feature Extraction

```python
from src.data_preprocessing import MolecularDataProcessor

processor = MolecularDataProcessor()

# Compute molecular descriptors
descriptors = processor.compute_molecular_descriptors(smiles)
# Returns: {'MolWt': 180.2, 'LogP': 2.5, ...}

# Generate fingerprints
fingerprint = processor.compute_morgan_fingerprint(smiles, radius=2)
# Returns: numpy array of shape (2048,)
```

## 📊 Results

**See [results/README.md](results/README.md) for complete analysis**

### Highlights

**🏆 Winner: XGBoost**
- wMAE: 0.030429 (competition metric)
- Training: 5 minutes on CPU
- Best for: Density (R²=0.798), FFV (R²=0.760)
- **Recommended for production**

**🥈 Close Second: Random Forest**
- wMAE: 0.031638 (3.9% behind XGBoost)
- Training: 3 minutes on CPU
- More robust to outliers
- Best for: Tc (R²=0.761), Tg (R²=0.629)

**🥉 Transformer (DistilBERT):**
- wMAE: 0.069180 (127% behind XGBoost)
- Training: 22 minutes on GPU
- No feature engineering required (raw SMILES)
- Shows promise for larger datasets

**4️⃣ GNN (Tuned):**
- wMAE: 0.177712 (484% behind XGBoost)
- Training: 30 seconds on GPU
- 14% improvement from hyperparameter tuning
- Needs larger dataset (100K+ samples) to excel

### Key Insights

1. **Traditional ML Wins:** For 8K dataset, XGBoost/RF outperform deep learning by 2-6x
2. **Feature Engineering:** Descriptors + fingerprints → 40% better than either alone
3. **Deep Learning Challenges:** Both GNN and Transformer underperform on small datasets
4. **Training Efficiency:** Traditional ML offers best accuracy/time tradeoff
5. **GPU Acceleration:** Essential for deep learning but not enough to overcome data scarcity
6. **Property Difficulty:** Density easiest (R²=0.80), Rg hardest (R²=0.56)

## 📚 Documentation

### Technical Documentation
- **[results/README.md](results/README.md)** - Complete results analysis, metrics, and performance insights
- **[src/README.md](src/README.md)** - Source code documentation, API reference, and usage examples
- **[models/README.md](models/README.md)** - Trained model specifications, benchmarks, and usage guide
- **[app/README.md](app/README.md)** - Interactive demo guide with screenshots

### Research & Analysis
- **[docs/RESULTS.md](docs/RESULTS.md)** - Comprehensive analysis of all 4 models with detailed metrics
- **[docs/COMPLETION_SUMMARY.md](docs/COMPLETION_SUMMARY.md)** - Project summary and achievement overview

### Business & Presentations
- **[docs/VC_PITCH.md](docs/VC_PITCH.md)** - Full investor presentation deck (20+ slides)
- **[docs/EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md)** - One-page business overview for investors

### Competition
- **Competition:** [Kaggle NeurIPS 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)

## 🛠️ Development

### Running Tests

```bash
# Test data preprocessing
python src/data_preprocessing.py

# Test individual models
python -c "from src.models import TraditionalMLModel; print('Import successful')"
```

### Adding New Features

1. Edit `src/data_preprocessing.py`
2. Add descriptor calculation method
3. Update feature combination in `create_descriptor_features()`

### Adding New Models

1. Create `src/models/your_model.py`
2. Implement with `train()` and `predict()` methods
3. Add import to `src/models/__init__.py`
4. Integrate into `src/train.py`

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Model Performance:** Hyperparameter tuning, ensemble methods
- **Feature Engineering:** Domain-specific descriptors
- **Deep Learning:** Fix GNN/Transformer training issues
- **Documentation:** Additional examples and tutorials
- **Testing:** Unit tests and integration tests

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{oh2025openpolymer,
  author = {Jihwan Oh},
  title = {Open Polymer: Machine Learning for Polymer Property Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jihwanksa/open_polymer}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Kaggle Competition:** [NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
- **RDKit:** Open-source cheminformatics toolkit
- **PyTorch Geometric:** Graph neural network library
- **ChemBERTa:** Pre-trained transformer for chemistry

## 📞 Contact

- **Author:** Jihwan Oh
- **GitHub:** [@jihwanksa](https://github.com/jihwanksa)
- **Repository:** [open_polymer](https://github.com/jihwanksa/open_polymer)

---

**Project Status:** ✅ All Models Complete | 🏆 XGBoost Best (wMAE: 0.030)

**Last Updated:** October 10, 2025
