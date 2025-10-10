# Open Polymer: Polymer Properties Prediction 🧪

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning framework for predicting polymer properties from SMILES strings. Implements and compares multiple model architectures including traditional ML (XGBoost, Random Forest), Graph Neural Networks, and Transformers.

> **Competition:** [NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) (Kaggle)

## 🎯 Quick Results

| Property | Best Model | RMSE | R² Score |
|----------|-----------|------|----------|
| **Density** | XGBoost | 0.064 g/cm³ | **0.798** ⭐⭐ |
| **FFV** | XGBoost | 0.015 | **0.760** ⭐ |
| **Tc** | Random Forest | 0.046 | **0.761** ⭐ |
| **Tg** | Random Forest | 69.49 °C | 0.629 |
| **Rg** | XGBoost | 3.14 Å | 0.561 |

**Average R² Score:** 0.698 (XGBoost), 0.678 (Random Forest)

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
  - ✅ Traditional ML: XGBoost, Random Forest
  - ✅ Graph Neural Networks (GNN): GCN with graph pooling
  - ✅ Transformers: ChemBERTa-based architecture

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

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Anaconda or Miniconda (recommended for RDKit)
- CUDA-capable GPU (optional, for deep learning models)

### Setup

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

# Set library path (Linux/Mac)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Dependencies

- **Chemistry:** RDKit
- **Traditional ML:** scikit-learn, XGBoost
- **Deep Learning:** PyTorch, PyTorch Geometric, Transformers
- **Utilities:** pandas, numpy, matplotlib, seaborn

See [`requirements.txt`](requirements.txt) for full list.

## 🎓 Quick Start

### 1. Explore the Data

```bash
cd src
python data_preprocessing.py
```

### 2. Train Models

```bash
# Train all models (traditional ML)
python train.py

# Or use the convenience script
bash scripts/run_training.sh
```

### 3. View Results

```bash
# See comparison metrics
cat results/model_comparison.csv

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
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py    # Feature extraction
│   ├── train.py                 # Main training pipeline
│   └── models/                  # Model implementations
│       ├── __init__.py
│       ├── traditional.py       # XGBoost & Random Forest
│       ├── gnn.py              # Graph Neural Network
│       └── transformer.py      # ChemBERTa-based model
│
├── scripts/                     # Utility scripts
│   └── run_training.sh         # Training launcher
│
├── notebooks/                   # Jupyter notebooks (future)
│
├── data/                        # Data directory
│   └── raw/                    # Original Kaggle data
│       ├── train.csv           # Training data (7,973 samples)
│       ├── test.csv            # Test data (3 samples)
│       ├── sample_submission.csv
│       └── train_supplement/   # Additional datasets
│
├── models/                      # Saved model checkpoints
│   ├── xgboost_model.pkl       # (8.7 MB)
│   └── random_forest_model.pkl # (65 MB)
│
├── results/                     # Training results
│   ├── model_comparison.csv    # Metrics table
│   └── model_comparison.png    # Performance plots
│
└── docs/                        # Additional documentation
    ├── RESULTS_SUMMARY.md      # Detailed analysis
    └── QUICK_START.md          # Quick reference guide
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

### 2. Graph Neural Networks

- **Architecture:** 3-layer GCN with batch normalization
- **Pooling:** Global mean + max pooling
- **Input:** Molecular graphs (atoms as nodes, bonds as edges)
- **Node Features:** Atom type, degree, charge, aromaticity (9 features)

### 3. Transformer (ChemBERTa)

- **Base:** Pre-trained ChemBERTa encoder
- **Head:** 2-layer MLP regression head (256 → 128 → 5)
- **Input:** Raw SMILES strings
- **Optimization:** Separate learning rates for encoder vs head

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

### Performance by Target Property

**FFV (Fractional Free Volume)** - Most abundant labels (88% coverage)
- XGBoost: RMSE=0.0147, R²=0.760 ⭐
- Data-rich enables strong predictions

**Density** - Best overall performance
- XGBoost: RMSE=0.064 g/cm³, R²=0.798 ⭐⭐
- Physical property well-captured by features

**Tc (Critical Temperature)** - Strong performance
- Random Forest: RMSE=0.046, R²=0.761 ⭐
- Ensemble method excels

**Tg (Glass Transition Temperature)** - Moderate performance
- Random Forest: RMSE=69.49 °C, R²=0.629
- Limited data (6% coverage) challenges predictions

**Rg (Radius of Gyration)** - Most challenging
- XGBoost: RMSE=3.14 Å, R²=0.561
- Structural complexity makes prediction difficult

### Key Insights

1. **Feature Engineering Matters:** Combining descriptors and fingerprints improves performance
2. **Sparse Labels:** Models handle missing data well via per-target training
3. **XGBoost vs Random Forest:** XGBoost slightly better on average
4. **Data Coverage:** Performance doesn't strictly correlate with label abundance

## 📚 Documentation

- **[QUICK_START.md](docs/QUICK_START.md)** - Commands, use cases, troubleshooting
- **[RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)** - Detailed analysis and insights
- **Source Code** - Extensively commented for readability

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

**Project Status:** ✅ Traditional ML Complete | ⚠️ Deep Learning 95% Complete

**Last Updated:** October 2025
