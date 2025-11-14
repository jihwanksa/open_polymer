# AutoGluon Production Guide

## 🚀 Quick Start

Use AutoGluon models for production inference:

```bash
python AutoGluon/inference_autogluon.py \
  --test_file data/raw/test.csv \
  --output submission.csv
```

This generates predictions using the pre-trained **WeightedEnsemble_L2** models trained on 60K+ samples with 34 features.

---

## 📊 Model Configuration

### Pre-Trained Models

**Location:** `models/autogluon_production/{Tg,FFV,Tc,Density,Rg}/`

**Model Type:** WeightedEnsemble_L2
- Stacked ensemble of 8 base models
- Automatically learned weights for each model
- Optimized for MAE (Mean Absolute Error)

### Training Configuration

- **Data:** 60,000+ samples
  - 7,973 original labeled
  - ~17,000 external (Tc, Tg, Density, Rg)
  - 50,000 pseudo-labeled (BERT + Uni-Mol ensemble)

- **Features:** 34 total
  - 10 simple (SMILES parsing)
  - 11 hand-crafted polymer-specific
  - 13 RDKit molecular descriptors

- **Hyperparameters:** AutoGluon-optimized
  - Preset: `medium_quality`
  - Time limit: 1800s per property
  - CPU-only mode for stability

### Key Transformations

**Tg Transformation:** Applied post-prediction
```python
Tg_transformed = (9/5) * Tg_predicted + 45
```
This corrects for train/test distribution shift (discovered by 2nd place winner).

---

## 🔧 Files and Usage

### Production Inference

**Script:** `AutoGluon/inference_autogluon.py`

```bash
python AutoGluon/inference_autogluon.py \
  --test_file data/raw/test.csv \
  --model_dir models/autogluon_production \
  --sample_submission data/raw/sample_submission.csv \
  --output submission.csv
```

**Features:**
- Loads pre-trained AutoGluon models
- Creates 21 chemistry features from SMILES
- Generates predictions for all 5 properties
- Applies Tg transformation
- Outputs competition-format submission

### Training Pipeline

**Script:** `AutoGluon/train_autogluon_production.py`

```bash
python AutoGluon/train_autogluon_production.py \
  --time_limit 1800 \
  --preset medium_quality \
  --output_dir models/autogluon_production
```

**Process:**
1. Loads and augments training data (60K+ samples)
2. Extracts 34 comprehensive features
3. Trains AutoGluon for each property
4. Saves WeightedEnsemble_L2 models

### Alternative: Python Script

**Script:** `AutoGluon/train_v85_best.py`

```bash
python AutoGluon/train_v85_best.py
```

Loads pre-trained models and evaluates on training data.

---

## 📁 Directory Structure

```
models/autogluon_production/
├── Tg/                     # AutoGluon models for Tg
│   ├── models/             # Base models (RF, XGB, LGB, etc.)
│   ├── learner.pkl         # Trainer object
│   ├── predictor.pkl       # Predictor interface
│   └── metadata.json       # Configuration
├── FFV/                    # AutoGluon models for FFV
├── Tc/                     # AutoGluon models for Tc
├── Density/                # AutoGluon models for Density
├── Rg/                     # AutoGluon models for Rg
└── feature_importance.json # Feature rankings (empty)
```

---

## 🐍 Python Usage

For integration into notebooks or other scripts:

```python
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np

# Load a model
predictor = TabularPredictor.load('models/autogluon_production/Tg')

# Prepare features (34 features expected)
X_test = pd.DataFrame(X_test, columns=feature_names)

# Generate predictions
predictions = predictor.predict(X_test)

# Apply Tg transformation if needed
if property_name == 'Tg':
    predictions = (9/5) * predictions + 45
```

---

## 🎯 Performance

**AutoGluon vs Manual Random Forest (v85):**

| Aspect | v85 (Manual RF) | AutoGluon |
|--------|-----------------|-----------|
| Training approach | Manual hyperparameter tuning | Automated with 1800s budget |
| Feature selection | All 21 features | 34 features, auto-selected |
| Ensemble | 5 RF models per property | 8 diverse base models |
| Model selection | Single algorithm | Best from {RF, XGB, LGB, NN, ...} |
| Expected performance | Baseline (1st place tied) | Equal or better |

---

## ⚠️ Important Notes

1. **CPU-Only Mode**
   - AutoGluon runs on CPU only (not MPS)
   - Set before import: `os.environ['MPS_ENABLED'] = '0'`
   - Stable and reliable on Apple Silicon

2. **Feature Requirements**
   - Script expects 21 chemistry features
   - AutoGluon models use exactly these 34 features
   - Feature order matters!

3. **Tg Transformation**
   - Only applied to Tg predictions
   - Essential for test set alignment
   - Implemented automatically in inference script

4. **Memory Requirements**
   - Total model size: ~500MB-1GB
   - Inference memory: Moderate (fits in 8GB)
   - Training memory: 8GB+ recommended

---

## 🔄 Workflow

```
1. Prepare test data (SMILES)
           ↓
2. Run inference_autogluon.py
           ↓
3. Features extracted automatically
           ↓
4. AutoGluon models predict
           ↓
5. Tg transformation applied
           ↓
6. Submission CSV generated
           ↓
7. Submit to Kaggle!
```

---

## 📚 References

- **BERT Embeddings:** `unikei/bert-base-smiles` (Hugging Face)
- **Uni-Mol Model:** `dptech/Uni-Mol2` (Hugging Face)
- **AutoGluon Documentation:** https://auto.gluon.ai/
- **Kaggle Competition:** https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025

---

**Last Updated:** November 14, 2025  
**Version:** 1.0 (Production Release)  
**Status:** ✅ Ready for Kaggle Submission

