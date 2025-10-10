# Models Directory

This directory contains trained model checkpoints ready for inference on new polymer molecules.

## 📦 Saved Models

| Model File | Type | Size | wMAE | Training Time | Best For |
|------------|------|------|------|---------------|----------|
| `xgboost_model.pkl` | Traditional ML | ~9 MB | **0.030** | 5 min | **Production** ⭐ |
| `random_forest_model.pkl` | Traditional ML | ~65 MB | 0.032 | 3 min | Robustness |
| `gnn_tuned_model.pt` | Deep Learning | ~2 MB | 0.178 | 30 sec | GPU Inference |
| `transformer_model.pt` | Deep Learning | ~250 MB | 0.069 | 22 min | Research |

## 🚀 Quick Usage

### Load and Predict

```python
from src.models import TraditionalMLModel, GNNModel, TransformerModel
from src.data_preprocessing import MolecularDataProcessor
import pandas as pd

# === XGBoost (Recommended) ===
model = TraditionalMLModel()
model.load('models/xgboost_model.pkl')

# Prepare features
processor = MolecularDataProcessor()
smiles_list = ["*CC(*)CCCC", "*c1ccccc1*"]
df = pd.DataFrame({'SMILES': smiles_list})

# Extract features
descriptors = processor.create_descriptor_features(df)
fingerprints = processor.create_fingerprint_features(df)
X = pd.concat([descriptors, fingerprints], axis=1)

# Predict
predictions = model.predict(X)
# Shape: (2, 5) for [Tg, FFV, Tc, Density, Rg]

print(f"Tg: {predictions[0, 0]:.2f} °C")
print(f"FFV: {predictions[0, 1]:.4f}")
print(f"Tc: {predictions[0, 2]:.4f}")
print(f"Density: {predictions[0, 3]:.4f} g/cm³")
print(f"Rg: {predictions[0, 4]:.2f} Å")
```

### Batch Inference

```python
# Load test data
test_df = pd.read_csv('data/raw/test.csv')

# Extract features
X_test = processor.create_combined_features(test_df)

# Predict
predictions = model.predict(X_test)

# Save submission
submission = pd.DataFrame(
    predictions,
    columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg']
)
submission.to_csv('submission.csv', index=False)
```

## 📊 Model Specifications

### 1. XGBoost (`xgboost_model.pkl`) ⭐

**Architecture**: Gradient Boosting Trees
- **Trees**: 500 estimators
- **Max Depth**: 8
- **Learning Rate**: 0.05
- **Features**: 2,063 (15 descriptors + 2,048 fingerprints)

**Performance**:
- wMAE: **0.030429** (Best Overall)
- Best for: Density (R²=0.798), FFV (R²=0.760)
- Inference Speed: ~1ms per molecule (CPU)

**When to Use**:
- ✅ Production deployments
- ✅ Best accuracy requirements
- ✅ CPU-only environments
- ✅ Small-medium batch sizes (<10K)

**Limitations**:
- ❌ Requires feature engineering
- ❌ Larger file size (9 MB)

### 2. Random Forest (`random_forest_model.pkl`)

**Architecture**: Ensemble of Decision Trees
- **Trees**: 300 estimators
- **Max Depth**: 20
- **Features**: 2,063 (same as XGBoost)

**Performance**:
- wMAE: **0.031638** (2nd Best)
- Best for: Tc (R²=0.761), Tg (R²=0.629)
- Inference Speed: ~2ms per molecule (CPU)

**When to Use**:
- ✅ Need fast training (<3 min)
- ✅ More robust to outliers
- ✅ Feature importance analysis
- ✅ Ensemble with XGBoost

**Limitations**:
- ❌ Large file size (65 MB)
- ❌ Slightly lower accuracy than XGBoost

### 3. GNN Tuned (`gnn_tuned_model.pt`)

**Architecture**: 4-layer Graph Convolutional Network
- **Layers**: GCNConv (128 → 256 → 256 → 128)
- **Pooling**: Global mean + max
- **Head**: MLP (256 → 128 → 64 → 5)
- **Dropout**: 0.3

**Performance**:
- wMAE: **0.177712** (4th)
- Training: 30 seconds (GPU)
- Inference Speed: ~5ms per molecule (GPU)

**When to Use**:
- ✅ GPU available
- ✅ Research/experimentation
- ✅ Direct graph input preferred
- ✅ Fast training needed

**Limitations**:
- ❌ Requires PyTorch Geometric
- ❌ Lower accuracy on small datasets
- ❌ Needs GPU for good performance

### 4. Transformer (`transformer_model.pt`)

**Architecture**: DistilBERT + Regression Head
- **Base**: distilbert-base-uncased (frozen)
- **Head**: 768 → 256 → 128 → 5
- **Input**: Raw SMILES strings
- **Max Length**: 256 tokens

**Performance**:
- wMAE: **0.069180** (3rd)
- Training: 22 minutes (GPU)
- Inference Speed: ~20ms per molecule (GPU)

**When to Use**:
- ✅ No feature engineering
- ✅ Raw SMILES input
- ✅ Transfer learning experiments
- ✅ Research on transformers

**Limitations**:
- ❌ Large file size (250 MB)
- ❌ Slower inference
- ❌ Needs GPU for practical use
- ❌ Would benefit from ChemBERTa

## 💾 File Formats

### Traditional ML (.pkl)
```python
# Pickle format containing:
{
    'model_type': 'xgboost',
    'models': {
        'Tg': XGBRegressor(...),
        'FFV': XGBRegressor(...),
        ...
    },
    'feature_names': [...],
    'target_names': ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
}
```

### PyTorch (.pt)
```python
# State dict format:
{
    'model_state_dict': OrderedDict(...),
    'config': {
        'hidden_channels': 128,
        'num_targets': 5,
        ...
    }
}
```

## 🔄 Retraining Models

### From Scratch

```bash
# Traditional ML
python src/train.py

# GNN
python src/train_gnn_tuned.py

# Transformer
python src/train_transformer.py
```

### Fine-tuning

```python
# Load existing model
model = TraditionalMLModel()
model.load('models/xgboost_model.pkl')

# Fine-tune on new data
model.train(X_new, y_new, X_val, y_val)

# Save updated model
model.save('models/xgboost_finetuned.pkl')
```

## 🎯 Model Selection Guide

### For Production
**Recommend**: XGBoost
- Highest accuracy
- Fast inference (CPU)
- Proven reliability
- 5-minute training

### For Research
**Recommend**: Transformer or GNN
- Modern architectures
- Transfer learning potential
- Active research area

### For Speed
**Recommend**: GNN (with GPU) or XGBoost (CPU)
- GNN: 30-second training
- XGBoost: 1ms inference

### For Interpretability
**Recommend**: Random Forest or XGBoost
- Feature importance scores
- Tree visualization
- SHAP values support

## 🔒 Model Versioning

Current models trained on:
- **Dataset**: NeurIPS 2025 Open Polymer
- **Training Samples**: 7,973
- **Date**: October 2025
- **Validation**: 80/20 split

Version history:
- `v1.0` - Initial training (all models)
- `v1.1` - GNN hyperparameter tuning
- `v1.2` - Transformer with NaN handling fixes

## 🐛 Troubleshooting

### "File not found"
```python
# Use absolute path
import os
model_path = os.path.join(os.getcwd(), 'models', 'xgboost_model.pkl')
model.load(model_path)
```

### "Pickle version mismatch"
```python
# Check Python version
import sys
print(sys.version)  # Should be 3.10+

# Retrain if needed
```

### "CUDA out of memory" (GNN/Transformer)
```python
# Reduce batch size
model.train(..., batch_size=8)

# Or use CPU
model = GNNModel(device='cpu')
```

### "Module not found: rdkit"
```bash
# Install with conda (required for RDKit)
conda install -c conda-forge rdkit
```

## 📊 Benchmarks

### Inference Speed (CPU: Intel i7, GPU: RTX 4070)

| Model | CPU (per mol) | GPU (per mol) | Batch 100 (CPU) | Batch 100 (GPU) |
|-------|---------------|---------------|-----------------|-----------------|
| XGBoost | 1ms | N/A | 0.1s | N/A |
| Random Forest | 2ms | N/A | 0.2s | N/A |
| GNN | 50ms | 5ms | 5.0s | 0.5s |
| Transformer | 100ms | 20ms | 10.0s | 2.0s |

### Memory Usage

| Model | RAM (Inference) | VRAM (GPU) | Disk Size |
|-------|-----------------|------------|-----------|
| XGBoost | 50 MB | N/A | 9 MB |
| Random Forest | 200 MB | N/A | 65 MB |
| GNN | 100 MB | 500 MB | 2 MB |
| Transformer | 500 MB | 2 GB | 250 MB |

## 🔐 Model Security

**Pickle Security Note**: 
- `.pkl` files can execute arbitrary code
- Only load models from trusted sources
- Consider using `joblib` for safer serialization

**Safe Loading**:
```python
import joblib
# Instead of pickle, use:
model = joblib.load('models/xgboost_model.pkl')
```

## 📚 References

- **XGBoost**: Chen & Guestrin (2016) - KDD
- **Random Forest**: Breiman (2001) - Machine Learning
- **GCN**: Kipf & Welling (2017) - ICLR
- **DistilBERT**: Sanh et al. (2019) - NeurIPS Workshop

## 🤝 Contributing

To add a new model:
1. Train using training scripts in `src/`
2. Save with consistent naming: `{model_name}_model.{ext}`
3. Update this README with specifications
4. Add usage example above

---

**Last Updated**: October 2025  
**Total Models**: 4  
**Best Model**: XGBoost (wMAE: 0.030)

