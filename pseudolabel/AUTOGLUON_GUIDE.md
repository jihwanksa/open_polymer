# AutoGluon Model Selection Guide

## Problem Statement

For the pseudo-label generation to work, we need three models:
1. **BERT-base-SMILES** ✅ (Clear: `unikei/bert-base-smiles`)
2. **Uni-Mol 2** ✅ (Clear: `dptech/Uni-Mol2` 84M checkpoint)
3. **AutoGluon** ❓ (Multiple options on Hugging Face)

## AutoGluon Options

### Option 1: Train Your Own AutoGluon Model (RECOMMENDED)

**When to use:** You want maximum control and best performance for the polymer dataset.

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load training data
train_df = pd.read_csv('data/raw/train.csv')

# Prepare data (handle missing values, normalize if needed)
# For each property separately
for target_property in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    print(f"Training AutoGluon for {target_property}...")
    
    # Filter to rows with this property
    df_property = train_df[train_df[target_property].notna()].copy()
    
    # Extract features (21 chemistry-based)
    # ... feature extraction code ...
    
    # Train AutoGluon
    predictor = TabularPredictor(
        label=target_property,
        problem_type='regression',
        eval_metric='mean_absolute_error'  # Match competition metric
    )
    
    predictor.fit(
        train_data=df_property,
        time_limit=600,  # 10 minutes per property
        presets='best_quality'  # Slow but accurate
    )
    
    # Save model
    predictor.save(f'models/autogluon_{target_property}')
    print(f"   ✅ Model saved to models/autogluon_{target_property}")
```

**Pros:**
- ✅ Trained on polymer data (best domain match)
- ✅ Ensemble of many algorithms (gradient boosting, random forest, neural nets)
- ✅ Automatically finds best hyperparameters
- ✅ Handles missing values well
- ✅ Already optimizes for MAE (competition metric)

**Cons:**
- ⏱️ Takes time to train (10 min per property × 5 properties)
- 💾 Models are large (hundreds of MB each)

---

### Option 2: Use Pre-trained Model from 1st Place Solution

**When to use:** If the 1st place solution released their AutoGluon models.

```bash
# Check if available on Hugging Face or their GitHub
https://huggingface.co/models?search=polymer+autogluon
https://kaggle.com/c/neurips-open-polymer-prediction-2025  # Check discussion
```

**Pros:**
- ✅ Already trained on the exact dataset
- ✅ Known to work for the competition
- ✅ No training time needed

**Cons:**
- ❌ Might not be public
- ❌ Need to find exact model details

---

### Option 3: Use a General Tabular Model

**When to use:** You want a quick solution without training.

#### 3a. AutoML Tabular (General Purpose)
```python
from huggingface_hub import hf_hub_download

# Download a general tabular AutoGluon model
model = hf_hub_download(
    repo_id="aurelien-git/autogluon-binary-classification",
    filename="model.pkl"
)
```

**Issues:**
- ❌ Not trained on polymer data
- ❌ Classification model (not regression)
- ⚠️ Probably won't work well

#### 3b. Chemistry/Molecular Models
```bash
# Search on Hugging Face for chemistry models
https://huggingface.co/models?search=chemistry+property
https://huggingface.co/models?search=molecular+property
```

**Issues:**
- ❌ Hard to find models that match 5 properties
- ⚠️ Quality unknown

---

## RECOMMENDATION

### Best Choice: **Option 1 - Train Your Own AutoGluon**

**Why:**
1. ✅ Guaranteed to use correct data (polymer training set)
2. ✅ Optimized for MAE (competition metric)
3. ✅ AutoGluon is proven ensemble method
4. ✅ Will work well with BERT + Uni-Mol for ensemble

### Quick Start (Takes ~1 hour)

```bash
# 1. Install AutoGluon
pip install autogluon

# 2. Create training script
python pseudolabel/train_autogluon_models.py

# 3. Models will be saved in models/autogluon_*

# 4. Generate pseudo-labels
python pseudolabel/generate_pseudolabels.py \
    --bert_model models/bert_smiles \
    --unimol_model models/unimol2 \
    --autogluon_dir models/autogluon_* \
    --output_path pseudolabel/pi1m_pseudolabels.csv
```

---

## Implementation: Train AutoGluon

Here's a complete script to train AutoGluon:

```python
# pseudolabel/train_autogluon_models.py
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os

# Load data
train_df = pd.read_csv('data/raw/train.csv')

# Add external augmentation data
try:
    tc_df = pd.read_csv('data/Tc_SMILES.csv')
    train_df = pd.concat([train_df, tc_df], ignore_index=True)
except:
    pass

# Extract chemistry features (same as RF)
from src.train_v85_best import create_chemistry_features

print("Creating chemistry features...")
features = create_chemistry_features(train_df)
train_with_features = pd.concat([train_df[['SMILES']], features, train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]], axis=1)

# Train AutoGluon for each property
target_properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for target in target_properties:
    print(f"\n{'='*60}")
    print(f"Training AutoGluon for {target}")
    print(f"{'='*60}")
    
    # Filter valid data
    df_valid = train_with_features[train_with_features[target].notna()].copy()
    print(f"Training samples: {len(df_valid)}")
    
    if len(df_valid) < 100:
        print(f"⚠️  Not enough samples for {target}, skipping")
        continue
    
    # Prepare features (drop SMILES and target columns for feature matrix)
    feature_cols = [c for c in df_valid.columns if c not in ['SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
    X = df_valid[feature_cols]
    y = df_valid[target]
    
    # Train AutoGluon
    predictor = TabularPredictor(
        label=target,
        problem_type='regression',
        eval_metric='mean_absolute_error'
    )
    
    predictor.fit(
        train_data=pd.DataFrame({**X, target: y}),
        time_limit=300,  # 5 minutes per property
        presets='best_quality'
    )
    
    # Save model
    model_dir = f'models/autogluon_{target}'
    predictor.save(model_dir)
    print(f"✅ Model saved to {model_dir}")

print("\n" + "="*60)
print("✅ All AutoGluon models trained!")
print("="*60)
```

---

## Full Setup Workflow

```bash
# 1. Setup BERT and Uni-Mol
python pseudolabel/setup_pretrained_models.py --bert --unimol

# 2. Train AutoGluon
python pseudolabel/train_autogluon_models.py

# 3. Generate pseudo-labels
python pseudolabel/generate_pseudolabels.py \
    --input_data data/PI1M_50000_v2.1.csv \
    --bert_model models/bert_smiles \
    --unimol_model models/unimol2 \
    --autogluon_models models/autogluon_* \
    --output_path pseudolabel/pi1m_pseudolabels_ensemble.csv

# 4. Create augmented training data
python pseudolabel/augment_training_data.py

# 5. Train final Random Forest
python src/train_v85_best.py

# 6. Done! You've reproduced the 1st place solution workflow!
```

---

## Summary Table

| Approach | Pros | Cons | Recommended |
|----------|------|------|-------------|
| **Train Own** | Domain-specific, controlled | Takes time | ✅ YES |
| **Pre-trained (1st place)** | No training needed | Hard to find | ⚠️ Maybe |
| **Pre-trained (General)** | Quick | Not domain-specific | ❌ No |
| **Skip AutoGluon** | Faster pipeline | Less ensemble diversity | ⚠️ Possible |

---

## Next Steps

1. **Choose approach** (recommended: train your own)
2. **Run setup script** for BERT + Uni-Mol
3. **Train AutoGluon** if using Option 1
4. **Generate pseudo-labels** with full ensemble
5. **Train Random Forest** with augmented data

Let me know which approach you prefer, and I'll create the implementation script!

