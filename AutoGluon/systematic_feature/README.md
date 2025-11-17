# Systematic Feature Analysis with AutoGluon

Testing 8 different feature configurations to find the optimal feature subset for polymer property prediction.

## Overview

Instead of manually selecting features, we let AutoGluon systematically evaluate different combinations:
- Which features does AutoGluon select as most important?
- How does performance vary with different feature subsets?
- Are we missing useful RDKit descriptors?
- Is domain knowledge (hand-crafted features) valuable?

## Configurations

| Config | Type | Features | Description |
|--------|------|----------|-------------|
| **A** | Baseline | 10 | Simple only (SMILES counting) |
| **B** | Domain | 11 | Hand-crafted only |
| **C** | Current | 34 | 10 Simple + 11 Hand-crafted + 13 RDKit |
| **D** | Expanded | 56 | 10 Simple + 11 Hand-crafted + 35 RDKit |
| **E** | Full | 221 | 10 Simple + 11 Hand-crafted + 200 RDKit |
| **F** | Isolated | 35 | RDKit only (expanded, no simple/HC) |
| **G** | Ablation | 24 | Hand-crafted + RDKit (no simple) |
| **H** | Ablation | 23 | Simple + RDKit (no hand-crafted) |

## Quick Start

### 1. Run Systematic Analysis (45 min)

```bash
conda activate pseudolabel_env
cd open_polymer/AutoGluon/systematic_feature

# Train all 8 configurations
python train_systematic_features.py
```

**Output:**
```
AutoGluon/systematic_feature/results/
├── A_simple_only/
│   ├── Tg/                    # AutoGluon models
│   ├── FFV/
│   ├── Tc/
│   ├── Density/
│   ├── Rg/
│   └── config_results.json    # Results for config A
├── B_hand_crafted_only/
│   └── ...
├── ... (8 configs total)
└── comparison_report.csv      # Side-by-side comparison
```

### 2. Analyze Results

```bash
python analyze_results.py results/
```

**Output:**
```
📊 FEATURE EFFICIENCY METRICS

Configuration               Input Features Avg Selected Features Feature Reduction %
A_simple_only                    10              8.2               18.0%
B_hand_crafted_only              11              9.1               17.2%
C_current_baseline               34              19.0              44.1%
D_expanded_rdkit                 56              20.3              63.8%
E_all_rdkit                      221             18.5              91.6%
F_rdkit_only_expanded            35              16.2              53.7%
G_no_simple                      24              17.8              25.9%
H_no_hand_crafted                23              16.9              26.6%
```

## What to Look For

### 1. Feature Reduction Efficiency

**Key Metric:** How much AutoGluon filters from input features

- **Configuration C vs D vs E:**
  - If E selects only 20 features from 221, that's **91.6% reduction**
  - Means only ~20 RDKit descriptors are truly useful
  - Our manual selection (13) is already capturing most value

- **What this tells us:**
  - If E ≈ D: Expanded RDKit (56) is sufficient, all 200 is overkill
  - If E >> D: Missing important RDKit descriptors in our manual selection
  - If E ≈ C: Our current 13 RDKit choices are near-optimal

### 2. Domain Knowledge Value

**Compare these pairs:**

1. **B (hand-crafted only) vs F (RDKit only)**
   - Which performs better?
   - If B >> F: Domain knowledge wins
   - If F >> B: Chemistry descriptors are more important

2. **G (no simple) vs H (no hand-crafted)**
   - Which ablation hurts more?
   - Shows which feature type is more critical

### 3. Simplicity vs Performance

**Compare A vs C:**
- Simple (10) vs Full (34)
- How much performance are we gaining?
- If difference < 5%: Use A (10 features)
- If difference > 15%: Hand-crafted features essential

### 4. Training Time vs Accuracy

**Expected Times:**
- A: ~2 min (10 features)
- B: ~2 min (11 features)
- C: ~3 min (34 features)
- D: ~4 min (56 features)
- E: ~10 min (221 features, but heavy filtering)
- F: ~3 min (35 features)
- G: ~3 min (24 features)
- H: ~3 min (23 features)

**Total: ~45 minutes** for all 8 configurations

## Key Questions Answered

### Q1: Should we use all 200+ RDKit descriptors?
- **Look at:** Config E vs Config C
- If E performs better but takes 3x longer, probably not worth it
- If E performs same as C, our manual 13 is sufficient

### Q2: Is domain knowledge important?
- **Look at:** Config B (hand-crafted) vs Config F (RDKit only)
- If B >> F: Domain knowledge is critical
- If F >> B: Stick with RDKit descriptors

### Q3: Are simple features useful?
- **Look at:** Config G (no simple) vs Config C (full)
- If G ≈ C: Simple features are redundant
- If C >> G: Simple features add value

### Q4: What's the minimal feature set that works well?
- **Look at:** Config A (10 simple)
- If A achieves 90% of Config C's performance, use A!
- Simpler models are faster, more interpretable

## File Reference

| File | Purpose |
|------|---------|
| `train_systematic_features.py` | Main script - trains all 8 configurations |
| `analyze_results.py` | Analysis script - generates report |
| `PLAN.md` | Detailed planning document |
| `README.md` | This file |
| `results/` | Output directory with all models and reports |

## Example Output Files

### config_results.json (per configuration)
```json
{
  "config": "C_current_baseline",
  "description": "Current baseline (10 simple + 11 hand-crafted + 13 RDKit) - 34 features",
  "input_features": 34,
  "models": {
    "Tg": {"selected_features": 19, "model_path": "results/C_current_baseline/Tg"},
    "FFV": {"selected_features": 18, "model_path": "results/C_current_baseline/FFV"},
    "Tc": {"selected_features": 20, "model_path": "results/C_current_baseline/Tc"},
    "Density": {"selected_features": 19, "model_path": "results/C_current_baseline/Density"},
    "Rg": {"selected_features": 21, "model_path": "results/C_current_baseline/Rg"}
  },
  "training_time": 185.4
}
```

### comparison_report.csv
```
Configuration,Description,Input Features,Models Trained,Training Time (s),Avg Selected Features
A_simple_only,Simple only SMILES counting - 10 features,10,5,120.2,8.2
B_hand_crafted_only,Hand-crafted domain knowledge only - 11 features,11,5,135.1,9.1
C_current_baseline,Current baseline (10 simple + 11 hand-crafted + 13 RDKit) - 34 features,34,5,180.5,19.0
...
```

## Interpreting Feature Selection

### Example: Why did AutoGluon select 19 from 34?

**Input 34 features:**
- 10 simple (carbon_count, nitrogen_count, etc.)
- 11 hand-crafted (num_side_chains, branching_ratio, etc.)
- 13 RDKit (MolWt, LogP, TPSA, etc.)

**AutoGluon selected 19:**
- Likely kept most hand-crafted (they're domain-optimized)
- Kept some simple features (carbon_count is useful)
- Kept 5-6 RDKit descriptors (redundancy with hand-crafted)

**This means:** 
- Hand-crafted features capture most information
- Simple features add some value
- RDKit adds marginal improvement

## Next Steps

1. **Run analysis:** `python train_systematic_features.py`
2. **Analyze results:** `python analyze_results.py results/`
3. **Review findings:** Check `results/ANALYSIS_REPORT.txt`
4. **Decide:** Pick optimal configuration based on trade-off:
   - Best performance? → Use E_all_rdkit
   - Best efficiency? → Use A_simple_only or B_hand_crafted_only
   - Balanced? → Use C_current_baseline or D_expanded_rdkit
5. **Update production:** Copy winning configuration to main pipeline

## Troubleshooting

### Script stops early
- Check memory: AutoGluon might run out of RAM with 221 features
- Reduce `time_limit` or use `preset='fast'`

### Results look wrong
- Verify data loaded correctly: Check first config's output
- RDKit descriptors failing? Check RDKIT_AVAILABLE flag

### Want to test custom configuration
- Edit `FEATURE_CONFIGURATIONS` dict in script
- Add new entry with custom feature set
- Re-run script

## Remember

- ✅ More features ≠ Better performance (overfitting risk)
- ✅ AutoGluon's feature selection is the key insight here
- ✅ Simpler models are faster and more interpretable
- ✅ Domain knowledge often beats raw feature count

---

**Status:** Ready to run | **Last Updated:** Nov 14, 2025

