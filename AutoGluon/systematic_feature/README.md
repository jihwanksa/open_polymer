# Systematic Feature Analysis Plan for AutoGluon

## Goal
Systematically evaluate different feature subsets using AutoGluon to find the optimal combination of RDKit descriptors + hand-crafted features for polymer property prediction.

---

## 1. Available Features Analysis

### 1.1 RDKit Descriptors
**Total Available: ~200 descriptors** (RDKit has comprehensive molecular descriptor coverage)

Currently using **13** hand-picked descriptors:
- MolWt, LogP, NumHDonors, NumHAcceptors, NumRotatableBonds
- NumAromaticRings, TPSA, NumSaturatedRings, NumAliphaticRings
- RingCount, FractionCsp3, NumHeteroatoms, BertzCT

**Opportunity:** We're only using ~6% of available RDKit descriptors!

### 1.2 Hand-Crafted Features
**Total: 11 polymer-specific features** (domain knowledge-based)
- num_side_chains, backbone_carbons, aromatic_count
- h_bond_donors, h_bond_acceptors, num_rings
- single_bonds, halogen_count, heteroatom_count
- mw_estimate, branching_ratio

### 1.3 Simple Features
**Total: 10 simple features** (SMILES string counting)
- smiles_length, carbon_count, nitrogen_count, oxygen_count
- sulfur_count, fluorine_count, ring_count, double_bond_count
- triple_bond_count, branch_count

### 1.4 Current Baseline
```
Current Setup: 10 simple + 11 hand-crafted + 13 RDKit = 34 features
AutoGluon Selection: 19 features selected (56% of input)
```

---

## 2. Systematic Feature Subsets to Test

### 2.1 Baseline Configurations

**Configuration A: Simple Only (Control)**
- 10 simple features
- Baseline for SMILES string parsing alone
- Expected: Poor performance (but fast)

**Configuration B: Hand-Crafted Only**
- 11 hand-crafted features
- Pure domain knowledge, no RDKit
- Expected: Good performance (current best w/o RDKit)

**Configuration C: Current (10 + 11 + 13)**
- 34 features (10 simple + 11 hand-crafted + 13 RDKit)
- Current working setup

### 2.2 RDKit Expansion Levels

**Configuration D: Expanded RDKit (35 descriptors)**
- All 10 simple + 11 hand-crafted + 35 RDKit
- Test if more RDKit helps
- Total: 56 features

**Configuration E: All RDKit (~200 descriptors)**
- All 10 simple + 11 hand-crafted + ALL RDKit
- Comprehensive screening (AutoGluon does feature selection)
- Total: ~221 features
- Expected: AutoGluon will heavily filter to find best subset

**Configuration F: RDKit Only (35 descriptors)**
- Skip hand-crafted, use 35 RDKit directly
- Isolate RDKit performance
- Total: 45 features

### 2.3 Ablation Study

**Configuration G: Remove Simple Features**
- 11 hand-crafted + 13 RDKit (no simple)
- Test if simple string-based features add value
- Total: 24 features

**Configuration H: Remove Hand-Crafted Features**
- 10 simple + 13 RDKit (no domain knowledge)
- Test if domain expertise adds value
- Total: 23 features

---

## 3. RDKit Descriptor Categories

To strategically expand RDKit descriptors, group them by category:

### 3.1 Key RDKit Descriptor Categories (~200 total)

**Molecular Weight & Size (5)**
- MolWt, MolFormula, ExactMolWt, HeavyAtomCount, NAtoms

**Lipophilicity (3)**
- LogP (MolLogP), TPSA, LabuteASA

**H-Bonding (4)**
- NumHDonors, NumHAcceptors, NumHeteroatoms, NumSaturatedCycles

**Aromaticity (6)**
- NumAromaticRings, NumAromaticCarbocycles, NumAromaticHeterocycles, FractionCsp3, etc.

**Rings & Cycles (8)**
- RingCount, NumRings, NumSaturatedRings, NumAliphaticRings, etc.

**Bonds (5)**
- NumRotatableBonds, NumAmideBonds, NumSulfonamideBonds, etc.

**Topological/Structural (40+)**
- Eccentricity, LabuteELF10, PercentVSA, SurfaceArea, etc.

**Scaffold/Graph (30+)**
- Ipc, Chi0, Chi1, Chi2, etc. (topological indices)

**Elemental Composition (20+)**
- Elemental counts: C, H, N, O, S, F, Cl, Br, I, etc.

**Other specialized (80+)**
- BCUT, MolFormula, MolWt variants, etc.

---

## 4. Testing Strategy

### 4.1 Scalability Considerations

**Training Time Estimate:**
- Simple features (10): ~2 min
- Hand-crafted (11): ~2 min
- Current (34): ~3 min
- Expanded (56): ~4 min
- All RDKit (221): ~10-15 min (AutoGluon's internal feature selection helps)

**Why more features OK:**
- AutoGluon internally filters features via feature importance
- Feature selection happens DURING training, not before
- More features = more to optimize, but not exponentially slower

### 4.2 Proposed Test Sequence

```
Phase 1: Quick Baselines (5-10 min total)
├─ A: Simple only          (10 features)
├─ B: Hand-crafted only    (11 features)
└─ C: Current baseline     (34 features)

Phase 2: RDKit Expansion (15-20 min total)
├─ D: Expanded RDKit       (56 features)
├─ E: All RDKit           (221 features)
└─ F: RDKit only          (45 features)

Phase 3: Ablation Study (10-15 min total)
├─ G: No simple features   (24 features)
└─ H: No hand-crafted      (23 features)
```

**Total Estimated Time: 30-45 minutes for all 8 configurations**

---

## 5. Implementation Plan

### 5.1 Script Structure

**New script: `train_systematic_features.py`**

```python
FEATURE_CONFIGURATIONS = {
    'A_simple_only': {
        'simple': True,
        'hand_crafted': False,
        'rdkit': False,
        'description': 'SMILES string counting only'
    },
    'B_hand_crafted_only': {
        'simple': False,
        'hand_crafted': True,
        'rdkit': False,
        'description': 'Domain knowledge only'
    },
    'C_current_baseline': {
        'simple': True,
        'hand_crafted': True,
        'rdkit': ['current_13'],  # Current 13 descriptors
        'description': 'Current setup (34 features)'
    },
    'D_expanded_rdkit': {
        'simple': True,
        'hand_crafted': True,
        'rdkit': ['expanded_35'],  # More RDKit descriptors
        'description': 'Expanded RDKit (56 features)'
    },
    'E_all_rdkit': {
        'simple': True,
        'hand_crafted': True,
        'rdkit': ['all_200'],  # All available RDKit
        'description': 'All RDKit descriptors (~221 features)'
    },
    'F_rdkit_only_expanded': {
        'simple': False,
        'hand_crafted': False,
        'rdkit': ['expanded_35'],
        'description': 'RDKit only, expanded (45 features)'
    },
    'G_no_simple': {
        'simple': False,
        'hand_crafted': True,
        'rdkit': ['current_13'],
        'description': 'No simple features (24 features)'
    },
    'H_no_hand_crafted': {
        'simple': True,
        'hand_crafted': False,
        'rdkit': ['current_13'],
        'description': 'No hand-crafted features (23 features)'
    }
}
```

### 5.2 Key Functions to Create

1. **`get_rdkit_descriptors(category='all')`**
   - Return list of RDKit descriptors by category
   - 'current_13', 'expanded_35', 'all_200'

2. **`extract_features(df, config_key)`**
   - Load configuration from FEATURE_CONFIGURATIONS
   - Extract selected feature set
   - Return features_df with only chosen features

3. **`train_and_evaluate(train_df, features_df, config_key)`**
   - Train AutoGluon for each property
   - Save models to config-specific directory
   - Log results (feature count, training time, etc.)

4. **`compare_all_configurations()`**
   - Run all 8 configurations sequentially
   - Create comparison report
   - Show feature selection patterns across configs

### 5.3 Output Structure

```
AutoGluon/systematic_feature/
├── results/
│   ├── A_simple_only/
│   │   ├── Tg/                    # AutoGluon models
│   │   ├── FFV/
│   │   ├── results.json           # Performance metrics
│   │   └── feature_analysis.txt
│   ├── B_hand_crafted_only/
│   ├── C_current_baseline/
│   ├── ... (6 more configs)
│   └── comparison_report.csv      # Side-by-side comparison
├── train_systematic_features.py   # Main script
├── RESULTS_SUMMARY.md             # Findings & recommendations
└── PLAN.md                        # This file
```

---

## 6. Success Metrics

For each configuration, track:

1. **Feature Efficiency**
   - Input features: How many features given?
   - Selected features: How many did AutoGluon keep?
   - Reduction ratio: Input / Selected (lower = better feature redundancy)

2. **Performance**
   - MAE (Mean Absolute Error) on validation set
   - R² Score
   - Training time

3. **Pattern Discovery**
   - Which features did AutoGluon rank as most important?
   - Do hand-crafted features consistently beat RDKit alone?
   - Is there an optimal balance?

---

## 7. Expected Findings

### 7.1 Hypothesis

1. **Configuration C (current) should perform well** because it combines domain knowledge + chemistry

2. **Configuration E (all RDKit) might find NEW useful descriptors** that we missed manually

3. **Configuration B (hand-crafted only) will be surprisingly good** - domain knowledge is powerful

4. **Configuration A (simple only) will be weakest** - but establishes baseline

5. **AutoGluon will filter heavily from 221 → ~15-20 features** when given all RDKit

### 7.2 Key Question

**Which is better?**
- Manually selected 13 RDKit descriptors (current)
- OR AutoGluon-selected from 200+ RDKit descriptors?

---

## 8. Next Steps

1. ✅ **Document plan** (this file)
2. **Extract all RDKit descriptors** into categorized lists
3. **Modify train_v85_best.py** → `train_systematic_features.py`
4. **Add feature selection logic** for each configuration
5. **Run all 8 configurations** with time tracking
6. **Analyze results** and create comparison report
7. **Generate recommendations** for best feature subset

---

## Timeline

- **Documentation:** ✅ Complete
- **Implementation:** 30 min
- **Testing all 8 configs:** 45 min
- **Analysis & Report:** 15 min
- **Total:** ~1.5 hours

---

## 9. Inference with Trained Models

After training with `train_for_colab_serial.py`, use the trained models for inference with `inference.py`.

### 9.1 Usage in Colab

**Step 1: Train models for a configuration (e.g., config G)**
```python
%run /content/open_polymer/AutoGluon/systematic_feature/train_for_colab_serial.py --config G --time_limit 300
```

Models will be saved to: `/content/autogluon_results/G/{Tg,FFV,Tc,Density,Rg}`

**Step 2: Run inference on the same configuration**
```python
%run /content/open_polymer/AutoGluon/systematic_feature/inference.py --config G
```

Predictions will be saved to: `/content/inference_results_config_G.csv`

### 9.2 Model Directory Structure

Each trained configuration has the following structure:

```
/content/autogluon_results/
├── A/  (Simple only)
│   ├── Tg/
│   ├── FFV/
│   ├── Tc/
│   ├── Density/
│   └── Rg/
├── B/  (Hand-crafted only)
│   ├── Tg/
│   ├── ...
├── C/  (Current baseline)
│   ├── Tg/
│   ├── ...
├── G/  (No simple features)
│   ├── Tg/
│   │   ├── AutoGluon_metadata.json
│   │   ├── model.pkl
│   │   ├── ...
│   ├── FFV/
│   ├── ...
```

Each target property folder contains:
- **AutoGluon_metadata.json** - Training configuration and feature selection
- **model.pkl** - The trained predictor object
- **data** folder - Input features used during training
- **config_results.json** - Summary of training (saved at parent level)

### 9.3 Feature Extraction by Configuration

The `inference.py` script automatically extracts the correct features based on `--config`:

| Config | Simple | Hand-crafted | RDKit | Total |
|--------|--------|--------------|-------|-------|
| A      | ✅ 10  | ❌           | ❌    | 10    |
| B      | ❌     | ✅ 11        | ❌    | 11    |
| C      | ✅ 10  | ✅ 11        | ✅ 13 | 34    |
| G      | ❌     | ✅ 11        | ✅ 13 | 24    |
| H      | ✅ 10  | ❌           | ✅ 13 | 23    |

### 9.4 Output

The `inference.py` script produces:
- **CSV predictions:** `inference_results_config_{CONFIG}.csv`
- **Console output:** Detailed logging of feature extraction and model loading
- **Statistics:** Min/max/mean/std for each property

Example output:
```
📂 Project root: /content/open_polymer
======================================================================
LOADING AUTOGLUON MODELS (Configuration G)
======================================================================
Model directory: /content/autogluon_results/G

📂 Loading Tg... ✅
   Features: 11
📂 Loading FFV... ✅
   Features: 11
...
```

### 9.5 Notes

- **Inference uses ONLY the features the model was trained on** - AutoGluon's feature selection is honored
- **Path detection is automatic** - Works with both `/content/open_polymer` and `/content/drive/MyDrive/open_polymer`
- **Data augmentation is applied during inference** - Uses the same datasets as training (Tc, Tg, PI1070, LAMALAB, pseudo-labels)


