# External Dataset Analysis

## Dataset 1: PI1070.csv (1070 rows)
**Source:** LAMALAB - Polymer simulations data

**Structure:**
- **SMILES column:** `smiles` (e.g., `*CC*`, `*C(C*)C`)
- **Target columns:** 
  - `density` (available)
  - `Rg` (radius of gyration, available)
  - Multiple thermal properties (Cp, Cv, thermal_conductivity, etc.)
  
**Usable properties:**
- ✅ **Density**: ~1070 samples (can augment our Density)
- ✅ **Rg**: ~1070 samples (can augment our Rg)

**Quality:** High - comes from molecular simulations
**Format:** All rows have SMILES + numerical properties

---

## Dataset 2: LAMALAB_CURATED_Tg_structured_polymerclass.csv (7369 rows)
**Source:** LAMALAB - Curated Tg literature data

**Structure:**
- **SMILES column:** `PSMILES` (polymer SMILES)
- **Target column:** `labels.Exp_Tg(K)` (Tg in Kelvin)
- Additional metadata and features

**Usable properties:**
- ✅ **Tg**: ~7369 samples (can significantly augment our Tg)

**Quality:** Curated from literature (varies)
**Data quantity:** Much larger than PI1070

---

## Integration Strategy

### Option 1: AGGRESSIVE (Use both fully)
```
New training data:
- Current Tg: 511 samples
- + LAMALAB Tg: 7369 samples
- = ~7,880 Tg samples (15x increase!)

- Current Density: 613 samples  
- + PI1070 Density: 1070 samples
- = ~1,683 Density samples (2.7x increase)

- Current Rg: 614 samples
- + PI1070 Rg: 1070 samples  
- = ~1,684 Rg samples (2.7x increase)
```

### Option 2: CONSERVATIVE (Check for overlap first)
1. Check SMILES overlap between Kaggle and external data
2. Only add non-overlapping samples
3. Check for outliers and data quality issues

### Option 3: SELECTIVE (Use high-quality subsets)
- Filter LAMALAB Tg by reliability score
- Use only validated Density from PI1070
- Only add samples that pass outlier checks

---

## Recommendation

**Use Option 2:** Check overlap, then aggressively add data
- Tg has huge potential (7x increase)
- Density has good potential (2.7x)
- Both properties need more samples
- Risk: Different data distributions may hurt model

**Key Actions:**
1. Load both datasets
2. Extract SMILES, canonicalize them
3. Check overlap with train_df
4. Add non-overlapping samples
5. Apply same outlier filtering
6. Re-train model with augmented data
7. Compare score (should improve with more Tg data!)

**Expected Impact:**
- More training data → Better generalization
- Especially for Tg where we have 7x more samples
- Could potentially reach 0.080-0.082 range
