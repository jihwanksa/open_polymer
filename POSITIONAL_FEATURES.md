# Positional Features - Radical Position Analysis

## Overview
v33 adds sophisticated positional features that capture the structural context of radical attachment points (*) in monomers. These features go beyond simple character counts to measure chemical topology.

## Features Extracted

### 1. **graph_star_distance** (int)
- **Definition**: Topological distance (number of bonds) between the two radicals
- **Range**: 1-20+ (typically)
- **Use Case**: Captures monomer backbone length
- **Example**: 
  - Linear: `*-C-C-C-C-C-*` → distance = 5
  - Branched: `*-C(-C)(-C)-C-*` → distance = 3

### 2. **rings_between_stars** (int)
- **Definition**: Count of distinct rings that intersect the shortest path between radicals
- **Range**: 0-5+ (typically)
- **Use Case**: Differentiates aromatic vs. aliphatic backbones
- **Example**:
  - Linear: `*-C-C-C-C-*` → 0 rings
  - Aromatic: `*-Ph-Ph-*` → 2 rings

### 3. **ecfp_similarity_stars** (float)
- **Definition**: Tanimoto similarity of Morgan fingerprints around each radical (0-1 scale)
- **Range**: 0.0-1.0
- **Use Case**: Measures symmetry of polymerization sites
- **Example**:
  - Symmetric: `*-C(=O)-NH-C(=O)-*` → high similarity (~0.8)
  - Asymmetric: `*-Ph-C(=O)-CH2-*` → low similarity (~0.3)

### 4. **peri_flag** (binary)
- **Definition**: Heuristic for peri-like topologies (e.g., 1,8-naphthalene positions)
- **Range**: 0 or 1
- **Use Case**: Identifies strained/close fused aromatic systems
- **Logic**: True if distance ≤4 AND radicals share common neighbors

### 5. **radical_distance** (int)
- **Definition**: Bond count between radicals (similar to graph_star_distance but computed differently)
- **Range**: 1-20+ (typically)
- **Use Case**: Alternative/redundant measure for backbone length

## Why These Features Matter

1. **Structural Diversity**: Different monomer topologies (linear, cyclic, aromatic) affect polymer properties
2. **Property Prediction**: Tc, Tg, and other properties often correlate with backbone structure
3. **Informed Features**: Unlike simple character counts, these capture chemical meaning
4. **Redundancy Check**: Having multiple distance measures helps validate model robustness

## Integration in v33

- Added after data augmentation (Tc, Tg)
- Computed for both train and test sets
- Used alongside the original 10 simple features
- Total feature count: 10 (simple) + 5 (positional) = **15 features**

## Expected Impact

**Potential Benefits:**
- ✅ Better capture of structural effects on properties
- ✅ Improved generalization for diverse backbone types
- ✅ Interpretable features vs. fingerprints

**Potential Risks:**
- ⚠️ Feature redundancy (distance measures correlate)
- ⚠️ Sparsity issues if many molecules have invalid radical positions
- ⚠️ Curse of dimensionality with 15 features on ~2,500 samples

**Baseline for Comparison:**
- v32: 0.085 (10 simple features)
- v33: ? (10 simple + 5 positional = 15 features)

## Testing Strategy

1. Push v33 to Kaggle
2. Compare score vs v32 (0.085 baseline)
3. If score improves: Keep positional features
4. If score worsens: Revert to v32 or try feature selection
