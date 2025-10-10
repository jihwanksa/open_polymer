# Git Setup Instructions

## Initial Repository Setup

Follow these steps to push your code to GitHub:

### 1. Initialize Git Repository

```bash
cd /home/jihwanoh/chem
git init
```

### 2. Add All Files

```bash
git add .
```

### 3. Review What Will Be Committed

```bash
git status
```

You should see:
- ✅ All source code files
- ✅ Documentation files
- ✅ Configuration files (requirements.txt, .gitignore, etc.)
- ❌ Large model files (*.pkl) - these are in .gitignore
- ❌ Data files (*.csv) - configured for Git LFS if needed

### 4. Create Initial Commit

```bash
git commit -m "Initial commit: Polymer properties prediction framework

- Implement XGBoost and Random Forest models (R²=0.698, 0.678)
- Add Graph Neural Network architecture
- Add Transformer (ChemBERTa) architecture
- Include comprehensive data preprocessing pipeline
- Add molecular descriptors and fingerprint extraction
- Provide detailed documentation and results
- Achieve R²=0.798 for Density prediction"
```

### 5. Add Remote Repository

```bash
git remote add origin https://github.com/jihwanksa/open_polymer.git
```

### 6. Verify Remote

```bash
git remote -v
```

Should show:
```
origin  https://github.com/jihwanksa/open_polymer.git (fetch)
origin  https://github.com/jihwanksa/open_polymer.git (push)
```

### 7. Push to GitHub

```bash
# For first push
git push -u origin main

# Or if using master branch
git branch -M main
git push -u origin main
```

## Git LFS Setup (Optional - for large files)

If you want to include model files and data:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "*.csv"
git lfs track "*.pt"

# Add .gitattributes
git add .gitattributes

# Now add and commit large files
git add models/
git add data/raw/*.csv
git commit -m "Add trained models and data via Git LFS"
git push
```

## Repository Status

### Files Included ✅
- [x] Source code (`src/`)
- [x] Documentation (`docs/`, `README.md`)
- [x] Configuration (`requirements.txt`, `.gitignore`)
- [x] Scripts (`scripts/`)
- [x] License (`LICENSE`)
- [x] Results metadata (`results/model_comparison.csv`)

### Files Excluded ❌ (via .gitignore)
- [ ] Model checkpoints (`models/*.pkl`) - 73 MB total
- [ ] Raw data files (`data/raw/*.csv`) - 691 KB
- [ ] Cached files (`__pycache__/`)
- [ ] Temporary files

### Repository Size
- **Without models/data:** ~100 KB (lightweight)
- **With results:** ~200 KB
- **With models (via LFS):** ~73 MB
- **With data (via LFS):** ~74 MB

## Recommended Approach

**Option 1: Lightweight Repository (Recommended)**
```bash
# Push code only, users download data from Kaggle
git add .
git commit -m "Initial commit"
git push -u origin main
```

**Option 2: Include Pre-trained Models**
```bash
# Setup Git LFS first
git lfs install
git lfs track "*.pkl"
git add .gitattributes models/
git commit -m "Add pre-trained models via Git LFS"
git push
```

**Option 3: Include Everything**
```bash
# Setup Git LFS for models and data
git lfs install
git lfs track "*.pkl" "*.csv"
git add .
git commit -m "Initial commit with models and data"
git push
```

## Post-Push Steps

1. **Verify on GitHub**: Visit https://github.com/jihwanksa/open_polymer
2. **Add Topics**: Machine Learning, Chemistry, Kaggle, PyTorch, XGBoost
3. **Enable Issues**: For bug reports and feature requests
4. **Add Description**: "ML framework for polymer property prediction"
5. **Update README**: Add any GitHub-specific badges or links

## Troubleshooting

### Authentication Issues
```bash
# Use personal access token
git remote set-url origin https://YOUR_TOKEN@github.com/jihwanksa/open_polymer.git
```

### Large File Warning
```bash
# If you see warnings about large files
git rm --cached models/*.pkl
git commit --amend
git push -f  # Careful with force push
```

### Branch Name Issues
```bash
# Rename branch to main if needed
git branch -M main
git push -u origin main
```

## Quick Reference

```bash
# Complete setup in one go
cd /home/jihwanoh/chem
git init
git add .
git commit -m "Initial commit: Polymer properties prediction framework"
git remote add origin https://github.com/jihwanksa/open_polymer.git
git branch -M main
git push -u origin main
```

## Next Steps After Push

1. ✅ Add GitHub repository description
2. ✅ Add topics/tags
3. ✅ Enable GitHub Actions (optional CI/CD)
4. ✅ Add CHANGELOG.md for future releases
5. ✅ Create GitHub releases for versions
6. ✅ Add shields.io badges to README

---

**Repository URL:** https://github.com/jihwanksa/open_polymer

**Ready to push!** 🚀

