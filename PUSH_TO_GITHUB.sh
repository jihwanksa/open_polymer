#!/bin/bash

# Quick script to push to GitHub
# Usage: bash PUSH_TO_GITHUB.sh

set -e

echo "=========================================="
echo "Pushing to GitHub: open_polymer"
echo "=========================================="

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files
echo "Adding files..."
git add .

# Create commit
echo "Creating commit..."
git commit -m "Initial commit: Polymer properties prediction framework

Features:
- XGBoost & Random Forest models (R²=0.698, 0.678)
- Graph Neural Network architecture
- Transformer (ChemBERTa) architecture  
- Comprehensive data preprocessing pipeline
- Molecular descriptors and fingerprints
- Complete documentation and results
- Best performance: R²=0.798 (Density prediction)

Models implemented:
- Traditional ML: XGBoost, Random Forest
- Deep Learning: GNN (GCN), Transformer (ChemBERTa)

Documentation:
- Comprehensive README with examples
- Quick start guide
- Detailed results analysis
- Contribution guidelines"

# Add remote (if not exists)
if ! git remote | grep -q origin; then
    echo "Adding remote repository..."
    git remote add origin https://github.com/jihwanksa/open_polymer.git
fi

# Rename branch to main
echo "Setting branch to main..."
git branch -M main

# Push to GitHub
echo "Pushing to GitHub..."
echo ""
echo "NOTE: You may need to enter your GitHub credentials"
echo "      Consider using a Personal Access Token for authentication"
echo ""

git push -u origin main

echo ""
echo "=========================================="
echo "✅ Successfully pushed to GitHub!"
echo "=========================================="
echo ""
echo "Visit your repository:"
echo "https://github.com/jihwanksa/open_polymer"
echo ""
echo "Next steps:"
echo "1. Add repository description"
echo "2. Add topics: machine-learning, chemistry, kaggle"
echo "3. Enable GitHub Issues"
echo "4. Review and update any settings"
echo ""
