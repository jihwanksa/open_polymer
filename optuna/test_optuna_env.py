#!/usr/bin/env python3
"""
Test script to verify Optuna environment is ready
"""

print("Testing Optuna environment...")
print("="*70)

# Test imports
try:
    import pandas as pd
    print("✅ pandas:", pd.__version__)
except ImportError as e:
    print("❌ pandas not available:", e)

try:
    import numpy as np
    print("✅ numpy:", np.__version__)
except ImportError as e:
    print("❌ numpy not available:", e)

try:
    import sklearn
    print("✅ scikit-learn:", sklearn.__version__)
except ImportError as e:
    print("❌ scikit-learn not available:", e)

try:
    import optuna
    print("✅ optuna:", optuna.__version__)
except ImportError as e:
    print("❌ optuna not available:", e)
    print("   Install with: pip install optuna")

try:
    from tqdm import tqdm
    print("✅ tqdm: available")
except ImportError:
    print("⚠️  tqdm not available (optional, for progress bars)")
    print("   Install with: pip install tqdm")

print("="*70)
print("\nIf all packages show ✅, you're ready to run:")
print("  python optuna_tune_rf.py")
print("\nIf any show ❌, install with:")
print("  pip install <package-name>")

