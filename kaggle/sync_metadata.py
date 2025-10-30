#!/usr/bin/env python3
"""
Manual sync: Update kernel-metadata.json with dataset inputs

Kaggle API doesn't provide automatic input detection, so this script:
1. Lists known datasets used by the kernel
2. Updates kernel-metadata.json
3. You can then push with: kaggle kernels push -p .

Usage:
    python kaggle/sync_metadata.py

Or manually edit kernel-metadata.json and add datasets under "dataset_sources"
"""

import json
from pathlib import Path

KERNEL_METADATA_PATH = Path(__file__).parent.parent / "kernel-metadata.json"

# Known datasets for this kernel
KNOWN_DATASETS = [
    "wpixiu/rdkit-2025-3-3-cp311",           # RDKit wheel
    "minatoyukinaxlisa/tc-smiles",           # External Tc data  
    "akihiroorita/tg-of-polymer-dataset",    # External Tg data
]


def get_user_confirmation():
    """Ask user to confirm dataset list"""
    print("\n📋 Datasets to add:")
    for i, ds in enumerate(KNOWN_DATASETS, 1):
        print(f"   {i}. {ds}")
    
    response = input("\n✓ Add these datasets? (y/n): ").strip().lower()
    return response == 'y'


def update_metadata():
    """Update kernel-metadata.json with datasets"""
    
    if not KERNEL_METADATA_PATH.exists():
        print(f"❌ {KERNEL_METADATA_PATH} not found")
        return False
    
    with open(KERNEL_METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    old_datasets = set(metadata.get("dataset_sources", []))
    new_datasets = set(KNOWN_DATASETS)
    
    if old_datasets == new_datasets:
        print("✅ Datasets already up to date!")
        return True
    
    metadata["dataset_sources"] = sorted(KNOWN_DATASETS)
    
    # Write back
    with open(KERNEL_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Updated {KERNEL_METADATA_PATH}")
    return True


def main():
    print("=" * 70)
    print("Kernel Metadata Sync")
    print("=" * 70)
    
    if not get_user_confirmation():
        print("Cancelled.")
        return False
    
    if update_metadata():
        print("\n✅ Done! Now push with:")
        print("   kaggle kernels push -p .")
        return True
    
    return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
