#!/usr/bin/env python3
"""
Inspect datasets available in /kaggle/input

Run this in your Kaggle notebook to see what datasets are available and their structure.
"""

import os
import pandas as pd

KAGGLE_INPUT = '/kaggle/input'

def inspect_datasets():
    """List and inspect all datasets in /kaggle/input"""
    
    if not os.path.exists(KAGGLE_INPUT):
        print(f"❌ {KAGGLE_INPUT} not found (not running on Kaggle?)")
        return
    
    print("=" * 70)
    print("AVAILABLE DATASETS IN /kaggle/input")
    print("=" * 70)
    
    datasets = sorted(os.listdir(KAGGLE_INPUT))
    
    for dataset in datasets:
        dataset_path = os.path.join(KAGGLE_INPUT, dataset)
        
        if not os.path.isdir(dataset_path):
            continue
        
        files = os.listdir(dataset_path)
        print(f"\n📁 {dataset}/ ({len(files)} files)")
        
        # Show first CSV files found
        csv_files = [f for f in files if f.endswith('.csv')]
        for csv_file in csv_files[:3]:  # Show first 3 CSVs
            csv_path = os.path.join(dataset_path, csv_file)
            try:
                df = pd.read_csv(csv_path, nrows=2)
                print(f"   📄 {csv_file}")
                print(f"      Columns: {list(df.columns)}")
                print(f"      Shape: {pd.read_csv(csv_path).shape}")
            except Exception as e:
                print(f"   ⚠️  {csv_file} - Error: {e}")

if __name__ == "__main__":
    inspect_datasets()
