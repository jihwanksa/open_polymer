"""
Evaluate comprehensive feature set (2647 features) with AutoGluon.

Refactored version with clean separation:
1. Data loading
2. Feature extraction (with caching)
3. Model training/loading (with checkpointing)
4. Feature importance (with caching)
5. Results aggregation
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure environment for GPU
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    del os.environ['CUDA_VISIBLE_DEVICES']
os.environ['RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'] = '1'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import feature extraction
from AutoGluon.systematic_feature.all_features import extract_all_features

# Import AutoGluon
try:
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon imported")
except ImportError:
    print("❌ AutoGluon not installed")
    sys.exit(1)

# Import RDKit
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
    print("✅ RDKit imported")
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available")


# ============================================================================
# DATA LOADING
# ============================================================================

def make_smile_canonical(smile):
    """Canonicalize SMILES"""
    if not RDKIT_AVAILABLE:
        return smile
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return np.nan


def load_and_augment_data(output_dir):
    """Load and augment training data (same as v85)"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Check for cached data
    train_df_path = output_dir / 'train_data.parquet'
    if train_df_path.exists():
        print(f"💾 Loading cached training data...")
        train_df = pd.read_parquet(train_df_path)
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        print(f"✅ {len(train_df)} samples")
        for col in target_cols:
            n = train_df[col].notna().sum()
            print(f"   {col}: {n} ({n/len(train_df)*100:.1f}%)")
        return train_df, target_cols
    
    # Load main data
    train_df = pd.read_csv(project_root / 'data/raw/train.csv')
    print(f"✅ Loaded {len(train_df)} samples")
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Canonicalize
    if RDKIT_AVAILABLE:
        train_df['SMILES'] = train_df['SMILES'].apply(make_smile_canonical)
        train_df['SMILES'] = train_df['SMILES'].fillna(train_df['SMILES'].bfill())
    
    # Augment with external datasets
    datasets = [
        ('Tc_SMILES.csv', 'Tc', 'TC_mean'),
        ('Tg_SMILES_class_pid_polyinfo_median.csv', 'Tg', 'Tg'),
        ('PI1070.csv', None, None),  # Special handling
        ('LAMALAB_CURATED_Tg_structured_polymerclass.csv', 'Tg', 'labels.Exp_Tg(K)'),
        ('PI1M_50000_v2.1.csv', None, None),  # Pseudo-labels
    ]
    
    for filename, target, value_col in datasets:
        try:
            paths = [project_root / 'data' / filename, project_root / 'data/raw' / filename]
            df = None
            for path in paths:
                if path.exists():
                    df = pd.read_csv(path)
                    break
            
            if df is None:
                continue
            
            # Handle different formats
            if 'PI1070' in filename:
                # Has density and Rg
                df = df.rename(columns={'smiles': 'SMILES'})
                for _, row in df.iterrows():
                    if row['SMILES'] not in train_df['SMILES'].values:
                        train_df = pd.concat([train_df, pd.DataFrame([{
                            'SMILES': row['SMILES'],
                            'Tg': np.nan, 'FFV': np.nan, 'Tc': np.nan,
                            'Density': row.get('density', np.nan),
                            'Rg': row.get('Rg', np.nan)
                        }])], ignore_index=True)
            
            elif 'LAMALAB' in filename:
                # Convert K to C
                df = df.rename(columns={'PSMILES': 'SMILES', value_col: 'Tg'})
                df['Tg'] = df['Tg'] - 273.15
                for _, row in df[df['Tg'].notna()].iterrows():
                    if row['SMILES'] not in train_df['SMILES'].values:
                        train_df = pd.concat([train_df, pd.DataFrame([{
                            'SMILES': row['SMILES'],
                            'Tg': row['Tg'],
                            'FFV': np.nan, 'Tc': np.nan, 'Density': np.nan, 'Rg': np.nan
                        }])], ignore_index=True)
            
            elif 'PI1M' in filename:
                # Pseudo-labels
                df_new = df[~df['SMILES'].isin(train_df['SMILES'])]
                if RDKIT_AVAILABLE:
                    df_new['SMILES'] = df_new['SMILES'].apply(make_smile_canonical)
                train_df = pd.concat([train_df, df_new], ignore_index=True)
                print(f"✅ Added {len(df_new)} pseudo-labels")
            
            else:
                # Standard format
                df = df.rename(columns={value_col: target})
                for _, row in df[df[target].notna()].iterrows():
                    if row['SMILES'] not in train_df['SMILES'].values:
                        new_row = {'SMILES': row['SMILES'], target: row[target]}
                        for col in target_cols:
                            if col not in new_row:
                                new_row[col] = np.nan
                        train_df = pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True)
        
        except Exception as e:
            print(f"⚠️  {filename}: {e}")
    
    train_df = train_df.reset_index(drop=True)
    print(f"\n📊 Final: {len(train_df)} samples")
    for col in target_cols:
        n = train_df[col].notna().sum()
        print(f"   {col}: {n} ({n/len(train_df)*100:.1f}%)")
    
    # Save for future runs
    train_df.to_parquet(train_df_path)
    print(f"💾 Cached training data")
    
    return train_df, target_cols


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_single(args):
    """Extract features for one SMILES"""
    idx, smiles, include_3d, include_fp_arrays = args
    try:
        features = extract_all_features(smiles, include_3d, include_fp_arrays)
        return (idx, features, False)
    except:
        try:
            features = extract_all_features(smiles, False, include_fp_arrays)
            return (idx, features, True)
        except:
            return (idx, {}, True)


def extract_or_load_features(train_df, output_dir, include_3d=True, include_fp_arrays=True):
    """Extract features or load from cache"""
    features_path = output_dir / 'extracted_features.parquet'
    
    if features_path.exists():
        print(f"\n💾 Loading cached features...")
        features_df = pd.read_parquet(features_path)
        print(f"✅ {len(features_df)} samples, {len(features_df.columns)} features")
        return features_df
    
    print(f"\n🔬 Extracting features...")
    n_jobs = max(1, cpu_count() - 1)
    print(f"   Using {n_jobs} CPU cores")
    
    args_list = [(idx, smiles, include_3d, include_fp_arrays) 
                 for idx, smiles in train_df['SMILES'].items()]
    
    features_dict = {}
    failed_count = 0
    
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap(extract_features_single, args_list),
                           total=len(args_list), desc="Extracting"))
    
    for idx, features, failed in results:
        features_dict[idx] = features
        if failed:
            failed_count += 1
    
    features_df = pd.DataFrame.from_dict(features_dict, orient='index')
    features_df = features_df.reindex(train_df.index).fillna(0)
    
    # Save
    features_df.to_parquet(features_path)
    
    print(f"✅ {len(features_df.columns)} features extracted")
    if failed_count > 0:
        print(f"⚠️  {failed_count} failed 3D (used 2D fallback)")
    
    return features_df


# ============================================================================
# MODEL TRAINING/LOADING
# ============================================================================

def train_or_load_model(target, train_df, features_df, output_dir, time_limit):
    """Train new model or load existing"""
    model_path = output_dir / target
    
    # Prepare data
    mask = train_df[target].notna()
    X = features_df[mask].copy()
    y = train_df.loc[mask, target].copy()
    
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    train_data = X.copy()
    train_data[target] = y
    
    print(f"📊 {len(X)} samples, {len(X.columns)} features")
    print(f"📊 Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Load existing
    if (model_path / 'predictor.pkl').exists():
        print(f"✅ Loading existing model...")
        try:
            predictor = TabularPredictor.load(str(model_path))
            return predictor, train_data, 0
        except Exception as e:
            print(f"⚠️  Load failed: {e}, retraining...")
    
    # Train new
    model_path.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Training (time_limit={time_limit}s)...")
    start = time.time()
    
    # Choose preset based on time limit
    preset = 'optimize_for_deployment' if time_limit < 300 else 'medium_quality'
    
    predictor = TabularPredictor(
        label=target,
        path=str(model_path),
        eval_metric='mean_absolute_error',
        problem_type='regression'
    ).fit(
        train_data,
        time_limit=time_limit,
        presets=preset,
        verbosity=2,
        num_bag_folds=2 if time_limit < 300 else 3,
        num_bag_sets=1,
        num_stack_levels=0 if time_limit < 300 else 1,
        hyperparameters={
            'NN_TORCH': [{'ag_args_fit': {'num_gpus': 1}}],
            'FASTAI': [{'ag_args_fit': {'num_gpus': 1}}],
            'GBM': [{'ag_args_fit': {'num_gpus': 0}}],
            'XGB': [{'ag_args_fit': {'num_gpus': 0}}],
            'CAT': [{'ag_args_fit': {'num_gpus': 0}}],
            'RF': [{'ag_args_fit': {'num_gpus': 0}}],
            'XT': [{'ag_args_fit': {'num_gpus': 0}}],
        }
    )
    
    return predictor, train_data, time.time() - start


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def compute_or_load_importance(predictor, train_data, target, output_dir, subsample_size=2000, num_shuffle_sets=3):
    """
    Compute feature importance or load from cache.
    Returns (importance_series, full_dataframe_with_stats).
    """
    importance_path = output_dir / target / 'feature_importance.parquet'
    
    # Load cached
    if importance_path.exists():
        print(f"   ✅ Loading cached importance...")
        imp_df = pd.read_parquet(importance_path)
        if 'importance' in imp_df.columns:
            return imp_df['importance'], imp_df
        else:
            return imp_df.iloc[:, 0], imp_df
    
    # Compute
    print(f"   📊 Computing importance ({num_shuffle_sets} shuffles, {subsample_size} samples)...")
    try:
        importance = predictor.feature_importance(
            train_data,
            feature_stage='original',
            subsample_size=subsample_size,
            num_shuffle_sets=num_shuffle_sets,
            silent=False
        )
    except Exception as e:
        print(f"   ⚠️  Permutation failed: {e}")
        print(f"   Using model-native importance...")
        leaderboard = predictor.leaderboard(train_data, silent=True)
        best_model = leaderboard.iloc[0]['model']
        importance = predictor.get_model_attribute(best_model, 'feature_importance')
        if importance is None:
            importance = pd.Series(1.0, index=train_data.columns.drop(target))
    
    # Save and return
    if isinstance(importance, pd.DataFrame):
        importance.to_parquet(importance_path)
        print(f"   💾 Saved importance with stats: {list(importance.columns)}")
        return importance['importance'], importance
    else:
        imp_df = pd.DataFrame({'importance': importance})
        imp_df.to_parquet(importance_path)
        print(f"   💾 Saved importance")
        return importance, imp_df
        return importance


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def evaluate_target(target, train_df, features_df, output_dir, time_limit, importance_samples=2000, num_shuffle_sets=3):
    """Evaluate one target"""
    print(f"\n{'='*80}")
    print(f"TARGET: {target}")
    print(f"{'='*80}")
    
    try:
        # Step 1: Train/load model
        predictor, train_data, training_time = train_or_load_model(
            target, train_df, features_df, output_dir, time_limit
        )
        
        # Step 2: Compute/load importance
        importance, importance_full = compute_or_load_importance(
            predictor, train_data, target, output_dir, importance_samples, num_shuffle_sets
        )
        
        # Step 3: Get performance
        leaderboard = predictor.leaderboard(train_data, silent=True)
        best_model = str(leaderboard.iloc[0]['model'])
        best_score = float(abs(leaderboard.iloc[0]['score_val']))
        
        # Step 4: Format results with stats
        if not isinstance(importance, pd.Series):
            importance = pd.Series(importance)
        
        # Top 10 with full stats
        top_10 = {}
        for feat in importance.head(10).index:
            feat_stats = {'importance': float(importance[feat])}
            # Add statistical metrics if available
            if isinstance(importance_full, pd.DataFrame) and feat in importance_full.index:
                for col in ['stddev', 'p_value', 'n', 'p99_high', 'p99_low']:
                    if col in importance_full.columns:
                        val = importance_full.loc[feat, col]
                        if pd.notna(val):
                            feat_stats[col] = float(val)
            top_10[str(feat)] = feat_stats
        
        top_50 = [str(x) for x in importance.head(50).index.tolist()]
        
        result = {
            'n_samples': int(len(train_data) - 1),
            'training_time': float(training_time),
            'best_model': best_model,
            'best_score_mae': best_score,
            'n_features_used': int(len(importance[importance > 0])),
            'top_10_features': top_10,
            'top_50_features': top_50
        }
        
        # Step 5: Print summary
        print(f"\n✅ Complete!")
        print(f"   Time: {training_time:.1f}s")
        print(f"   Model: {best_model}")
        print(f"   MAE: {best_score:.4f}")
        print(f"   Features: {result['n_features_used']}/{len(features_df.columns)}")
        print(f"\n   Top 10:")
        for feat, feat_stats in list(top_10.items())[:10]:
            imp_val = feat_stats['importance']
            print(f"      {feat}: {imp_val:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'n_samples': 0}


def main():
    """Main evaluation pipeline"""
    print("\n" + "="*80)
    print("🚀 FEATURE EVALUATION")
    print("="*80)
    
    # Config - FAST TEST MODE
    FAST_TEST = '--fast' in sys.argv or '-f' in sys.argv
    
    if FAST_TEST:
        print("⚡ FAST TEST MODE ENABLED")
        INCLUDE_3D = False  # Skip 3D (faster)
        INCLUDE_FP_ARRAYS = False  # Skip fingerprint arrays (fewer features)
        TIME_LIMIT = 60  # 1 minute per target
        IMPORTANCE_SAMPLES = 100  # Minimal samples for importance
        NUM_SHUFFLE_SETS = 3
    else:
        INCLUDE_3D = True
        INCLUDE_FP_ARRAYS = True
        TIME_LIMIT = 3600  # 1 hour per target
        IMPORTANCE_SAMPLES = 2000
        NUM_SHUFFLE_SETS = 3
    
    output_dir = project_root / 'AutoGluon' / 'systematic_feature' / 'evaluation_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data (with caching)
    train_df, target_cols = load_and_augment_data(output_dir)
    
    # Step 2: Extract features (with caching)
    features_df = extract_or_load_features(
        train_df, output_dir, INCLUDE_3D, INCLUDE_FP_ARRAYS
    )
    
    # Step 3: Evaluate each target
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(train_df),
            'n_features': len(features_df.columns),
            'time_limit_per_target': TIME_LIMIT
        },
        'targets': {}
    }
    
    for target in target_cols:
        result = evaluate_target(
            target, train_df, features_df, output_dir, TIME_LIMIT, 
            IMPORTANCE_SAMPLES, NUM_SHUFFLE_SETS
        )
        results['targets'][target] = result
        
        # Save checkpoint with error handling
        checkpoint_path = output_dir / f'results_checkpoint_{target}.json'
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Checkpoint: {checkpoint_path}")
        except (TypeError, ValueError) as e:
            print(f"\n⚠️  JSON serialization failed: {e}")
            print(f"\n📋 RAW RESULTS FOR {target}:")
            print(f"{result}")
            # Save as text fallback
            txt_path = output_dir / f'results_checkpoint_{target}.txt'
            with open(txt_path, 'w') as f:
                f.write(str(results))
            print(f"💾 Saved as text: {txt_path}")
    
    # Step 4: Save final results
    results_path = output_dir / 'evaluation_results.json'
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*80}")
        print(f"✅ COMPLETE!")
        print(f"{'='*80}")
        print(f"Results: {results_path}")
    except (TypeError, ValueError) as e:
        print(f"\n{'='*80}")
        print(f"⚠️  JSON SERIALIZATION FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        print(f"\n📋 RAW RESULTS (copy this):")
        print("="*80)
        import pprint
        pprint.pprint(results)
        print("="*80)
        # Save as text fallback
        txt_path = output_dir / 'evaluation_results.txt'
        with open(txt_path, 'w') as f:
            pprint.pprint(results, stream=f)
        print(f"\n💾 Saved as text: {txt_path}")
    
    # Summary
    for target, info in results['targets'].items():
        if 'error' not in info:
            print(f"\n{target}: MAE={info['best_score_mae']:.4f}, "
                  f"Features={info['n_features_used']}/{results['metadata']['n_features']}")


if __name__ == "__main__":
    main()
