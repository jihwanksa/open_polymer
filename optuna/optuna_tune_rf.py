"""
Optuna Hyperparameter Optimization for Random Forest
Uses correct competition metric (wMAE) for local validation
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from tqdm import tqdm
import json
from datetime import datetime

# Don't use complex wMAE - use simple MAE like successful manual configs!
# The issue: wMAE with validation stats ≠ wMAE with test stats
# Solution: Use simple unweighted MAE (what v53 optimized)


def create_chemistry_features(df):
    """Create 21 chemistry-based features from SMILES"""
    features = []
    
    for idx, smiles in tqdm(df['SMILES'].items(), total=len(df), desc="Creating features"):
        try:
            smiles_str = str(smiles) if pd.notna(smiles) else ""
            
            # Basic counts (10 features)
            basic = {
                'smiles_length': len(smiles_str),
                'carbon_count': smiles_str.count('C'),
                'nitrogen_count': smiles_str.count('N'),
                'oxygen_count': smiles_str.count('O'),
                'sulfur_count': smiles_str.count('S'),
                'fluorine_count': smiles_str.count('F'),
                'ring_count': smiles_str.count('c') + smiles_str.count('C1'),
                'double_bond_count': smiles_str.count('='),
                'triple_bond_count': smiles_str.count('#'),
                'branch_count': smiles_str.count('('),
            }
            
            # Chemistry features (11 features)
            num_side_chains = smiles_str.count('(')
            backbone_carbons = smiles_str.count('C') - smiles_str.count('C(')
            aromatic_count = smiles_str.count('c')
            h_bond_donors = smiles_str.count('O') + smiles_str.count('N')
            h_bond_acceptors = smiles_str.count('O') + smiles_str.count('N')
            num_rings = smiles_str.count('1') + smiles_str.count('2')
            single_bonds = len(smiles_str) - smiles_str.count('=') - smiles_str.count('#') - aromatic_count
            halogen_count = smiles_str.count('F') + smiles_str.count('Cl') + smiles_str.count('Br')
            heteroatom_count = smiles_str.count('N') + smiles_str.count('O') + smiles_str.count('S')
            mw_estimate = (smiles_str.count('C') * 12 + smiles_str.count('O') * 16 + 
                          smiles_str.count('N') * 14 + smiles_str.count('S') * 32 + 
                          smiles_str.count('F') * 19)
            branching_ratio = num_side_chains / max(backbone_carbons, 1)
            
            desc = {
                **basic,
                'num_side_chains': num_side_chains,
                'backbone_carbons': backbone_carbons,
                'aromatic_count': aromatic_count,
                'h_bond_donors': h_bond_donors,
                'h_bond_acceptors': h_bond_acceptors,
                'num_rings': num_rings,
                'single_bonds': single_bonds,
                'halogen_count': halogen_count,
                'heteroatom_count': heteroatom_count,
                'mw_estimate': mw_estimate,
                'branching_ratio': branching_ratio,
            }
            features.append(desc)
        except:
            # Fallback to zeros
            features.append({k: 0 for k in ['smiles_length', 'carbon_count', 'nitrogen_count', 
                                            'oxygen_count', 'sulfur_count', 'fluorine_count',
                                            'ring_count', 'double_bond_count', 'triple_bond_count',
                                            'branch_count', 'num_side_chains', 'backbone_carbons',
                                            'aromatic_count', 'h_bond_donors', 'h_bond_acceptors',
                                            'num_rings', 'single_bonds', 'halogen_count',
                                            'heteroatom_count', 'mw_estimate', 'branching_ratio']})
    
    return pd.DataFrame(features, index=df.index)


def load_and_prepare_data():
    """Load data and create features"""
    print("Loading data...")
    
    # Load training data
    train_df = pd.read_csv('../data/raw/train.csv')
    
    # Load external data for augmentation
    print("Loading external Tc data...")
    try:
        tc_external = pd.read_csv('../data/Tc_SMILES.csv')
        tc_external = tc_external.rename(columns={'TC_mean': 'Tc'})
        # Add non-overlapping samples
        train_smiles = set(train_df['SMILES'])
        tc_new = tc_external[~tc_external['SMILES'].isin(train_smiles)].copy()
        if len(tc_new) > 0:
            tc_new_rows = []
            for _, row in tc_new.iterrows():
                tc_new_rows.append({
                    'SMILES': row['SMILES'],
                    'Tg': np.nan, 'FFV': np.nan, 'Tc': row['Tc'],
                    'Density': np.nan, 'Rg': np.nan
                })
            train_df = pd.concat([train_df, pd.DataFrame(tc_new_rows)], ignore_index=True)
            print(f"  Added {len(tc_new)} Tc samples")
    except:
        print("  Tc data not found, skipping")
    
    print("Loading external Tg data...")
    try:
        tg_external = pd.read_csv('../data/Tg_SMILES_class_pid_polyinfo_median.csv')
        train_smiles = set(train_df['SMILES'])
        tg_new = tg_external[~tg_external['SMILES'].isin(train_smiles)].copy()
        if len(tg_new) > 0:
            tg_new_rows = []
            for _, row in tg_new.iterrows():
                tg_new_rows.append({
                    'SMILES': row['SMILES'],
                    'Tg': row['Tg'], 'FFV': np.nan, 'Tc': np.nan,
                    'Density': np.nan, 'Rg': np.nan
                })
            train_df = pd.concat([train_df, pd.DataFrame(tg_new_rows)], ignore_index=True)
            print(f"  Added {len(tg_new)} Tg samples")
    except:
        print("  Tg data not found, skipping")
    
    print(f"\nTotal training samples: {len(train_df)}")
    
    # Create features
    print("\nCreating chemistry features...")
    features_df = create_chemistry_features(train_df)
    
    # Prepare targets
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    y = train_df[target_cols].values
    X = features_df.values
    
    # Remove samples with NaN/inf in features
    feature_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[feature_mask]
    y = y[feature_mask]
    
    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Note: We don't need property statistics for simple MAE
    # But keep the return signature for compatibility
    n_samples_per_property = None
    ranges_per_property = None
    
    return X, y, target_cols, n_samples_per_property, ranges_per_property


def train_ensemble_rf(X_train, y_train, X_val, y_val, params, n_ensemble=5):
    """Train ensemble of Random Forest models"""
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    predictions = np.zeros((len(X_val), len(target_cols)))
    
    for i, target in enumerate(target_cols):
        y_train_target = y_train[:, i]
        train_mask = ~np.isnan(y_train_target)
        
        if train_mask.sum() == 0:
            continue
        
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train_target[train_mask]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filtered)
        X_val_scaled = scaler.transform(X_val)
        
        # Train ensemble
        ensemble_preds = []
        for j in range(n_ensemble):
            model = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=42 + i * 10 + j,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train_filtered)
            pred = model.predict(X_val_scaled)
            ensemble_preds.append(pred)
        
        # Average ensemble predictions
        predictions[:, i] = np.mean(ensemble_preds, axis=0)
    
    return predictions


def objective(trial, X, y, target_cols, n_samples_per_property, ranges_per_property):
    """Optuna objective function using competition wMAE metric"""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
    }
    
    # Train and predict
    y_pred = train_ensemble_rf(X_train, y_train, X_val, y_val, params, n_ensemble=3)
    
    # Calculate SIMPLE MAE (what v53 used!)
    # Don't use complex wMAE with validation stats - it doesn't generalize!
    maes = []
    for i, prop in enumerate(target_cols):
        mask = ~np.isnan(y_val[:, i])
        if mask.sum() > 0:
            mae = mean_absolute_error(y_val[mask, i], y_pred[mask, i])
            maes.append(mae)
            trial.set_user_attr(f'{prop}_mae', mae)
    
    # Simple average - no complex weighting
    simple_mae = np.mean(maes) if maes else float('inf')
    
    return simple_mae


def main():
    """Run Optuna optimization"""
    print("="*70)
    print("Optuna Hyperparameter Optimization for Random Forest")
    print("Using SIMPLE MAE (what v53 used!)")
    print("="*70)
    print()
    
    # Load data
    X, y, target_cols, n_samples_per_property, ranges_per_property = load_and_prepare_data()
    
    # Create Optuna study
    print("\nStarting Optuna optimization...")
    print("This will take 20-40 minutes for 100 trials")
    print()
    
    # Create NEW study with simple MAE (not complex wMAE)
    study = optuna.create_study(
        direction='minimize',
        study_name='polymer_rf_simple_mae',
        storage='sqlite:///optuna_polymer_rf_simple_mae.db',
        load_if_exists=False  # Fresh start with correct metric
    )
    
    # Run optimization (no longer need n_samples/ranges for simple MAE)
    study.optimize(
        lambda trial: objective(trial, X, y, target_cols, n_samples_per_property, ranges_per_property),
        n_trials=150,  # More trials for better exploration
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"\nBest Simple MAE: {study.best_value:.6f}")
    print("(Using unweighted MAE - matches what v53 optimized!)")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Get top 5 trials
    print("\n" + "="*70)
    print("TOP 5 CONFIGURATIONS FOR KAGGLE TESTING")
    print("="*70)
    
    best_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:5]
    
    configs_for_kaggle = []
    for rank, trial in enumerate(best_trials, 1):
        print(f"\n[Rank {rank}] Trial #{trial.number}")
        print(f"  wMAE: {trial.value:.6f}")
        print(f"  Hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Show per-property MAE
        print(f"  Per-property MAE:")
        for prop in target_cols:
            if f'{prop}_mae' in trial.user_attrs:
                print(f"    {prop}: {trial.user_attrs[f'{prop}_mae']:.4f}")
        
        configs_for_kaggle.append({
            'rank': rank,
            'trial_number': trial.number,
            'wmae': trial.value,
            'params': trial.params,
            'per_property_mae': {prop: trial.user_attrs.get(f'{prop}_mae', None) 
                                for prop in target_cols}
        })
    
    # Save configurations to JSON
    output_file = f'optuna_best_configs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(configs_for_kaggle, f, indent=2)
    
    print(f"\n✅ Saved top 5 configurations to: {output_file}")
    print("\nNext steps:")
    print("1. Update notebook with Rank 1 hyperparameters")
    print("2. Push to Kaggle and submit")
    print("3. Compare Kaggle score with local wMAE")
    print("4. If good, try Rank 2-3 configs")
    
    # Save visualizations if optuna-dashboard is available
    try:
        import optuna.visualization as vis
        
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html('optuna_optimization_history.html')
        
        fig2 = vis.plot_param_importances(study)
        fig2.write_html('optuna_param_importances.html')
        
        print("\n📊 Visualizations saved:")
        print("  - optuna_optimization_history.html")
        print("  - optuna_param_importances.html")
        print("\n  Open these in your browser to see:")
        print("  - How wMAE improved over trials")
        print("  - Which hyperparameters matter most")
    except:
        print("\n💡 Install optuna-dashboard for visualizations:")
        print("  pip install plotly kaleido")


if __name__ == "__main__":
    main()

