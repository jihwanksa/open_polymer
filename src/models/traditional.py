"""
Traditional ML models: XGBoost, Random Forest
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
from tqdm import tqdm


class TraditionalMLModel:
    """Base class for traditional ML models"""
    
    def __init__(self, model_type='xgboost', n_targets=5):
        self.model_type = model_type
        self.n_targets = n_targets
        self.models = []
        self.feature_importances = []
        
    def create_model(self, target_idx):
        """Create a model instance"""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + target_idx,
                n_jobs=-1,
                tree_method='hist'
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42 + target_idx,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_single_target(self, X_train, y_train, X_val, y_val, target_name, target_idx):
        """Train model for a single target"""
        # Filter out NaN values (missing data)
        train_mask = ~np.isnan(y_train)
        val_mask = ~np.isnan(y_val)
        
        if train_mask.sum() < 10:
            print(f"  Skipping {target_name} - insufficient training data")
            return None, None, None
        
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train[train_mask]
        
        # Train model
        model = self.create_model(target_idx)
        model.fit(X_train_filtered, y_train_filtered)
        
        # Evaluate
        metrics = {}
        if val_mask.sum() > 0:
            X_val_filtered = X_val[val_mask]
            y_val_filtered = y_val[val_mask]
            
            y_pred = model.predict(X_val_filtered)
            metrics['mse'] = mean_squared_error(y_val_filtered, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_val_filtered, y_pred)
            metrics['r2'] = r2_score(y_val_filtered, y_pred)
            metrics['n_samples'] = len(y_val_filtered)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        else:
            feature_importance = None
        
        return model, metrics, feature_importance
    
    def train(self, X_train, y_train, X_val, y_val, target_names):
        """Train models for all targets"""
        print(f"\nTraining {self.model_type.upper()} models...")
        print("=" * 80)
        
        self.models = []
        self.feature_importances = []
        all_metrics = {}
        
        for i, target_name in enumerate(target_names):
            print(f"\nTarget: {target_name}")
            print("-" * 40)
            
            y_train_target = y_train[:, i]
            y_val_target = y_val[:, i]
            
            model, metrics, feat_imp = self.train_single_target(
                X_train, y_train_target, X_val, y_val_target, 
                target_name, i
            )
            
            self.models.append(model)
            self.feature_importances.append(feat_imp)
            
            if metrics:
                all_metrics[target_name] = metrics
                print(f"  Training samples: {(y_train_target != 0).sum()}")
                print(f"  Validation samples: {metrics['n_samples']}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  R²: {metrics['r2']:.4f}")
        
        print("\n" + "=" * 80)
        print("Training Summary:")
        print("-" * 40)
        
        for target_name, metrics in all_metrics.items():
            print(f"{target_name:12s} | RMSE: {metrics['rmse']:8.4f} | MAE: {metrics['mae']:8.4f} | R²: {metrics['r2']:6.4f}")
        
        return all_metrics
    
    def train_with_cv(self, X, y, target_names, n_folds=5):
        """Train with cross-validation"""
        print(f"\nTraining {self.model_type.upper()} with {n_folds}-fold CV...")
        print("=" * 80)
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = {target: [] for target in target_names}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/{n_folds}")
            print("-" * 40)
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for i, target_name in enumerate(target_names):
                y_train_target = y_train[:, i]
                y_val_target = y_val[:, i]
                
                model, metrics, _ = self.train_single_target(
                    X_train, y_train_target, X_val, y_val_target,
                    target_name, i
                )
                
                if metrics:
                    cv_results[target_name].append(metrics)
        
        # Aggregate results
        print("\n" + "=" * 80)
        print("Cross-Validation Summary:")
        print("-" * 40)
        
        aggregated_results = {}
        for target_name in target_names:
            if cv_results[target_name]:
                results = cv_results[target_name]
                aggregated_results[target_name] = {
                    'rmse_mean': np.mean([r['rmse'] for r in results]),
                    'rmse_std': np.std([r['rmse'] for r in results]),
                    'mae_mean': np.mean([r['mae'] for r in results]),
                    'mae_std': np.std([r['mae'] for r in results]),
                    'r2_mean': np.mean([r['r2'] for r in results]),
                    'r2_std': np.std([r['r2'] for r in results]),
                }
                
                print(f"{target_name:12s} | RMSE: {aggregated_results[target_name]['rmse_mean']:.4f} ± {aggregated_results[target_name]['rmse_std']:.4f} | "
                      f"MAE: {aggregated_results[target_name]['mae_mean']:.4f} ± {aggregated_results[target_name]['mae_std']:.4f} | "
                      f"R²: {aggregated_results[target_name]['r2_mean']:.4f} ± {aggregated_results[target_name]['r2_std']:.4f}")
        
        return aggregated_results
    
    def predict(self, X):
        """Predict all targets"""
        predictions = []
        for model in self.models:
            if model is not None:
                pred = model.predict(X)
            else:
                pred = np.zeros(len(X))
            predictions.append(pred)
        
        return np.stack(predictions, axis=1)
    
    def save(self, path):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'model_type': self.model_type,
                'feature_importances': self.feature_importances
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.model_type = data['model_type']
            self.feature_importances = data.get('feature_importances', [])
        print(f"Model loaded from {path}")

