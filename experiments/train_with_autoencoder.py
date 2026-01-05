"""
Test autoencoder features with the current best ensemble.

Approach:
1. Extract top 100 features (baseline)
2. Add autoencoder reconstruction error features
3. Test with 5-model diverse ensemble + calibration
4. Compare to baseline 0.8866 AUC
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

from sb import data_loader
from sb.features import autoencoder

print("="*70)
print("TESTING AUTOENCODER FEATURES WITH ENSEMBLE")
print("="*70)
print()

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(y)} series, break rate: {y.mean()*100:.2f}%\n")

# Extract base features
print("Extracting base features...")
import importlib
from sb.features import (
    coefficient_of_variation,
    transform_features,
    compression_features,
    cusum_features,
    statistical_tests,
)

# Refresh modules to get latest code
for module in [coefficient_of_variation, transform_features, compression_features, cusum_features, statistical_tests]:
    importlib.reload(module)

features = []
features.append(coefficient_of_variation.extract_features(df))
features.append(transform_features.extract_features(df))
features.append(compression_features.extract_features(df))
features.append(cusum_features.extract_features(df))
features.append(statistical_tests.extract_features(df))

X_base = pd.concat(features, axis=1)
print(f"Base features: {X_base.shape}")

# Extract autoencoder features
print("\nExtracting autoencoder features (this may take a while)...")
X_ae = autoencoder.compute_autoencoder_features(df, window_size=20, stride=5, epochs=30, verbose=True)
print(f"Autoencoder features: {X_ae.shape}")

# Combine features
X_combined = pd.concat([X_base, X_ae], axis=1)

# Remove duplicates and handle NaN
X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
X_combined = X_combined.fillna(0).replace([np.inf, -np.inf], 0)

print(f"Combined features: {X_combined.shape}")
print(f"Autoencoder features added: {X_ae.shape[1]}")
print(f"Autoencoder feature names: {list(X_ae.columns)}\n")

# Select top 100 + all autoencoder features
print("Selecting features...")

# Train a quick model to get base feature importance
X_train = X_combined.values
y_train = y.values

train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1,
}

model = lgb.train(params, train_data, num_boost_round=100)
importance = pd.DataFrame({
    'feature': X_combined.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

# Keep top 100 non-autoencoder features + all autoencoder features
ae_cols = list(X_ae.columns)
non_ae_cols = [col for col in X_combined.columns if col not in ae_cols]

top_100_non_ae = importance[importance['feature'].isin(non_ae_cols)].head(100)['feature'].tolist()
selected_features = top_100_non_ae + ae_cols

X_selected = X_combined[selected_features]
print(f"Selected {len(selected_features)} features (100 base + {len(ae_cols)} autoencoder)")
print(f"Autoencoder features: {ae_cols}\n")

# Rank transform (winners used this)
X = X_selected.rank(pct=True)

print("="*70)
print("TRAINING 5-MODEL DIVERSE ENSEMBLE WITH CALIBRATION")
print("="*70)
print()

# Model configurations (same as train_final_push.py)
model_configs = [
    {'name': 'Model 1 (baseline)', 'seed': 42, 'num_leaves': 63, 'learning_rate': 0.03, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
    {'name': 'Model 2 (deep+reg)', 'seed': 123, 'num_leaves': 127, 'learning_rate': 0.03, 'reg_alpha': 0.3, 'reg_lambda': 0.3},
    {'name': 'Model 3 (shallow)', 'seed': 456, 'num_leaves': 31, 'learning_rate': 0.05, 'reg_alpha': 0.0, 'reg_lambda': 0.0},
    {'name': 'Model 4 (high sampling)', 'seed': 789, 'num_leaves': 63, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 0.9},
    {'name': 'Model 5 (low LR)', 'seed': 999, 'num_leaves': 63, 'learning_rate': 0.01, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
]

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_predictions = [[] for _ in range(len(model_configs))]
fold_scores = [[] for _ in range(len(model_configs))]

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold_idx + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    for model_idx, config in enumerate(model_configs):
        base_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting': 'gbdt',
            'verbosity': -1,
            'seed': config['seed'],
        }
        
        # Add config-specific params
        for key in ['num_leaves', 'learning_rate', 'reg_alpha', 'reg_lambda', 'subsample', 'colsample_bytree']:
            if key in config:
                base_params[key] = config[key]
        
        train_data = lgb.Dataset(X_train.values, label=y_train.values)
        val_data = lgb.Dataset(X_val.values, label=y_val.values, reference=train_data)
        
        model = lgb.train(
            base_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)
            ]
        )
        
        pred = model.predict(X_val.values)
        score = roc_auc_score(y_val, pred)
        
        fold_predictions[model_idx].append((val_idx, pred))
        fold_scores[model_idx].append(score)

print("\n" + "="*70)
print("INDIVIDUAL MODEL RESULTS")
print("="*70)

for model_idx, config in enumerate(model_configs):
    scores = fold_scores[model_idx]
    print(f"{config['name']}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Combine predictions (all folds)
print("\n" + "="*70)
print("ENSEMBLE RESULTS")
print("="*70)

all_predictions = []
for model_idx in range(len(model_configs)):
    preds = np.zeros(len(y))
    for val_idx, pred in fold_predictions[model_idx]:
        preds[val_idx] = pred
    all_predictions.append(preds)

# Simple average ensemble
ensemble_pred = np.mean(all_predictions, axis=0)
ensemble_score = roc_auc_score(y, ensemble_pred)
print(f"\nEnsemble (avg): {ensemble_score:.4f}")

# Isotonic calibration
print("\nApplying isotonic calibration...")
calibrated_pred = np.zeros(len(y))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Get ensemble predictions for this fold
    fold_ensemble = np.zeros(len(val_idx))
    for model_idx in range(len(model_configs)):
        val_idx_fold, pred_fold = fold_predictions[model_idx][fold_idx]
        fold_ensemble += pred_fold
    fold_ensemble /= len(model_configs)
    
    # Fit calibrator on training folds
    train_ensemble = np.zeros(len(train_idx))
    for other_fold_idx in range(5):
        if other_fold_idx == fold_idx:
            continue
        for model_idx in range(len(model_configs)):
            other_val_idx, other_pred = fold_predictions[model_idx][other_fold_idx]
            # Map to train_idx positions
            mask = np.isin(other_val_idx, train_idx)
            if mask.any():
                train_positions = np.searchsorted(train_idx, other_val_idx[mask])
                train_ensemble[train_positions] += other_pred[mask]
    
    # Average over folds that contributed
    train_ensemble /= 4.0 * len(model_configs)
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_ensemble, y.iloc[train_idx].values)
    
    calibrated_pred[val_idx] = calibrator.transform(fold_ensemble)

calibrated_score = roc_auc_score(y, calibrated_pred)
print(f"Calibrated:     {calibrated_score:.4f}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"\nBaseline (100 features, no autoencoder): 0.8866 AUC")
print(f"With autoencoder ({len(selected_features)} features):    {calibrated_score:.4f} AUC")

improvement = calibrated_score - 0.8866
if improvement > 0:
    print(f"\n✅ Improvement: +{improvement:.4f} AUC")
else:
    print(f"\n❌ No improvement: {improvement:.4f} AUC")

# Show autoencoder feature importance
print("\n" + "="*70)
print("AUTOENCODER FEATURE IMPORTANCE")
print("="*70)

final_importance = importance[importance['feature'].isin(ae_cols)].sort_values('importance', ascending=False)
if len(final_importance) > 0:
    print("\n", final_importance.to_string(index=False))
else:
    print("\nNo autoencoder features in importance ranking (may need more data)")
