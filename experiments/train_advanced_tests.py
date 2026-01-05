"""
Test advanced statistical test features with the current best ensemble.

Tests:
- Cramér-von Mises
- Energy statistic
- Wilcoxon rank-sum
- Mood's test
- Tail behavior (quantiles, kurtosis, extremes)
- Spectral features (FFT)
- Permutation entropy

Compares to baseline 0.8866 AUC (100 features).
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

print("="*70)
print("TESTING ADVANCED STATISTICAL FEATURES")
print("="*70)
print()

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(y)} series, break rate: {y.mean()*100:.2f}%\n")

# Extract base features
print("Extracting base features...")
from sb import features

X_base = features.base.compute_features(
    df,
    use_multiscale=True,
    use_cv=True,
    use_transforms=True,
    use_compression=True,
    use_cusum=True,
    use_boundary_dist=True,
    use_boundary_tail_shape=True
)
print(f"Base features: {X_base.shape}")

from sb.features.statistical_tests import compute_statistical_test_features
X_stats = compute_statistical_test_features(
    df,
    use_anderson=True,
    use_cohens_d=True,
    use_variance_ratios=True,
    use_iqr_ratios=True,
    use_hypothesis_tests=True,
    use_rolling_stats=True,
)
print(f"Statistical features: {X_stats.shape}")

# Import and use advanced tests
sys.path.insert(0, str(Path(__file__).parent / "src" / "sb" / "features"))
import advanced_tests
import importlib
importlib.reload(advanced_tests)

print("Extracting advanced statistical test features...")
X_advanced = advanced_tests.extract_features(df)
print(f"Advanced features: {X_advanced.shape}")

features_list = [X_base, X_stats, X_advanced]
X_all = pd.concat(features_list, axis=1)

# Remove duplicates and handle NaN
X_all = X_all.loc[:, ~X_all.columns.duplicated()]
X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)

print(f"Total features: {X_all.shape}")
print(f"Advanced test features: {[col for col in X_all.columns if col.startswith(('cvm_', 'energy_', 'wilcoxon_', 'mood_', 'tail_', 'spectral_', 'perm_'))][:10]}...\n")

# Select top 100 features
print("Selecting top 100 features...")

train_data = lgb.Dataset(X_all.values, label=y.values)
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
    'feature': X_all.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

top_100_features = importance.head(100)['feature'].tolist()
X = X_all[top_100_features]

# Check how many advanced features made the cut
advanced_in_top100 = [f for f in top_100_features if f.startswith(('cvm_', 'energy_', 'wilcoxon_', 'mood_', 'tail_', 'spectral_', 'perm_'))]
print(f"Advanced features in top 100: {len(advanced_in_top100)}")
if advanced_in_top100:
    print(f"Top advanced features: {advanced_in_top100[:10]}\n")

# Rank transform
X = X.rank(pct=True)

print("="*70)
print("TRAINING 5-MODEL DIVERSE ENSEMBLE WITH CALIBRATION")
print("="*70)
print()

# Model configurations (same as previous best)
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

# Ensemble
all_predictions = []
for model_idx in range(len(model_configs)):
    preds = np.zeros(len(y))
    for val_idx, pred in fold_predictions[model_idx]:
        preds[val_idx] = pred
    all_predictions.append(preds)

ensemble_pred = np.mean(all_predictions, axis=0)
ensemble_score = roc_auc_score(y, ensemble_pred)

print("\n" + "="*70)
print("ENSEMBLE RESULTS")
print("="*70)
print(f"\nEnsemble (avg): {ensemble_score:.4f}")

# Isotonic calibration
print("\nApplying isotonic calibration...")
calibrated_pred = np.zeros(len(y))

# Properly reconstruct out-of-fold predictions for each fold
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Get validation ensemble predictions for this fold
    val_ensemble = np.zeros(len(val_idx))
    for model_idx in range(len(model_configs)):
        val_idx_fold, pred_fold = fold_predictions[model_idx][fold_idx]
        val_ensemble += pred_fold
    val_ensemble /= len(model_configs)
    
    # Get training ensemble predictions (from OTHER folds' validation predictions)
    train_ensemble = np.zeros(len(train_idx))
    train_y = np.zeros(len(train_idx))
    
    # For each training sample, find which fold it was a validation sample in
    for train_pos, train_sample_idx in enumerate(train_idx):
        train_y[train_pos] = y.iloc[train_sample_idx]
        
        # Find which fold this training sample was in validation
        for other_fold_idx in range(5):
            if other_fold_idx == fold_idx:
                continue
            other_train_idx, other_val_idx = list(skf.split(X, y))[other_fold_idx]
            
            # Check if this sample is in the other fold's validation set
            if train_sample_idx in other_val_idx:
                # Get the ensemble prediction for this sample from that fold
                other_val_pos = np.where(other_val_idx == train_sample_idx)[0][0]
                
                sample_pred = 0
                for model_idx in range(len(model_configs)):
                    _, other_pred_fold = fold_predictions[model_idx][other_fold_idx]
                    sample_pred += other_pred_fold[other_val_pos]
                train_ensemble[train_pos] = sample_pred / len(model_configs)
                break
    
    # Fit calibrator on training predictions
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_ensemble, train_y)
    
    # Apply to validation predictions
    calibrated_pred[val_idx] = calibrator.transform(val_ensemble)

calibrated_score = roc_auc_score(y, calibrated_pred)
print(f"Calibrated:     {calibrated_score:.4f}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"\nBaseline (100 features, no advanced tests): 0.8866 AUC")
print(f"With advanced tests (100 features total):   {calibrated_score:.4f} AUC")

improvement = calibrated_score - 0.8866
if improvement > 0:
    print(f"\n✅ Improvement: +{improvement:.4f} AUC")
    if improvement > 0.001:
        print(f"🎉 Significant improvement!")
else:
    print(f"\n❌ No improvement: {improvement:.4f} AUC")

# Show advanced feature importance
print("\n" + "="*70)
print("ADVANCED FEATURE IMPORTANCE")
print("="*70)

advanced_importance = importance[importance['feature'].str.startswith(('cvm_', 'energy_', 'wilcoxon_', 'mood_', 'tail_', 'spectral_', 'perm_'))].sort_values('importance', ascending=False)
if len(advanced_importance) > 0:
    print("\nTop 20 advanced features:")
    print(advanced_importance.head(20).to_string(index=False))
else:
    print("\nNo advanced features in top importance ranks")

# Show overall top 20 features
print("\n" + "="*70)
print("TOP 20 FEATURES OVERALL")
print("="*70)
print()
print(importance.head(20).to_string(index=False))
