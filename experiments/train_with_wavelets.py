"""
Test wavelet features with the current best ensemble.

Previous best: 0.8966 AUC (uncalibrated ensemble with advanced tests)
Testing: Adding wavelet decomposition features
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
print("TESTING WAVELET FEATURES")
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

# Import advanced tests
sys.path.insert(0, str(Path(__file__).parent / "src" / "sb" / "features"))
import advanced_tests
import wavelet_features
import importlib
importlib.reload(advanced_tests)
importlib.reload(wavelet_features)

print("Extracting advanced statistical test features...")
X_advanced = advanced_tests.extract_features(df)
print(f"Advanced features: {X_advanced.shape}")

print("Extracting wavelet features...")
X_wavelet = wavelet_features.extract_features(df, wavelet='db4', level=3)
print(f"Wavelet features: {X_wavelet.shape}")

features_list = [X_base, X_stats, X_advanced, X_wavelet]
X_all = pd.concat(features_list, axis=1)

# Remove duplicates and handle NaN
X_all = X_all.loc[:, ~X_all.columns.duplicated()]
X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)

print(f"Total features: {X_all.shape}")
print(f"Wavelet feature names: {[col for col in X_all.columns if col.startswith('wavelet_')][:10]}...\n")

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

# Check how many wavelet features made the cut
wavelet_in_top100 = [f for f in top_100_features if f.startswith('wavelet_')]
print(f"Wavelet features in top 100: {len(wavelet_in_top100)}")
if wavelet_in_top100:
    print(f"Top wavelet features: {wavelet_in_top100[:10]}\n")

# Rank transform
X = X.rank(pct=True)

print("="*70)
print("TRAINING 5-MODEL DIVERSE ENSEMBLE")
print("="*70)
print()

# Model configurations
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

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"\nPrevious best (advanced tests only): 0.8966 AUC")
print(f"With wavelets (100 features total):  {ensemble_score:.4f} AUC")

improvement = ensemble_score - 0.8966
if improvement > 0:
    print(f"\n✅ Improvement: +{improvement:.4f} AUC")
    if improvement > 0.001:
        print(f"🎉 Significant improvement!")
else:
    print(f"\n❌ No improvement: {improvement:.4f} AUC")

# Show wavelet feature importance
print("\n" + "="*70)
print("WAVELET FEATURE IMPORTANCE")
print("="*70)

wavelet_importance = importance[importance['feature'].str.startswith('wavelet_')].sort_values('importance', ascending=False)
if len(wavelet_importance) > 0:
    print(f"\nTop 20 wavelet features:")
    print(wavelet_importance.head(20).to_string(index=False))
else:
    print("\nNo wavelet features in importance ranking")

# Show overall top 20 features
print("\n" + "="*70)
print("TOP 20 FEATURES OVERALL")
print("="*70)
print()
print(importance.head(20).to_string(index=False))
