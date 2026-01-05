"""
Training with RFE (Recursive Feature Elimination) - like 10th place winner.

10th place used RFE to select only 33 features and achieved 85.86 AUC.
This script:
1. Adds new statistical test features
2. Uses RFE to select best 33-150 features (model-aware selection)
3. Trains final model on selected features

Expected: Better than MI-based selection (which made performance worse).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
import lightgbm as lgb
import joblib

from sb import data_loader, features, config

print("="*70)
print("RFE-BASED FEATURE SELECTION (10TH PLACE APPROACH)")
print("="*70)
print()

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(df['id'].unique())} series, break rate: {y.mean()*100:.2f}%\n")

# Extract Phase 1 features
print("Extracting Phase 1 features...")
X_phase1 = features.base.compute_features(
    df,
    use_multiscale=True,
    use_cv=True,
    use_transforms=True,
    use_compression=True,
    use_cusum=True,
    use_boundary_dist=True,
    use_boundary_tail_shape=True
)
print(f"Phase 1 features: {X_phase1.shape}")

# Add NEW statistical test features
print("\nAdding statistical test features (from winners)...")
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
print(f"Statistical test features: {X_stats.shape}")

# Combine all features
X_all = pd.concat([X_phase1, X_stats], axis=1)
print(f"Total features: {X_all.shape}")

# Remove duplicate column names (keep first occurrence)
duplicate_cols = X_all.columns[X_all.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"Removing {len(duplicate_cols)} duplicate columns: {duplicate_cols[:10]}...")
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    print(f"After deduplication: {X_all.shape}")

# Impute NaN
print("\nImputing NaN values...")
X_all = X_all.fillna(X_all.median())

# Baseline with all features
print("\n" + "="*70)
print("BASELINE WITH ALL FEATURES")
print("="*70 + "\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_aucs = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y), 1):
    X_train = X_all.iloc[train_idx].rank(pct=True)
    y_train = y.iloc[train_idx]
    X_val = X_all.iloc[val_idx].rank(pct=True)
    y_val = y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 5,
        'min_data_in_leaf': 100,
        'lambda_l2': 0.5,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    baseline_aucs.append(auc)
    print(f"  Fold {fold_idx}/5: AUC = {auc:.4f}")

baseline_mean = np.mean(baseline_aucs)
baseline_std = np.std(baseline_aucs)
print(f"\nBaseline ({X_all.shape[1]} features): {baseline_mean:.4f} ± {baseline_std:.4f}")

# RFE Feature Selection (10th place approach)
print("\n" + "="*70)
print("RFE FEATURE SELECTION")
print("="*70 + "\n")

# Try multiple feature counts
n_features_to_try = [33, 50, 100, 150, 200]

results = []

for n_features in n_features_to_try:
    print(f"\n--- Testing with {n_features} features ---")
    
    # RFE selector
    base_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        random_state=42,
        verbosity=-1
    )
    
    print(f"Running RFE (step=10, target={n_features} features)...")
    selector = RFE(
        base_model,
        n_features_to_select=n_features,
        step=10,
        verbose=0
    )
    
    # Fit RFE on all data (rank normalized)
    X_all_rank = X_all.rank(pct=True)
    selector.fit(X_all_rank, y)
    
    # Get selected features
    selected_features = X_all.columns[selector.support_].tolist()
    print(f"Selected {len(selected_features)} features")
    
    # Evaluate with CV
    X_selected = X_all[selected_features]
    fold_aucs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_selected, y), 1):
        X_train = X_selected.iloc[train_idx].rank(pct=True)
        y_train = y.iloc[train_idx]
        X_val = X_selected.iloc[val_idx].rank(pct=True)
        y_val = y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42,
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'min_data_in_leaf': 100,
            'lambda_l2': 0.5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        fold_aucs.append(auc)
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    print(f"CV AUC with {n_features} features: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Folds: {[f'{auc:.4f}' for auc in fold_aucs]}")
    
    results.append({
        'n_features': n_features,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'selected_features': selected_features,
        'selector': selector
    })

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70 + "\n")

print(f"Baseline ({X_all.shape[1]} features): {baseline_mean:.4f} ± {baseline_std:.4f}")
print()

for r in results:
    improvement = r['mean_auc'] - baseline_mean
    print(f"{r['n_features']:3d} features: {r['mean_auc']:.4f} ± {r['std_auc']:.4f}  "
          f"[{improvement:+.4f}]")

# Select best configuration
best_result = max(results, key=lambda x: x['mean_auc'])
print(f"\n✅ Best: {best_result['n_features']} features with {best_result['mean_auc']:.4f} AUC")

# Train final model with best configuration
print("\n" + "="*70)
print("TRAINING FINAL MODEL")
print("="*70 + "\n")

X_best = X_all[best_result['selected_features']]
X_best_rank = X_best.rank(pct=True)

train_data = lgb.Dataset(X_best_rank, label=y)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': 42,
    'n_estimators': 200,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 5,
    'min_data_in_leaf': 100,
    'lambda_l2': 0.5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
}

final_model = lgb.train(params, train_data, callbacks=[lgb.log_evaluation(0)])

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_best.columns,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(f"\nTop {min(30, len(feature_importance))} feature importances:")
for idx, row in feature_importance.head(30).iterrows():
    print(f"  {row['feature']:50s}: {row['importance']:.1f}")

# Check feature types
top_30 = feature_importance.head(30)
stat_count = sum(1 for f in top_30['feature'] if any(x in f for x in ['anderson', 'cohens', 'vard_ratio', 'iqr_ratio', 'f_test', 'levene', 'ks_', 'rolling']))
cv_count = sum(1 for f in top_30['feature'] if 'cv_' in f)
boundary_count = sum(1 for f in top_30['feature'] if 'bl_' in f)
compression_count = sum(1 for f in top_30['feature'] if any(x in f for x in ['ncd_', 'zlib_', 'lz_']))

print(f"\nTop 30 feature breakdown:")
print(f"  NEW Statistical tests: {stat_count}")
print(f"  CV features: {cv_count}")
print(f"  Boundary features: {boundary_count}")
print(f"  Compression features: {compression_count}")
print(f"  Other: {30 - stat_count - cv_count - boundary_count - compression_count}")

# Save
model_path = Path("models")
model_path.mkdir(exist_ok=True)
joblib.dump(final_model, model_path / "model_rfe.joblib")
joblib.dump(best_result['selector'], model_path / "rfe_selector.joblib")

with open(model_path / "selected_features_rfe.txt", 'w') as f:
    for feat in best_result['selected_features']:
        f.write(f"{feat}\n")

print(f"\n✅ Model saved to: {model_path / 'model_rfe.joblib'}")
print(f"✅ RFE selector saved to: {model_path / 'rfe_selector.joblib'}")
print(f"✅ Feature list saved to: {model_path / 'selected_features_rfe.txt'}")

print("\n" + "="*70)
print(f"FINAL CV AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
print(f"Features: {best_result['n_features']} (reduced from {X_all.shape[1]})")
print(f"Improvement over baseline: {best_result['mean_auc'] - baseline_mean:+.4f}")
print("="*70)
