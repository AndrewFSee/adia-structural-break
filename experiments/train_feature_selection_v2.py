"""
Feature selection V2: Include boundary features (they were helping!)

Previous results:
- Phase 1 only (372 features): 0.8588 AUC
- Phase 1 + boundary (920 features): 0.8720 AUC
- Selected without boundary (150 features): 0.8614 AUC

Strategy: Start with all 920 features, aggressively select best 150-200.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
import joblib
from collections import defaultdict

from sb import data_loader, features, config

print("="*70)
print("AGGRESSIVE FEATURE SELECTION V2 (WITH BOUNDARY FEATURES)")
print("="*70)
print()

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(df['id'].unique())} series, break rate: {y.mean()*100:.2f}%\n")

# Extract Phase 1 features + boundary features
print("Extracting Phase 1 + boundary features...")
X_all = features.base.compute_features(
    df,
    use_multiscale=True,
    use_cv=True,
    use_transforms=True,
    use_compression=True,
    use_cusum=True,
    use_boundary_dist=True,
    use_boundary_tail_shape=True
)
print(f"All features: {X_all.shape}")

# Impute NaN
print("Imputing NaN values...")
X_all = X_all.fillna(X_all.median())
print(f"Feature shape after imputation: {X_all.shape}\n")

# Quick baseline check
print("="*70)
print("BASELINE CHECK")
print("="*70 + "\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_aucs = []

print("Running 5-fold CV with all features (this establishes our baseline)...")
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

# Aggressive feature selection
print("\n" + "="*70)
print("FEATURE SELECTION")
print("="*70 + "\n")

print("Step 1: Computing mutual information scores...")
mi_scores = mutual_info_classif(X_all, y, random_state=42)
mi_ranking = pd.DataFrame({
    'feature': X_all.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop 20 features by MI:")
for idx, row in mi_ranking.head(20).iterrows():
    print(f"  {row['feature']:50s}: {row['mi_score']:.4f}")

# Select top 200 by MI, then correlation filter to 150
print("\nStep 2: Selecting top 200 by MI, then correlation filter to 150...")
top_200 = mi_ranking.head(200)['feature'].tolist()
X_top200 = X_all[top_200]

# Correlation filtering
selected_features = []
mi_dict = dict(zip(mi_ranking['feature'], mi_ranking['mi_score']))

for feat in top_200:
    is_redundant = False
    if len(selected_features) > 0:
        corr_with_selected = X_all[[feat]].corrwith(X_all[selected_features], axis=0).abs().max()
        if corr_with_selected > 0.90:
            is_redundant = True
    
    if not is_redundant:
        selected_features.append(feat)
    
    if len(selected_features) >= 150:
        break

print(f"Selected {len(selected_features)} features after correlation filtering")

# Evaluate
print("\n" + "="*70)
print("EVALUATION WITH SELECTED FEATURES")
print("="*70 + "\n")

X_selected = X_all[selected_features]
print(f"Feature count: {len(selected_features)}\n")

fold_aucs_selected = []

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
    fold_aucs_selected.append(auc)
    print(f"Fold {fold_idx}/5: AUC = {auc:.4f}")

mean_auc = np.mean(fold_aucs_selected)
std_auc = np.std(fold_aucs_selected)

print("\n" + "="*70)
print("RESULTS")
print("="*70 + "\n")

print(f"Baseline ({X_all.shape[1]} features): {baseline_mean:.4f} ± {baseline_std:.4f}")
print(f"Selected ({len(selected_features)} features): {mean_auc:.4f} ± {std_auc:.4f}")

improvement = mean_auc - baseline_mean
print(f"\nImprovement: {improvement:+.4f}")

if improvement >= 0:
    status = "✅ Feature selection helped!" if improvement > 0.005 else "✅ Slight improvement"
else:
    status = "⚠️  Baseline is better - use all features"

print(f"\n{status}")

# Train final model
print("\n" + "="*70)
print("TRAINING FINAL MODEL")
print("="*70 + "\n")

X_final = X_selected if improvement >= 0 else X_all
X_final_rank = X_final.rank(pct=True)

train_data = lgb.Dataset(X_final_rank, label=y)

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
    'feature': X_final.columns,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nTop 30 feature importances:")
for idx, row in feature_importance.head(30).iterrows():
    print(f"  {row['feature']:50s}: {row['importance']:.1f}")

# Count feature types in top 30
top_30 = feature_importance.head(30)
cv_count = sum(1 for f in top_30['feature'] if 'cv_' in f)
boundary_count = sum(1 for f in top_30['feature'] if 'bl_' in f)
compression_count = sum(1 for f in top_30['feature'] if 'ncd_' in f or 'zlib_' in f or 'lz_' in f)
cusum_count = sum(1 for f in top_30['feature'] if 'cusum_' in f)

print(f"\nTop 30 feature breakdown:")
print(f"  CV features: {cv_count}")
print(f"  Boundary features: {boundary_count}")
print(f"  Compression features: {compression_count}")
print(f"  CUSUM features: {cusum_count}")
print(f"  Other: {30 - cv_count - boundary_count - compression_count - cusum_count}")

# Save
model_path = Path("models")
model_path.mkdir(exist_ok=True)
joblib.dump(final_model, model_path / "model_v2.joblib")

if improvement >= 0:
    with open(model_path / "selected_features_v2.txt", 'w') as f:
        for feat in X_final.columns:
            f.write(f"{feat}\n")
    print(f"\n✅ Model saved to: {model_path / 'model_v2.joblib'}")
    print(f"✅ Feature list saved to: {model_path / 'selected_features_v2.txt'}")
else:
    print(f"\n✅ Model saved to: {model_path / 'model_v2.joblib'} (using all {X_all.shape[1]} features)")

print("\n" + "="*70)
final_auc = mean_auc if improvement >= 0 else baseline_mean
final_std = std_auc if improvement >= 0 else baseline_std
final_count = len(X_final.columns)
print(f"FINAL CV AUC: {final_auc:.4f} ± {final_std:.4f}")
print(f"Features: {final_count}")
print("="*70)
