"""
Aggressive feature selection to reduce noise and improve signal-to-noise ratio.

Strategy:
1. Start with Phase 1 features (372 proven features at 0.87 AUC)
2. Use multiple selection criteria:
   - Mutual information (relevance)
   - Stability across CV folds (consistency)
   - Low redundancy (correlation filtering)
3. Select best 100-150 features (matching winning solutions)

Expected: 0.87 → 0.88-0.89 AUC from better signal-to-noise ratio
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
print("AGGRESSIVE FEATURE SELECTION")
print("="*70)
print()

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(df['id'].unique())} series, break rate: {y.mean()*100:.2f}%\n")

# Extract Phase 1 features (no boundary features - too many)
print("Extracting Phase 1 features (CV + transforms + compression + CUSUM)...")
X_phase1 = features.base.compute_features(
    df,
    use_multiscale=True,
    use_cv=True,
    use_transforms=True,
    use_compression=True,
    use_cusum=True
)
print(f"Phase 1 features: {X_phase1.shape}")

# Impute NaN
print("Imputing NaN values...")
X_phase1 = X_phase1.fillna(X_phase1.median())
print(f"Feature shape after imputation: {X_phase1.shape}\n")

# Step 1: Mutual Information
print("="*70)
print("STEP 1: MUTUAL INFORMATION RANKING")
print("="*70 + "\n")

print("Computing mutual information scores...")
mi_scores = mutual_info_classif(X_phase1, y, random_state=42)
mi_ranking = pd.DataFrame({
    'feature': X_phase1.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("Top 20 features by MI:")
for idx, row in mi_ranking.head(20).iterrows():
    print(f"  {row['feature']:50s}: {row['mi_score']:.4f}")

# Step 2: Stability across folds (which features consistently appear in top importances?)
print("\n" + "="*70)
print("STEP 2: STABILITY ACROSS CV FOLDS")
print("="*70 + "\n")

print("Training LightGBM on each fold to measure feature stability...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_importances = defaultdict(list)
fold_aucs = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_phase1, y), 1):
    X_train = X_phase1.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X_phase1.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    # Rank normalize
    X_train_rank = X_train.rank(pct=True)
    X_val_rank = X_val.rank(pct=True)
    
    # Train
    train_data = lgb.Dataset(X_train_rank, label=y_train)
    val_data = lgb.Dataset(X_val_rank, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'n_estimators': 300,
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
    
    # Get feature importances
    importance = model.feature_importance(importance_type='gain')
    for feat, imp in zip(X_phase1.columns, importance):
        fold_importances[feat].append(imp)
    
    # Evaluate
    y_pred = model.predict(X_val_rank)
    auc = roc_auc_score(y_val, y_pred)
    fold_aucs.append(auc)
    print(f"  Fold {fold_idx}/5: AUC = {auc:.4f}")

print(f"\nBaseline CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# Calculate stability metrics
stability_scores = []
for feat in X_phase1.columns:
    importances = fold_importances[feat]
    # Stability = mean importance with penalty for high variance
    mean_imp = np.mean(importances)
    std_imp = np.std(importances)
    # Coefficient of variation as stability metric (lower is more stable)
    cv = std_imp / (mean_imp + 1e-8)
    # Stability score: high mean, low CV
    stability = mean_imp / (1 + cv)
    stability_scores.append({
        'feature': feat,
        'mean_importance': mean_imp,
        'std_importance': std_imp,
        'cv': cv,
        'stability_score': stability
    })

stability_df = pd.DataFrame(stability_scores).sort_values('stability_score', ascending=False)

print("\nTop 20 most stable features:")
for idx, row in stability_df.head(20).iterrows():
    print(f"  {row['feature']:50s}: stability={row['stability_score']:8.1f} (mean={row['mean_importance']:6.1f}, cv={row['cv']:.2f})")

# Step 3: Combine rankings
print("\n" + "="*70)
print("STEP 3: COMBINED RANKING")
print("="*70 + "\n")

# Normalize scores to 0-1 range
mi_ranking['mi_score_norm'] = (mi_ranking['mi_score'] - mi_ranking['mi_score'].min()) / (mi_ranking['mi_score'].max() - mi_ranking['mi_score'].min())
stability_df['stability_score_norm'] = (stability_df['stability_score'] - stability_df['stability_score'].min()) / (stability_df['stability_score'].max() - stability_df['stability_score'].min())

# Merge and create combined score (60% stability, 40% MI)
combined = mi_ranking.merge(stability_df[['feature', 'stability_score_norm', 'mean_importance']], on='feature')
combined['combined_score'] = 0.6 * combined['stability_score_norm'] + 0.4 * combined['mi_score_norm']
combined = combined.sort_values('combined_score', ascending=False)

print("Top 30 features by combined score (60% stability + 40% MI):")
for idx, row in combined.head(30).iterrows():
    print(f"  {row['feature']:50s}: combined={row['combined_score']:.4f} (MI={row['mi_score']:.4f}, imp={row['mean_importance']:.1f})")

# Step 4: Correlation filtering on top features
print("\n" + "="*70)
print("STEP 4: CORRELATION FILTERING")
print("="*70 + "\n")

# Start with top 200 by combined score
top_200 = combined.head(200)['feature'].tolist()
X_top200 = X_phase1[top_200]

print(f"Starting with top 200 features by combined score")
print("Removing highly correlated features (threshold: 0.90)...")

# Greedy correlation filtering: keep feature with higher combined score
selected_features = []
combined_dict = dict(zip(combined['feature'], combined['combined_score']))

for feat in top_200:
    # Check correlation with already selected features
    is_redundant = False
    if len(selected_features) > 0:
        corr_with_selected = X_phase1[[feat]].corrwith(X_phase1[selected_features], axis=0).abs().max()
        if corr_with_selected > 0.90:
            is_redundant = True
    
    if not is_redundant:
        selected_features.append(feat)
    
    # Stop at 150 features
    if len(selected_features) >= 150:
        break

print(f"Selected {len(selected_features)} features after correlation filtering")

# Step 5: Evaluate selected features
print("\n" + "="*70)
print("STEP 5: EVALUATION WITH SELECTED FEATURES")
print("="*70 + "\n")

X_selected = X_phase1[selected_features]
print(f"Feature count: {len(selected_features)}\n")

# Cross-validation with selected features
fold_aucs_selected = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_selected, y), 1):
    X_train = X_selected.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X_selected.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    # Rank normalize
    X_train_rank = X_train.rank(pct=True)
    X_val_rank = X_val.rank(pct=True)
    
    # Train
    train_data = lgb.Dataset(X_train_rank, label=y_train)
    val_data = lgb.Dataset(X_val_rank, label=y_val, reference=train_data)
    
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
    
    y_pred = model.predict(X_val_rank)
    auc = roc_auc_score(y_val, y_pred)
    fold_aucs_selected.append(auc)
    print(f"Fold {fold_idx}/5: AUC = {auc:.4f}")

mean_auc_selected = np.mean(fold_aucs_selected)
std_auc_selected = np.std(fold_aucs_selected)

print("\n" + "="*70)
print("RESULTS")
print("="*70 + "\n")

print(f"Baseline (372 features):  {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"Selected ({len(selected_features)} features): {mean_auc_selected:.4f} ± {std_auc_selected:.4f}")

improvement = mean_auc_selected - np.mean(fold_aucs)
print(f"\nImprovement: {improvement:+.4f}")

if improvement > 0.005:
    print("\n✅ Feature selection significantly improved performance!")
elif improvement > 0:
    print("\n✅ Slight improvement from feature selection")
else:
    print("\n⚠️  Feature selection didn't help. Baseline is better.")

# Train final model if improvement
if improvement >= 0:
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL WITH SELECTED FEATURES")
    print("="*70 + "\n")
    
    X_selected_rank = X_selected.rank(pct=True)
    train_data = lgb.Dataset(X_selected_rank, label=y)
    
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
        'feature': X_selected.columns,
        'importance': final_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 30 feature importances:")
    for idx, row in feature_importance.head(30).iterrows():
        print(f"  {row['feature']:50s}: {row['importance']:.1f}")
    
    # Save model and feature list
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    joblib.dump(final_model, model_path / "model_selected_features.joblib")
    
    # Save selected features for future use
    with open(model_path / "selected_features.txt", 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    
    print(f"\n✅ Model saved to: {model_path / 'model_selected_features.joblib'}")
    print(f"✅ Feature list saved to: {model_path / 'selected_features.txt'}")

print("\n" + "="*70)
print(f"FINAL CV AUC: {mean_auc_selected:.4f} ± {std_auc_selected:.4f}")
print(f"Features: {len(selected_features)} (reduced from 372)")
print("="*70)
