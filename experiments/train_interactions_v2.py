"""
Training script with feature interactions (Phase 2) - CORRECTED VERSION.

Key fix: KEEP all Phase 1 features + ADD selected interactions (not replace).
This preserves the magic features (cv_std_interaction, ncd, etc.) while adding interactions.

Expected: 0.87 → 0.88-0.89 AUC
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

from sb import data_loader, features, config

print("="*70)
print("PHASE 2: FEATURE INTERACTIONS (CORRECTED)")
print("="*70)
print()

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(df['id'].unique())} series, break rate: {y.mean()*100:.2f}%\n")

# Extract Phase 1 features (these WORK - 0.87 AUC!)
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

# Add boundary features
print("\nAdding boundary features for differentiation...")
X_boundary = features.base.compute_features(
    df,
    use_multiscale=True,
    use_boundary_dist=True,
    use_boundary_tail_shape=True
)
boundary_cols = [c for c in X_boundary.columns if c not in X_phase1.columns]
X_phase1 = pd.concat([X_phase1, X_boundary[boundary_cols]], axis=1)
print(f"Features with boundary: {X_phase1.shape}")

# Store Phase 1 column names
phase1_cols = list(X_phase1.columns)
n_phase1 = len(phase1_cols)
print(f"Phase 1 column count: {n_phase1}")

# Impute NaN
print("\nImputing NaN values...")
X_phase1 = X_phase1.fillna(X_phase1.median())
print(f"Feature shape after imputation: {X_phase1.shape}\n")

# Generate interactions from top Phase 1 features
from sb.features.interactions import generate_interactions, add_differentiation_features

print("="*70)
print("FEATURE INTERACTION GENERATION")
print("="*70 + "\n")

# Select top 50 Phase 1 features for interaction generation
print("Selecting top 50 Phase 1 features by mutual information...")
mi_scores = mutual_info_classif(X_phase1, y, random_state=42)
top_50_idx = np.argsort(mi_scores)[-50:][::-1]
top_50_features = X_phase1.columns[top_50_idx].tolist()

print("Top 10 features selected:")
for i, idx in enumerate(top_50_idx[:10], 1):
    feat = X_phase1.columns[idx]
    print(f"  {i:2d}. {feat:50s} MI={mi_scores[idx]:.4f}")

# Generate interactions from top 50 features
print(f"\nGenerating interactions from top {len(top_50_features)} features...")
X_top50 = X_phase1[top_50_features]

operations = ['mul', 'div', 'add', 'sub', 'sqmul', 'ratio', 'harmonic']
print(f"Operations: {operations}")

# Generate pairwise interactions
interaction_features = {}
interaction_count = 0
max_interactions = 5000

for i, col1 in enumerate(X_top50.columns):
    for j, col2 in enumerate(X_top50.columns):
        if i >= j:  # Skip self and duplicates
            continue
        
        if interaction_count >= max_interactions:
            break
        
        x1 = X_top50[col1].values
        x2 = X_top50[col2].values
        
        # Generate different operations
        for op in operations:
            if op == 'mul':
                interaction_features[f'mul_{col1}_{col2}'] = x1 * x2
            elif op == 'div':
                with np.errstate(divide='ignore', invalid='ignore'):
                    interaction_features[f'div_{col1}_{col2}'] = np.where(x2 != 0, x1 / x2, 0)
            elif op == 'add':
                interaction_features[f'add_{col1}_{col2}'] = x1 + x2
            elif op == 'sub':
                interaction_features[f'sub_{col1}_{col2}'] = x1 - x2
            elif op == 'sqmul':
                interaction_features[f'sqmul_{col1}_{col2}'] = (x1 ** 2) * (x2 ** 2)
            elif op == 'ratio':
                with np.errstate(divide='ignore', invalid='ignore'):
                    interaction_features[f'ratio_{col1}_{col2}'] = np.where(
                        x1 + x2 != 0, (x1 - x2) / (x1 + x2), 0
                    )
            elif op == 'harmonic':
                # Harmonic mean: 2*x1*x2/(x1+x2) - NOT used by winners
                with np.errstate(divide='ignore', invalid='ignore'):
                    interaction_features[f'harmonic_{col1}_{col2}'] = np.where(
                        x1 + x2 != 0, 2 * x1 * x2 / (x1 + x2), 0
                    )
        
        interaction_count += 1
        if interaction_count >= max_interactions:
            break
    
    if interaction_count >= max_interactions:
        break

X_interactions = pd.DataFrame(interaction_features, index=X_phase1.index)
print(f"\nGenerated {len(interaction_features)} interaction features")

# Add differentiation features (trig, polynomials, etc.)
print("\nAdding unique differentiation features...")
diff_features = {}

# Trigonometric transformations (unique to our approach)
for col in top_50_features[:10]:  # Top 10 only
    x_norm = (X_phase1[col] - X_phase1[col].mean()) / (X_phase1[col].std() + 1e-8)
    diff_features[f'{col}_sin'] = np.sin(x_norm)
    diff_features[f'{col}_cos'] = np.cos(x_norm)

# Polynomial features
for col in top_50_features[:5]:
    diff_features[f'{col}_sqrt'] = np.sqrt(np.abs(X_phase1[col]))
    diff_features[f'{col}_log'] = np.log1p(np.abs(X_phase1[col]))

X_diff = pd.DataFrame(diff_features, index=X_phase1.index)
print(f"Added {len(diff_features)} differentiation features")

# Combine all interactions
X_all_interactions = pd.concat([X_interactions, X_diff], axis=1)
print(f"Total interaction features: {X_all_interactions.shape[1]}")

# Feature selection on INTERACTIONS ONLY
print("\n" + "="*70)
print("FEATURE SELECTION ON INTERACTIONS")
print("="*70 + "\n")

# Remove constant interactions
print("Step 1: Removing constant interactions...")
non_constant = X_all_interactions.std() > 0
X_all_interactions = X_all_interactions.loc[:, non_constant]
print(f"  Removed {(~non_constant).sum()} constant features")
print(f"  Remaining: {X_all_interactions.shape[1]}")

# Correlation filter on interactions
print("\nStep 2: Correlation filter on interactions (threshold: 0.95)...")
corr_matrix = X_all_interactions.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X_all_interactions = X_all_interactions.drop(columns=to_drop)
print(f"  Removed {len(to_drop)} highly correlated features")
print(f"  Remaining: {X_all_interactions.shape[1]}")

# Select top 150 interactions by mutual information
print("\nStep 3: Selecting top 150 interactions by mutual information...")
mi_interact = mutual_info_classif(X_all_interactions, y, random_state=42)
top_150_idx = np.argsort(mi_interact)[-150:][::-1]
selected_interaction_cols = X_all_interactions.columns[top_150_idx].tolist()
X_selected_interactions = X_all_interactions[selected_interaction_cols]
print(f"  Selected {len(selected_interaction_cols)} interactions")

# COMBINE Phase 1 + selected interactions (THIS IS THE KEY FIX!)
print("\n" + "="*70)
print("COMBINING PHASE 1 + INTERACTIONS")
print("="*70 + "\n")

X_final = pd.concat([X_phase1, X_selected_interactions], axis=1)
print(f"Phase 1 features: {n_phase1}")
print(f"Interaction features: {len(selected_interaction_cols)}")
print(f"Total features: {X_final.shape[1]}\n")

print("Top 20 selected interactions:")
for i, idx in enumerate(top_150_idx[:20], 1):
    feat = X_all_interactions.columns[idx]
    print(f"  {i:2d}. {feat:60s} MI={mi_interact[idx]:.4f}")

# Cross-validation
print("\n" + "="*70)
print("CROSS-VALIDATION WITH PHASE 1 + INTERACTIONS")
print("="*70 + "\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []
models = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_final, y), 1):
    X_train = X_final.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X_final.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    # Rank normalize inside fold
    X_train_rank = X_train.rank(pct=True)
    X_val_rank = X_val.rank(pct=True)
    
    # Train LightGBM
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
    
    models.append(model)
    
    y_pred = model.predict(X_val_rank)
    auc = roc_auc_score(y_val, y_pred)
    fold_aucs.append(auc)
    
    print(f"Fold {fold_idx}/5: AUC = {auc:.4f}")

mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)

print("\n" + "="*70)
print("RESULTS")
print("="*70 + "\n")

print(f"CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"Fold AUCs: {[f'{auc:.4f}' for auc in fold_aucs]}")

# Compare to Phase 1 baseline
phase1_auc = 0.8720
improvement = mean_auc - phase1_auc
print(f"\nImprovement over Phase 1 (0.8720): {improvement:+.4f}")

if improvement > 0.005:
    print("\n✅ Interactions helped! Model improved significantly.")
elif improvement > 0:
    print("\n✅ Slight improvement from interactions.")
else:
    print("\n⚠️  Interactions didn't help. Consider trying:")
    print("   - Different interaction operations")
    print("   - Ensemble approach")
    print("   - More selective interaction pairs")

# Train final model on all data
print("\n" + "="*70)
print("TRAINING FINAL MODEL")
print("="*70 + "\n")

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
    print(f"  {row['feature']:60s}: {row['importance']:.1f}")

# Check how many top features are interactions vs Phase 1
top_20 = feature_importance.head(20)
n_phase1_top20 = sum(1 for f in top_20['feature'] if f in phase1_cols)
n_interact_top20 = 20 - n_phase1_top20
print(f"\nTop 20 features breakdown:")
print(f"  Phase 1 features: {n_phase1_top20}")
print(f"  Interaction features: {n_interact_top20}")

# Save model
model_path = Path("models")
model_path.mkdir(exist_ok=True)
joblib.dump(final_model, model_path / "model_interactions_v2.joblib")
print(f"\n✅ Model saved to: {model_path / 'model_interactions_v2.joblib'}")

print("\n" + "="*70)
print(f"FINAL CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print("="*70)
