"""
Train with aggressive feature selection - only features with AUC > threshold.

Based on diagnostic showing:
- 640 features, only 137 (21%) have AUC > 0.55
- Best feature: 0.6093, most are ~0.50
- Aggregate baseline: 0.5786

Strategy: Use only strong features to reduce noise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sb import data_loader, features

print("="*70)
print("SELECTIVE FEATURE TRAINING")
print("="*70)
print()

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(df['id'].unique())} series, break rate: {y.mean()*100:.2f}%\n")

# Extract features
print("Extracting features (multiscale + boundary features)...")
X_raw = features.base.compute_features(
    df,
    use_multiscale=True,
    use_spectral=False,
    use_wavelet=False,
    use_break_likelihood=False,
    use_boundary=False,
    use_boundary_dist=True,
    use_boundary_tail_shape=True
)
print(f"Initial feature shape: {X_raw.shape}\n")

# Impute NaN
print("Imputing NaN values...")
X = X_raw.fillna(X_raw.median())
print(f"Feature shape after imputation: {X.shape}\n")

# === STEP 1: Compute per-feature AUC ===
print("="*70)
print("STEP 1: Computing per-feature AUC")
print("="*70)

feature_aucs = {}
for col in X.columns:
    try:
        # Try both directions
        auc1 = roc_auc_score(y, X[col])
        auc2 = roc_auc_score(y, -X[col])
        feature_aucs[col] = max(auc1, auc2)
    except:
        feature_aucs[col] = 0.5

auc_df = pd.DataFrame([
    {'feature': k, 'auc': v} 
    for k, v in feature_aucs.items()
]).sort_values('auc', ascending=False)

print(f"\nFeature AUC statistics:")
print(f"  Mean:   {auc_df['auc'].mean():.4f}")
print(f"  Median: {auc_df['auc'].median():.4f}")
print(f"  Max:    {auc_df['auc'].max():.4f}")
print(f"  Min:    {auc_df['auc'].min():.4f}")

# === STEP 2: Try different thresholds ===
print("\n" + "="*70)
print("STEP 2: Testing different feature selection thresholds")
print("="*70 + "\n")

from sklearn.model_selection import StratifiedKFold

def cv_with_features(X_subset, y, n_folds=5):
    """Quick CV with given features."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aucs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_subset, y)):
        X_train = X_subset.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X_subset.iloc[val_idx]
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
        
        y_pred = model.predict(X_val_rank)
        auc = roc_auc_score(y_val, y_pred)
        fold_aucs.append(auc)
    
    return np.mean(fold_aucs), np.std(fold_aucs)

# Test thresholds
thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
results = []

for thresh in thresholds:
    selected = auc_df[auc_df['auc'] > thresh]['feature'].tolist()
    n_selected = len(selected)
    
    if n_selected == 0:
        print(f"Threshold {thresh:.2f}: No features selected, skipping")
        continue
    
    X_selected = X[selected]
    
    print(f"Threshold {thresh:.2f}: {n_selected} features selected")
    print(f"  Testing with 5-fold CV...")
    
    mean_auc, std_auc = cv_with_features(X_selected, y)
    
    print(f"  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")
    
    results.append({
        'threshold': thresh,
        'n_features': n_selected,
        'cv_auc': mean_auc,
        'cv_std': std_auc
    })

# === STEP 3: Summary ===
print("="*70)
print("SUMMARY")
print("="*70 + "\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_idx = results_df['cv_auc'].idxmax()
best = results_df.iloc[best_idx]

print(f"\n✅ Best threshold: {best['threshold']:.2f}")
print(f"   Features: {best['n_features']}")
print(f"   CV AUC: {best['cv_auc']:.4f} ± {best['cv_std']:.4f}")

# Compare to baseline (all features)
print(f"\nBaseline (all {X.shape[1]} features):")
mean_auc, std_auc = cv_with_features(X, y)
print(f"  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

improvement = best['cv_auc'] - mean_auc
print(f"\n{'✅' if improvement > 0 else '❌'} Improvement: {improvement:+.4f}")

# Show top features
print(f"\nTop 30 features (AUC > {best['threshold']:.2f}):")
selected_features = auc_df[auc_df['auc'] > best['threshold']].head(30)
for idx, row in selected_features.iterrows():
    print(f"  {row['feature']:50s} {row['auc']:.4f}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if best['cv_auc'] > 0.72:
    print("\n✅ Feature selection helps significantly!")
    print(f"   Use threshold {best['threshold']:.2f} ({best['n_features']} features)")
    print("   Add feature selection to your pipeline.")
elif best['cv_auc'] > 0.70 and improvement > 0.005:
    print("\n⚠️  Modest improvement from feature selection.")
    print(f"   Using top {best['n_features']} features gives +{improvement:.4f} AUC")
    print("   May be worth it, but won't solve the problem.")
else:
    print("\n❌ Feature selection doesn't help meaningfully.")
    print("   The problem is signal quality, not feature noise.")
    print("\n   The harsh truth:")
    print("   • Best single feature: 0.61 AUC")
    print("   • Current GBM: 0.70 AUC") 
    print("   • GBM is doing its job - features are weak")
    print("\n   Options:")
    print("   1. Accept 0.70 AUC as the ceiling for this data")
    print("   2. Get more/better training data")
    print("   3. Use completely different features (external data?)")
    print("   4. Ensemble with other detection methods")
