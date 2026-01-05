"""
Test autoencoder features on a small subset.

This script:
1. Tests autoencoder on 100 series (fast)
2. Compares AUC with/without autoencoder features
3. Shows feature importance

If promising, we'll add to full training pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from sb import data_loader

print("="*70)
print("TESTING AUTOENCODER FEATURES (SMALL SUBSET)")
print("="*70)
print()

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")

# Test on first 1000 series only (for speed)
n_test = 1000
series_ids = df.index.get_level_values(0).unique()[:n_test]
df_test = df.loc[series_ids]
y_test = y.loc[series_ids]

print(f"Testing on {len(series_ids)} series, break rate: {y_test.mean()*100:.2f}%\n")

# Compute simple baseline features (fast)
print("Computing baseline features...")
baseline_features = []

for series_id in series_ids:
    series_data = df_test.loc[series_id]
    
    pre = series_data[series_data['period'] == 0]['value'].values
    post = series_data[series_data['period'] == 1]['value'].values
    
    baseline_features.append({
        'mean_pre': np.mean(pre),
        'mean_post': np.mean(post),
        'std_pre': np.std(pre),
        'std_post': np.std(post),
        'mean_diff': np.mean(post) - np.mean(pre),
        'std_diff': np.std(post) - np.std(pre),
        'cv_pre': np.std(pre) / (np.abs(np.mean(pre)) + 1e-8),
        'cv_post': np.std(post) / (np.abs(np.mean(post)) + 1e-8),
    })

X_baseline = pd.DataFrame(baseline_features, index=series_ids)
print(f"Baseline features: {X_baseline.shape}")

# Compute autoencoder features
print("\nComputing autoencoder features (this may take a few minutes)...")
from sb.features.autoencoder import compute_autoencoder_features

X_ae = compute_autoencoder_features(
    df_test,
    window_size=20,
    stride=5,
    epochs=30,
    verbose=True
)
print(f"Autoencoder features: {X_ae.shape}")

# Baseline evaluation
print("\n" + "="*70)
print("BASELINE (without autoencoder)")
print("="*70 + "\n")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
baseline_aucs = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_baseline, y_test), 1):
    X_train = X_baseline.iloc[train_idx]
    y_train = y_test.iloc[train_idx]
    X_val = X_baseline.iloc[val_idx]
    y_val = y_test.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    baseline_aucs.append(auc)
    print(f"Fold {fold_idx}/3: AUC = {auc:.4f}")

baseline_mean = np.mean(baseline_aucs)
print(f"\nBaseline AUC: {baseline_mean:.4f} ± {np.std(baseline_aucs):.4f}")

# With autoencoder features
print("\n" + "="*70)
print("WITH AUTOENCODER FEATURES")
print("="*70 + "\n")

X_combined = pd.concat([X_baseline, X_ae], axis=1)
print(f"Combined features: {X_combined.shape}")

ae_aucs = []
ae_importances = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_test), 1):
    X_train = X_combined.iloc[train_idx]
    y_train = y_test.iloc[train_idx]
    X_val = X_combined.iloc[val_idx]
    y_val = y_test.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    ae_aucs.append(auc)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_combined.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    ae_importances.append(importance)
    
    print(f"Fold {fold_idx}/3: AUC = {auc:.4f}")

ae_mean = np.mean(ae_aucs)
print(f"\nWith Autoencoder AUC: {ae_mean:.4f} ± {np.std(ae_aucs):.4f}")

# Summary
print("\n" + "="*70)
print("RESULTS")
print("="*70 + "\n")

print(f"Baseline:       {baseline_mean:.4f}")
print(f"With AE:        {ae_mean:.4f}")
print(f"Improvement:    {ae_mean - baseline_mean:+.4f}")

if ae_mean > baseline_mean:
    print("\n✅ Autoencoder features HELPED!")
else:
    print("\n⚠️  Autoencoder features didn't help on this subset")

# Show autoencoder feature importance
print("\nAutoencoder feature importances (average across folds):")
avg_importance = pd.concat(ae_importances).groupby('feature')['importance'].mean().sort_values(ascending=False)
ae_features = [f for f in avg_importance.index if f.startswith('ae_')]

if ae_features:
    print("\nTop autoencoder features:")
    for feat in ae_features[:10]:
        print(f"  {feat:40s}: {avg_importance[feat]:.1f}")
    
    ae_total = avg_importance[ae_features].sum()
    baseline_total = avg_importance[[f for f in avg_importance.index if not f.startswith('ae_')]].sum()
    ae_percent = 100 * ae_total / (ae_total + baseline_total)
    
    print(f"\nAutoencoder features account for {ae_percent:.1f}% of total importance")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if ae_mean - baseline_mean > 0.005:
    print("\n✅ RECOMMENDED: Add autoencoder features to full pipeline")
    print("   They provide meaningful improvement (+{:.4f} AUC)".format(ae_mean - baseline_mean))
elif ae_mean > baseline_mean:
    print("\n⚠️  MARGINAL: Slight improvement but may not be worth the computation time")
    print("   Consider if you need every 0.001 AUC")
else:
    print("\n❌ NOT RECOMMENDED: Autoencoder features didn't improve performance")
    print("   Focus on other feature types (statistical tests, etc.)")
