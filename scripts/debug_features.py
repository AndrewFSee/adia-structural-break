"""
Quick feature quality check.
Run this to see if your features have signal.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sb import config, data_loader, features

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(y)} series, break rate: {y.mean():.2%}")

# Extract features
print("\nExtracting features...")
X_raw = features.base.compute_features(
    df,
    use_multiscale=True,
    use_boundary_dist=True,
    use_boundary_tail_shape=True
)
print(f"Feature shape: {X_raw.shape}")

# Check NaN statistics
nan_pct = X_raw.isna().mean(axis=0)
print(f"\nNaN Statistics:")
print(f"  Mean NaN %: {nan_pct.mean():.1%}")
print(f"  Max NaN %: {nan_pct.max():.1%}")
print(f"  Features with >50% NaN: {(nan_pct > 0.5).sum()}/{len(nan_pct)}")
print(f"  Features with >80% NaN: {(nan_pct > 0.8).sum()}/{len(nan_pct)}")

# Compute per-feature AUC
print("\nComputing per-feature AUC...")
aucs = {}
for col in X_raw.columns:
    x = X_raw[col].fillna(X_raw[col].median())
    if x.std() > 0:
        try:
            auc1 = roc_auc_score(y, x)
            auc2 = roc_auc_score(y, -x)
            aucs[col] = max(auc1, auc2)
        except:
            aucs[col] = 0.5
    else:
        aucs[col] = 0.5

aucs_series = pd.Series(aucs).sort_values(ascending=False)

print(f"\nFeature AUC Statistics:")
print(f"  Mean: {aucs_series.mean():.4f}")
print(f"  Median: {aucs_series.median():.4f}")
print(f"  Max: {aucs_series.max():.4f}")
print(f"  Min: {aucs_series.min():.4f}")
print(f"  Features with AUC > 0.60: {(aucs_series > 0.60).sum()}/{len(aucs_series)}")
print(f"  Features with AUC > 0.55: {(aucs_series > 0.55).sum()}/{len(aucs_series)}")
print(f"  Features with AUC ≈ 0.50: {((aucs_series > 0.49) & (aucs_series < 0.51)).sum()}/{len(aucs_series)}")

print(f"\nTop 20 features:")
for feat, auc in aucs_series.head(20).items():
    print(f"  {feat:45s}: {auc:.4f}")

print(f"\nWorst 10 features:")
for feat, auc in aucs_series.tail(10).items():
    print(f"  {feat:45s}: {auc:.4f}")

# Simple rank-mean baseline
print("\n" + "="*70)
print("BASELINE: Simple Rank-Mean Aggregation")
print("="*70)
X_ranked = X_raw.rank(pct=True).fillna(0.5)
baseline_score = X_ranked.mean(axis=1)
baseline_auc = roc_auc_score(y, baseline_score)
print(f"Baseline AUC: {baseline_auc:.4f}")

if baseline_auc < 0.65:
    print("\n❌ CRITICAL: Features contain very weak signal!")
    print("   Problem is FEATURE ENGINEERING, not modeling.")
elif baseline_auc < 0.72:
    print("\n⚠️  WARNING: Features contain moderate signal.")
    print("   GBM can help but won't achieve >0.80 AUC.")
else:
    print("\n✅ Features contain good signal.")
    print("   GBM should be able to improve further.")
