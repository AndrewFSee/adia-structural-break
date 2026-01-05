"""
Alternative feature engineering approaches when standard features aren't working.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sb import data_loader

def simple_robust_features(x0, x1):
    """
    Very simple, robust features that often work when fancy features don't.
    """
    features = {}
    
    # Robust location shifts
    features['median_shift'] = np.median(x1) - np.median(x0)
    features['mean_shift'] = np.mean(x1) - np.mean(x0)
    
    # Robust scale changes
    mad0 = np.median(np.abs(x0 - np.median(x0)))
    mad1 = np.median(np.abs(x1 - np.median(x1)))
    features['mad_ratio'] = mad1 / (mad0 + 1e-8)
    
    features['std_ratio'] = np.std(x1) / (np.std(x0) + 1e-8)
    
    # Range changes
    features['range0'] = np.percentile(x0, 90) - np.percentile(x0, 10)
    features['range1'] = np.percentile(x1, 90) - np.percentile(x1, 10)
    features['range_ratio'] = features['range1'] / (features['range0'] + 1e-8)
    
    # Quantile shifts at multiple levels
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        q0 = np.percentile(x0, q * 100)
        q1 = np.percentile(x1, q * 100)
        features[f'q{int(q*100)}_shift'] = q1 - q0
    
    # Extreme value changes
    features['max_shift'] = np.max(x1) - np.max(x0)
    features['min_shift'] = np.min(x1) - np.min(x0)
    
    # Sign changes
    features['positive_ratio0'] = (x0 > 0).mean()
    features['positive_ratio1'] = (x1 > 0).mean()
    features['sign_change'] = features['positive_ratio1'] - features['positive_ratio0']
    
    # Crossing zero
    features['crosses_zero0'] = int(np.any(x0 > 0) and np.any(x0 < 0))
    features['crosses_zero1'] = int(np.any(x1 > 0) and np.any(x1 < 0))
    
    # Simple autocorrelation
    if len(x0) > 1:
        features['autocorr0'] = np.corrcoef(x0[:-1], x0[1:])[0, 1] if np.std(x0) > 0 else 0
    else:
        features['autocorr0'] = 0
        
    if len(x1) > 1:
        features['autocorr1'] = np.corrcoef(x1[:-1], x1[1:])[0, 1] if np.std(x1) > 0 else 0
    else:
        features['autocorr1'] = 0
    
    features['autocorr_shift'] = features['autocorr1'] - features['autocorr0']
    
    # Trend (simple linear fit)
    if len(x0) > 2:
        t0 = np.arange(len(x0))
        features['trend0'] = np.polyfit(t0, x0, 1)[0]
    else:
        features['trend0'] = 0
        
    if len(x1) > 2:
        t1 = np.arange(len(x1))
        features['trend1'] = np.polyfit(t1, x1, 1)[0]
    else:
        features['trend1'] = 0
    
    features['trend_change'] = features['trend1'] - features['trend0']
    
    return features

# Test on actual data
print("Testing simple robust features...")
df, y = data_loader.load_for_training("data")

features_list = []
for series_id in df['id'].unique():
    series_data = df[df['id'] == series_id]
    x0 = series_data[series_data['period'] == 0]['value'].values
    x1 = series_data[series_data['period'] == 1]['value'].values
    
    if len(x0) > 0 and len(x1) > 0:
        feats = simple_robust_features(x0, x1)
        feats['id'] = series_id
        features_list.append(feats)

X = pd.DataFrame(features_list).set_index('id')
X = X.reindex(y.index)

print(f"\nSimple features shape: {X.shape}")
print(f"NaN count: {X.isna().sum().sum()}")

# Fill NaN
X_filled = X.fillna(X.median())

# Rank and aggregate
X_ranked = X_filled.rank(pct=True)
scores = X_ranked.mean(axis=1)

auc = roc_auc_score(y, scores)
print(f"\nSimple features baseline AUC: {auc:.4f}")

# Per-feature AUC
print(f"\nTop 10 simple features:")
aucs = {}
for col in X_filled.columns:
    try:
        auc1 = roc_auc_score(y, X_filled[col])
        auc2 = roc_auc_score(y, -X_filled[col])
        aucs[col] = max(auc1, auc2)
    except:
        aucs[col] = 0.5

for feat, auc in sorted(aucs.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {feat:30s}: {auc:.4f}")

print("\n" + "="*70)
if auc > 0.72:
    print("✅ Simple features work better than complex features!")
    print("   Consider using these instead.")
elif auc > 0.68:
    print("⚠️  Simple features are competitive.")
    print("   Complex features may not be adding value.")
else:
    print("❌ Even simple features struggle.")
    print("   This problem may be fundamentally difficult.")
