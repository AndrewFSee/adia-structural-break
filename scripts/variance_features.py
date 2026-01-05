"""
Variance-reduction focused features for structural break detection.

Key insight: Breaks are associated with LOWER variance (0.0030 vs 0.0119).
Standard features miss this pattern.
"""

import numpy as np
import pandas as pd
from typing import Dict

def variance_reduction_features(x0: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """
    Features specifically designed to detect variance reduction.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # === VARIANCE CHANGES (Primary Signal) ===
    
    var0 = np.var(x0) if len(x0) > 1 else 0
    var1 = np.var(x1) if len(x1) > 1 else 0
    
    # Variance ratio (breaks have ratio < 1)
    features['var_ratio'] = var1 / (var0 + 1e-8)
    features['var_log_ratio'] = np.log((var1 + 1e-8) / (var0 + 1e-8))
    features['var_reduction'] = var0 - var1  # Positive = variance decreased
    features['var_reduction_pct'] = (var0 - var1) / (var0 + 1e-8)
    
    # MAD-based (more robust)
    mad0 = np.median(np.abs(x0 - np.median(x0)))
    mad1 = np.median(np.abs(x1 - np.median(x1)))
    features['mad_ratio'] = mad1 / (mad0 + 1e-8)
    features['mad_reduction'] = mad0 - mad1
    features['mad_reduction_pct'] = (mad0 - mad1) / (mad0 + 1e-8)
    
    # IQR-based
    iqr0 = np.percentile(x0, 75) - np.percentile(x0, 25)
    iqr1 = np.percentile(x1, 75) - np.percentile(x1, 25)
    features['iqr_ratio'] = iqr1 / (iqr0 + 1e-8)
    features['iqr_reduction'] = iqr0 - iqr1
    
    # Range-based
    range0 = np.percentile(x0, 95) - np.percentile(x0, 5)
    range1 = np.percentile(x1, 95) - np.percentile(x1, 5)
    features['range_ratio'] = range1 / (range0 + 1e-8)
    features['range_reduction'] = range0 - range1
    
    # === STABILITY MEASURES ===
    
    # Coefficient of variation (CV = std/mean)
    mean0, mean1 = np.mean(x0), np.mean(x1)
    std0, std1 = np.std(x0), np.std(x1)
    
    cv0 = std0 / (np.abs(mean0) + 1e-8)
    cv1 = std1 / (np.abs(mean1) + 1e-8)
    features['cv_ratio'] = cv1 / (cv0 + 1e-8)
    features['cv_reduction'] = cv0 - cv1
    
    # Relative stability (inverse CV)
    features['stability0'] = 1.0 / (cv0 + 1e-8)
    features['stability1'] = 1.0 / (cv1 + 1e-8)
    features['stability_increase'] = features['stability1'] - features['stability0']
    
    # === SMOOTHNESS MEASURES ===
    
    # First differences (measure jumpiness)
    if len(x0) > 1:
        diff0 = np.diff(x0)
        features['diff_std0'] = np.std(diff0)
        features['diff_mad0'] = np.median(np.abs(diff0 - np.median(diff0)))
    else:
        features['diff_std0'] = 0
        features['diff_mad0'] = 0
    
    if len(x1) > 1:
        diff1 = np.diff(x1)
        features['diff_std1'] = np.std(diff1)
        features['diff_mad1'] = np.median(np.abs(diff1 - np.median(diff1)))
    else:
        features['diff_std1'] = 0
        features['diff_mad1'] = 0
    
    features['diff_std_ratio'] = features['diff_std1'] / (features['diff_std0'] + 1e-8)
    features['diff_mad_ratio'] = features['diff_mad1'] / (features['diff_mad0'] + 1e-8)
    features['smoothness_increase'] = features['diff_std0'] - features['diff_std1']
    
    # === EXTREME VALUE CHANGES ===
    
    # Max absolute deviation from median
    max_dev0 = np.max(np.abs(x0 - np.median(x0)))
    max_dev1 = np.max(np.abs(x1 - np.median(x1)))
    features['max_dev_ratio'] = max_dev1 / (max_dev0 + 1e-8)
    features['max_dev_reduction'] = max_dev0 - max_dev1
    
    # Outlier counts (values beyond 2*MAD)
    outliers0 = np.sum(np.abs(x0 - np.median(x0)) > 2 * mad0)
    outliers1 = np.sum(np.abs(x1 - np.median(x1)) > 2 * mad1)
    features['outlier_ratio'] = outliers1 / (outliers0 + 1e-8)
    features['outlier_reduction'] = outliers0 - outliers1
    
    # === DISTRIBUTION SHAPE ===
    
    # Kurtosis (tailedness)
    from scipy import stats
    try:
        kurt0 = stats.kurtosis(x0, fisher=True, nan_policy='omit')
        kurt1 = stats.kurtosis(x1, fisher=True, nan_policy='omit')
        features['kurtosis_change'] = kurt1 - kurt0
        features['kurtosis_reduction'] = kurt0 - kurt1  # Positive = tails got thinner
    except:
        features['kurtosis_change'] = 0
        features['kurtosis_reduction'] = 0
    
    # Skewness
    try:
        skew0 = stats.skew(x0, nan_policy='omit')
        skew1 = stats.skew(x1, nan_policy='omit')
        features['skewness_change'] = np.abs(skew1) - np.abs(skew0)
    except:
        features['skewness_change'] = 0
    
    # === AUTOCORRELATION (Predictability) ===
    
    if len(x0) > 2 and np.std(x0) > 0:
        features['acf1_0'] = np.corrcoef(x0[:-1], x0[1:])[0, 1]
    else:
        features['acf1_0'] = 0
        
    if len(x1) > 2 and np.std(x1) > 0:
        features['acf1_1'] = np.corrcoef(x1[:-1], x1[1:])[0, 1]
    else:
        features['acf1_1'] = 0
    
    features['acf1_change'] = features['acf1_1'] - features['acf1_0']
    features['acf1_increase'] = np.abs(features['acf1_1']) - np.abs(features['acf1_0'])
    
    # === MEAN CHANGES (Secondary) ===
    
    features['mean_shift'] = mean1 - mean0
    features['mean_shift_normalized'] = (mean1 - mean0) / (std0 + 1e-8)
    features['median_shift'] = np.median(x1) - np.median(x0)
    
    # === RATIO OF RATIOS (Interaction) ===
    
    # Capture if variance changed MORE than mean
    mean_change_pct = np.abs(mean1 - mean0) / (np.abs(mean0) + 1e-8)
    var_change_pct = np.abs(var1 - var0) / (var0 + 1e-8)
    features['var_dominance'] = var_change_pct / (mean_change_pct + 1e-8)
    
    return features


def compute_variance_features_multiscale(x0: np.ndarray, x1: np.ndarray, 
                                         windows=[50, 100, 250]) -> Dict[str, float]:
    """
    Compute variance features at multiple scales.
    """
    features = {}
    
    # Full-scale features
    full_feats = variance_reduction_features(x0, x1)
    for k, v in full_feats.items():
        features[f'{k}_full'] = v
    
    # Multi-scale features (boundary-focused)
    for w in windows:
        # Last w points of x0, first w points of x1
        x0_w = x0[-w:] if len(x0) >= w else x0
        x1_w = x1[:w] if len(x1) >= w else x1
        
        if len(x0_w) > 0 and len(x1_w) > 0:
            w_feats = variance_reduction_features(x0_w, x1_w)
            for k, v in w_feats.items():
                features[f'{k}_w{w}'] = v
    
    return features


if __name__ == '__main__':
    import sys
    from pathlib import Path
    from sklearn.metrics import roc_auc_score
    
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from sb import data_loader
    
    print("Testing variance-reduction features...")
    print("="*70)
    
    # Load data
    df, y = data_loader.load_for_training("data")
    
    # Compute features
    features_list = []
    for series_id in df['id'].unique()[:5000]:  # Test on subset first
        series_data = df[df['id'] == series_id]
        x0 = series_data[series_data['period'] == 0]['value'].values
        x1 = series_data[series_data['period'] == 1]['value'].values
        
        if len(x0) > 10 and len(x1) > 10:
            feats = compute_variance_features_multiscale(x0, x1)
            feats['id'] = series_id
            features_list.append(feats)
    
    X = pd.DataFrame(features_list).set_index('id')
    y_subset = y[y.index.isin(X.index)]
    
    print(f"Feature shape: {X.shape}")
    print(f"NaN count: {X.isna().sum().sum()} ({X.isna().sum().sum() / X.size * 100:.1f}%)")
    
    # Fill NaN
    X_filled = X.fillna(X.median())
    
    # Rank and aggregate
    X_ranked = X_filled.rank(pct=True)
    scores = X_ranked.mean(axis=1)
    
    auc = roc_auc_score(y_subset, scores)
    print(f"\nVariance-focused baseline AUC: {auc:.4f}")
    
    # Per-feature AUC
    print(f"\nTop 20 variance-focused features:")
    aucs = {}
    for col in X_filled.columns:
        try:
            auc1 = roc_auc_score(y_subset, X_filled[col])
            auc2 = roc_auc_score(y_subset, -X_filled[col])
            aucs[col] = max(auc1, auc2)
        except:
            aucs[col] = 0.5
    
    for feat, auc_val in sorted(aucs.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {feat:40s}: {auc_val:.4f}")
    
    print("\n" + "="*70)
    if auc > 0.75:
        print("✅ EXCELLENT! Variance features work much better!")
        print("   Integrate these into your main pipeline.")
    elif auc > 0.70:
        print("✅ GOOD! Variance features are better.")
        print("   Combine with existing features for best results.")
    elif auc > 0.65:
        print("⚠️  MODERATE improvement.")
        print("   May help but won't solve the problem alone.")
    else:
        print("❌ Still weak. Problem may be fundamental.")
