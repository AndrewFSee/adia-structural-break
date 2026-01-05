"""
Time series transformations for feature engineering.

Winning solutions computed features on multiple transformations of the raw series:
- CUMSUM: Cumulative sum (emphasizes level shifts)
- DIFF: First differences (emphasizes volatility/variance changes)
- RANK: Dense ranking (removes outlier influence)
- EWMA: Exponentially weighted moving average (smoothed series)
- MOSUM: Moving sum of squares (local variance)

Computing the same features on different transformations captures different aspects
of structural breaks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats


def transform_series(x0: np.ndarray, x1: np.ndarray, 
                     transform_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply transformation to time series.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        transform_type: One of 'raw', 'cumsum', 'diff', 'rank', 'ewma', 'mosum', 'abs', 'zscore'
        
    Returns:
        Transformed (x0, x1)
    """
    x_all = np.concatenate([x0, x1])
    
    if transform_type == 'raw':
        return x0.copy(), x1.copy()
    
    elif transform_type == 'cumsum':
        # Cumulative sum - emphasizes mean level shifts
        x_cumsum = np.cumsum(x_all)
        return x_cumsum[:len(x0)], x_cumsum[len(x0):]
    
    elif transform_type == 'diff':
        # First differences - emphasizes volatility changes
        if len(x_all) > 1:
            x_diff = np.diff(x_all)
            # Split back (one element shorter)
            split_idx = len(x0) - 1
            return x_diff[:split_idx], x_diff[split_idx:]
        else:
            return x0.copy(), x1.copy()
    
    elif transform_type == 'rank':
        # Dense ranking - removes outlier influence
        x_rank = stats.rankdata(x_all, method='dense')
        return x_rank[:len(x0)], x_rank[len(x0):]
    
    elif transform_type == 'ewma':
        # Exponentially weighted moving average (span=20)
        x_ewma = pd.Series(x_all).ewm(span=20, adjust=False).mean().values
        return x_ewma[:len(x0)], x_ewma[len(x0):]
    
    elif transform_type == 'mosum':
        # Moving sum of squares (window=50) - local variance proxy
        window = min(50, len(x_all) // 2)
        if window > 0:
            x_sq = x_all ** 2
            x_mosum = pd.Series(x_sq).rolling(window=window, min_periods=1).sum().values
            return x_mosum[:len(x0)], x_mosum[len(x0):]
        else:
            return x0.copy(), x1.copy()
    
    elif transform_type == 'abs':
        # Absolute value
        return np.abs(x0), np.abs(x1)
    
    elif transform_type == 'zscore':
        # Z-score normalization
        mean = np.mean(x_all)
        std = np.std(x_all)
        if std > 1e-10:
            x_z = (x_all - mean) / std
            return x_z[:len(x0)], x_z[len(x0):]
        else:
            return x_all[:len(x0)], x_all[len(x0):]
    
    elif transform_type == 'residual':
        # Residuals from EWMA (standardized)
        x_ewma = pd.Series(x_all).ewm(span=20, adjust=False).mean().values
        x_ewma_std = pd.Series(x_all).ewm(span=20, adjust=False).std().values
        residuals = (x_all - x_ewma) / (x_ewma_std + 1e-8)
        return residuals[:len(x0)], residuals[len(x0):]
    
    else:
        raise ValueError(f"Unknown transform: {transform_type}")


def get_transform_names() -> list:
    """
    Get list of available transformations.
    
    Returns priority transformations used by winning solutions first.
    """
    return [
        'raw',      # Baseline
        'cumsum',   # Level shifts (used by 2nd place + Chinese team)
        'diff',     # Volatility changes (used by Chinese team)
        'rank',     # Outlier-robust (used by 2nd place)
        'ewma',     # Smoothed (used by 6th place)
        'mosum',    # Local variance (used by 6th place)
        'residual', # Standardized residuals (used by 6th place)
        'abs',      # Absolute value (used by 2nd place)
        'zscore',   # Normalized (used by 2nd place)
    ]


def get_priority_transforms() -> list:
    """
    Get the most important transformations for quick implementation.
    
    These 3 transformations were used by ALL top solutions.
    """
    return ['raw', 'cumsum', 'diff']


def compute_transform_features(x0: np.ndarray, x1: np.ndarray,
                                feature_func,
                                transforms=['raw', 'cumsum', 'diff']) -> Dict[str, float]:
    """
    Apply a feature function to multiple transformations.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        feature_func: Function that takes (x0, x1) and returns Dict[str, float]
        transforms: List of transformations to apply
        
    Returns:
        Dictionary with keys like 'feature_raw', 'feature_cumsum', etc.
    """
    all_features = {}
    
    for transform in transforms:
        # Transform the series
        x0_t, x1_t = transform_series(x0, x1, transform)
        
        # Skip if transformation failed
        if len(x0_t) == 0 or len(x1_t) == 0:
            continue
        
        # Compute features on transformed series
        feats = feature_func(x0_t, x1_t)
        
        # Add transform suffix
        for k, v in feats.items():
            all_features[f'{k}_{transform}'] = v
    
    return all_features


if __name__ == '__main__':
    # Test transformations
    import matplotlib.pyplot as plt
    
    # Create test series with a structural break
    np.random.seed(42)
    x0 = np.random.normal(0, 1, 500)
    x1 = np.random.normal(0.5, 0.5, 500)  # Mean shift + variance reduction
    
    transforms = get_priority_transforms()
    
    fig, axes = plt.subplots(len(transforms), 1, figsize=(12, 3*len(transforms)))
    
    for i, transform in enumerate(transforms):
        x0_t, x1_t = transform_series(x0, x1, transform)
        x_combined = np.concatenate([x0_t, x1_t])
        
        axes[i].plot(x_combined)
        axes[i].axvline(len(x0_t), color='red', linestyle='--', label='Break point')
        axes[i].set_title(f'Transform: {transform}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformations_test.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to transformations_test.png")
    
    # Test feature computation
    def dummy_feature(x0, x1):
        return {
            'mean_diff': np.mean(x1) - np.mean(x0),
            'std_ratio': np.std(x1) / (np.std(x0) + 1e-8)
        }
    
    features = compute_transform_features(x0, x1, dummy_feature, transforms)
    print(f"\nComputed {len(features)} features across {len(transforms)} transforms:")
    for k, v in list(features.items())[:6]:
        print(f"  {k}: {v:.4f}")
