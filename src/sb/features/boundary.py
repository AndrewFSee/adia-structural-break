"""
Boundary-localized contrast features.

Compares windows near the boundary between pre-break (x0) and post-break (x1) segments.
These features capture local differences in distribution, trend, and variability at the boundary.

All features are leakage-free: computed per-series from x0 and x1 only.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats as scipy_stats


def safe_window(arr: np.ndarray, n: int, side: str = "tail") -> np.ndarray:
    """
    Extract a window of n points from array.
    
    Args:
        arr: Input array
        n: Window size
        side: "tail" for last n points, "head" for first n points
        
    Returns:
        Window array (full arr if len(arr) < n)
    """
    if len(arr) <= n:
        return arr
    
    if side == "tail":
        return arr[-n:]
    elif side == "head":
        return arr[:n]
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'tail' or 'head'.")


def robust_stats(arr: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """
    Compute robust statistics for array.
    
    Args:
        arr: Input array
        eps: Small constant to avoid division by zero
        
    Returns:
        Dictionary with median, mad, iqr, mean, std, skew, kurtosis
    """
    if len(arr) == 0:
        return {
            'median': np.nan,
            'mad': np.nan,
            'iqr': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'skew': np.nan,
            'kurtosis': np.nan
        }
    
    # Basic stats
    median = np.median(arr)
    mean = np.mean(arr)
    std = np.std(arr) + eps
    
    # Median absolute deviation
    mad = np.median(np.abs(arr - median)) + eps
    
    # Interquartile range
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = (q75 - q25) + eps
    
    # Higher moments (only if enough points)
    if len(arr) >= 3:
        try:
            skew = scipy_stats.skew(arr, nan_policy='omit')
            kurtosis = scipy_stats.kurtosis(arr, nan_policy='omit')
        except:
            skew = np.nan
            kurtosis = np.nan
    else:
        skew = np.nan
        kurtosis = np.nan
    
    return {
        'median': median,
        'mad': mad,
        'iqr': iqr,
        'mean': mean,
        'std': std,
        'skew': skew,
        'kurtosis': kurtosis
    }


def quantiles(arr: np.ndarray, qs: Tuple[float, ...]) -> Dict[str, float]:
    """
    Compute quantiles for array.
    
    Args:
        arr: Input array
        qs: Tuple of quantile levels (0 to 1)
        
    Returns:
        Dictionary mapping quantile label to value
    """
    if len(arr) == 0:
        return {f'q{int(q*100)}': np.nan for q in qs}
    
    result = {}
    for q in qs:
        result[f'q{int(q*100)}'] = np.percentile(arr, q * 100)
    
    return result


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov statistic between two samples.
    
    Args:
        a, b: Input arrays
        
    Returns:
        KS statistic (max distance between empirical CDFs)
    """
    if len(a) == 0 or len(b) == 0:
        return np.nan
    
    try:
        # Use scipy if available
        stat, _ = scipy_stats.ks_2samp(a, b)
        return stat
    except:
        # Manual implementation
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        
        # Combine and sort all values
        all_vals = np.concatenate([a_sorted, b_sorted])
        all_vals = np.unique(all_vals)
        
        # Compute empirical CDFs at each unique value
        max_dist = 0.0
        for val in all_vals:
            cdf_a = np.sum(a_sorted <= val) / len(a_sorted)
            cdf_b = np.sum(b_sorted <= val) / len(b_sorted)
            dist = np.abs(cdf_a - cdf_b)
            max_dist = max(max_dist, dist)
        
        return max_dist


def sign_flip_rate(arr: np.ndarray) -> float:
    """
    Compute rate of sign changes in array.
    
    Args:
        arr: Input array
        
    Returns:
        Fraction of consecutive pairs with sign change
    """
    if len(arr) < 2:
        return np.nan
    
    # Remove zeros to avoid ambiguity
    arr_nonzero = arr[arr != 0]
    
    if len(arr_nonzero) < 2:
        return np.nan
    
    signs = np.sign(arr_nonzero)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    
    return sign_changes / (len(arr_nonzero) - 1)


def fit_slope(arr: np.ndarray) -> float:
    """
    Fit linear trend (slope) to array.
    
    Args:
        arr: Input array
        
    Returns:
        Slope coefficient
    """
    if len(arr) < 3:
        return np.nan
    
    try:
        x = np.arange(len(arr))
        coeffs = np.polyfit(x, arr, deg=1)
        return coeffs[0]  # Slope
    except:
        return np.nan


def compute_boundary_contrast_features(
    x0: np.ndarray,
    x1: np.ndarray,
    windows: Tuple[int, ...] = (25, 50, 100, 250),
    qs: Tuple[float, ...] = (0.1, 0.5, 0.9)
) -> Dict[str, float]:
    """
    Compute boundary-localized contrast features.
    
    Compares windows near the boundary between pre-break (x0) and post-break (x1).
    
    Args:
        x0: Pre-break segment
        x1: Post-break segment
        windows: Window sizes to analyze
        qs: Quantile levels for comparison
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Process each window size
    for w in windows:
        # Extract boundary windows
        a = safe_window(x0, w, side="tail")
        b = safe_window(x1, w, side="head")
        
        # Skip if windows are too small
        if len(a) < 2 or len(b) < 2:
            # Fill with NaN for this window
            features[f'bl_median_delta_w{w}'] = np.nan
            features[f'bl_mean_delta_w{w}'] = np.nan
            features[f'bl_iqr_ratio_w{w}'] = np.nan
            features[f'bl_mad_ratio_w{w}'] = np.nan
            features[f'bl_std_ratio_w{w}'] = np.nan
            features[f'bl_skew_delta_w{w}'] = np.nan
            features[f'bl_kurt_delta_w{w}'] = np.nan
            features[f'bl_ks_stat_w{w}'] = np.nan
            features[f'bl_signflip_rate_delta_w{w}'] = np.nan
            features[f'bl_slope_delta_w{w}'] = np.nan
            
            for q in qs:
                features[f'bl_q{int(q*100)}_delta_w{w}'] = np.nan
            
            continue
        
        # Compute robust stats
        stats_a = robust_stats(a)
        stats_b = robust_stats(b)
        
        # Location deltas
        features[f'bl_median_delta_w{w}'] = stats_b['median'] - stats_a['median']
        features[f'bl_mean_delta_w{w}'] = stats_b['mean'] - stats_a['mean']
        
        # Scale ratios
        features[f'bl_iqr_ratio_w{w}'] = stats_b['iqr'] / stats_a['iqr']
        features[f'bl_mad_ratio_w{w}'] = stats_b['mad'] / stats_a['mad']
        features[f'bl_std_ratio_w{w}'] = stats_b['std'] / stats_a['std']
        
        # Shape deltas
        features[f'bl_skew_delta_w{w}'] = stats_b['skew'] - stats_a['skew']
        features[f'bl_kurt_delta_w{w}'] = stats_b['kurtosis'] - stats_a['kurtosis']
        
        # Quantile deltas
        quants_a = quantiles(a, qs)
        quants_b = quantiles(b, qs)
        
        for q in qs:
            q_key = f'q{int(q*100)}'
            features[f'bl_{q_key}_delta_w{w}'] = quants_b[q_key] - quants_a[q_key]
        
        # Distribution distance
        features[f'bl_ks_stat_w{w}'] = ks_statistic(a, b)
        
        # Sign flip rate delta
        signflip_a = sign_flip_rate(a)
        signflip_b = sign_flip_rate(b)
        features[f'bl_signflip_rate_delta_w{w}'] = signflip_b - signflip_a
        
        # Slope delta
        slope_a = fit_slope(a)
        slope_b = fit_slope(b)
        features[f'bl_slope_delta_w{w}'] = slope_b - slope_a
    
    # Also compute for full segments (suffix _full)
    if len(x0) >= 2 and len(x1) >= 2:
        stats_x0 = robust_stats(x0)
        stats_x1 = robust_stats(x1)
        
        features['bl_median_delta_full'] = stats_x1['median'] - stats_x0['median']
        features['bl_mean_delta_full'] = stats_x1['mean'] - stats_x0['mean']
        features['bl_iqr_ratio_full'] = stats_x1['iqr'] / stats_x0['iqr']
        features['bl_mad_ratio_full'] = stats_x1['mad'] / stats_x0['mad']
        features['bl_std_ratio_full'] = stats_x1['std'] / stats_x0['std']
        features['bl_skew_delta_full'] = stats_x1['skew'] - stats_x0['skew']
        features['bl_kurt_delta_full'] = stats_x1['kurtosis'] - stats_x0['kurtosis']
        features['bl_ks_stat_full'] = ks_statistic(x0, x1)
        
        signflip_x0 = sign_flip_rate(x0)
        signflip_x1 = sign_flip_rate(x1)
        features['bl_signflip_rate_delta_full'] = signflip_x1 - signflip_x0
        
        slope_x0 = fit_slope(x0)
        slope_x1 = fit_slope(x1)
        features['bl_slope_delta_full'] = slope_x1 - slope_x0
        
        quants_x0 = quantiles(x0, qs)
        quants_x1 = quantiles(x1, qs)
        
        for q in qs:
            q_key = f'q{int(q*100)}'
            features[f'bl_{q_key}_delta_full'] = quants_x1[q_key] - quants_x0[q_key]
    else:
        # Fill with NaN
        features['bl_median_delta_full'] = np.nan
        features['bl_mean_delta_full'] = np.nan
        features['bl_iqr_ratio_full'] = np.nan
        features['bl_mad_ratio_full'] = np.nan
        features['bl_std_ratio_full'] = np.nan
        features['bl_skew_delta_full'] = np.nan
        features['bl_kurt_delta_full'] = np.nan
        features['bl_ks_stat_full'] = np.nan
        features['bl_signflip_rate_delta_full'] = np.nan
        features['bl_slope_delta_full'] = np.nan
        
        for q in qs:
            features[f'bl_q{int(q*100)}_delta_full'] = np.nan
    
    return features


if __name__ == "__main__":
    # Simple self-check
    print("=" * 70)
    print("BOUNDARY FEATURES SELF-CHECK")
    print("=" * 70)
    
    # Create toy data
    np.random.seed(42)
    x0 = np.random.randn(100) + 0.0  # Mean 0
    x1 = np.random.randn(100) + 0.5  # Mean 0.5 (shifted)
    
    # Compute features
    features = compute_boundary_contrast_features(x0, x1, windows=(25, 50), qs=(0.1, 0.5, 0.9))
    
    print(f"\nComputed {len(features)} boundary features")
    print("\nSample features:")
    print(f"  bl_mean_delta_w25:  {features['bl_mean_delta_w25']:.4f}")
    print(f"  bl_median_delta_w50: {features['bl_median_delta_w50']:.4f}")
    print(f"  bl_ks_stat_w25:     {features['bl_ks_stat_w25']:.4f}")
    print(f"  bl_slope_delta_full: {features['bl_slope_delta_full']:.4f}")
    
    print("\nAll feature keys:")
    for i, key in enumerate(sorted(features.keys()), 1):
        print(f"  {i:2d}. {key}")
    
    print("\n✓ Self-check complete")
