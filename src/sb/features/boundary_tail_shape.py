"""
Boundary-localized tail-shape change features.

Focuses on tail-shape characteristics (heaviness, curvature, asymmetry) rather than
distribution distances. All features are leakage-free: computed per-series from x0 and x1 only.

This module complements boundary_dist.py (which computes Wasserstein/Energy distances)
by characterizing how the tail shape itself changes near the structural break.

Feature families:
- Hill tail index (upper/lower, q90/q95)
- Quantile spacing ratios and curvature
- Tail asymmetry metrics
- Exceedance probabilities and mean excess
- Localization: window vs full segment
- DoD: statistics across window scales
"""

import numpy as np
from typing import Dict, Tuple


def extract_boundary_segments(x0: np.ndarray, x1: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract boundary windows from segments.
    
    Args:
        x0: Pre-break segment
        x1: Post-break segment
        w: Window size
        
    Returns:
        (x0_boundary, x1_boundary): Last w points of x0, first w points of x1
    """
    # Remove NaNs first
    x0_clean = x0[~np.isnan(x0)]
    x1_clean = x1[~np.isnan(x1)]
    
    # Extract windows
    if len(x0_clean) <= w:
        x0b = x0_clean
    else:
        x0b = x0_clean[-w:]
    
    if len(x1_clean) <= w:
        x1b = x1_clean
    else:
        x1b = x1_clean[:w]
    
    return x0b, x1b


def robust_scale(arr: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute robust scale estimate using MAD.
    
    Args:
        arr: Input array
        eps: Minimum scale to avoid division by zero
        
    Returns:
        Robust scale (MAD * 1.4826)
    """
    if len(arr) == 0:
        return 1.0
    
    # Remove NaNs
    arr_clean = arr[~np.isnan(arr)]
    
    if len(arr_clean) == 0:
        return 1.0
    
    # Median absolute deviation
    median = np.median(arr_clean)
    mad = np.median(np.abs(arr_clean - median))
    
    # Convert MAD to standard deviation estimate
    scale = mad * 1.4826
    
    # Ensure minimum scale
    if scale < eps:
        scale = 1.0
    
    return scale


def robust_zscore(arr: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float, float]:
    """
    Compute robust z-scores using median and MAD.
    
    Args:
        arr: Input array (already cleaned of NaNs)
        eps: Minimum scale to avoid division by zero
        
    Returns:
        (z_scores, median, scale): Standardized values, median, and robust scale
    """
    if len(arr) == 0:
        return np.array([]), 0.0, 1.0
    
    median = np.median(arr)
    scale = robust_scale(arr, eps=eps)
    
    z = (arr - median) / scale
    
    return z, median, scale


def topk_tail(z: np.ndarray, frac: float = 0.10, min_k: int = 8) -> np.ndarray:
    """
    Extract top-k largest values (upper tail) adaptively.
    
    Args:
        z: Standardized array
        frac: Fraction of samples to select
        min_k: Minimum number of samples required
        
    Returns:
        Top-k values, or empty array if insufficient data
    """
    if len(z) < min_k * 2:
        return np.array([])
    
    k = max(min_k, int(len(z) * frac))
    k = min(k, len(z) - 1)  # Ensure we don't select all
    
    if k < min_k:
        return np.array([])
    
    # Use partition for O(n) selection
    threshold = np.partition(z, -k)[-k]
    tail = z[z >= threshold]
    
    return tail


def bottomk_tail(z: np.ndarray, frac: float = 0.10, min_k: int = 8) -> np.ndarray:
    """
    Extract bottom-k smallest values (lower tail) adaptively.
    
    Args:
        z: Standardized array
        frac: Fraction of samples to select
        min_k: Minimum number of samples required
        
    Returns:
        Bottom-k values, or empty array if insufficient data
    """
    if len(z) < min_k * 2:
        return np.array([])
    
    k = max(min_k, int(len(z) * frac))
    k = min(k, len(z) - 1)
    
    if k < min_k:
        return np.array([])
    
    # Use partition for O(n) selection
    threshold = np.partition(z, k)[k]
    tail = z[z <= threshold]
    
    return tail


def hill_estimator_robust(z: np.ndarray, frac: float = 0.10, min_k: int = 8, eps: float = 1e-12) -> float:
    """
    Compute Hill estimator for tail index on standardized residuals.
    
    More robust version using adaptive tail selection and positive shift.
    Lower alpha => heavier tail (more extreme values).
    
    Args:
        z: Standardized residuals (z-scores)
        frac: Fraction of tail to use
        min_k: Minimum number of tail samples required
        eps: Small constant for numerical stability
        
    Returns:
        Hill alpha estimate, or NaN if insufficient data
    """
    # Extract upper tail
    tail_samples = topk_tail(z, frac=frac, min_k=min_k)
    
    if len(tail_samples) < min_k:
        return np.nan
    
    # Shift to positive: u_i = x_i - min(x) + 1
    x = tail_samples - tail_samples.min() + 1.0
    
    if x.min() <= eps:
        return np.nan
    
    # Hill estimator: alpha = 1 / mean(log(x_i / x_min))
    x_min = x.min()
    log_ratios = np.log(x / x_min + eps)
    mean_log_ratio = np.mean(log_ratios)
    
    if mean_log_ratio < eps:
        return np.nan
    
    alpha = 1.0 / mean_log_ratio
    
    return alpha


def _collect_window_series_ts(
    features: Dict[str, float],
    key_pattern: str,
    windows: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect valid (finite) feature values across windows for tail-shape features.
    
    Args:
        features: Feature dictionary
        key_pattern: Pattern like 'bl_ts_hill_alpha_delta_q95_w{w}'
        windows: Window sizes to check
        
    Returns:
        (logw, d): Arrays of log(window) and feature values for valid windows,
                   sorted by logw ascending
    """
    logw_list = []
    d_list = []
    
    for w in windows:
        key = key_pattern.format(w=w)
        if key in features:
            val = features[key]
            if np.isfinite(val):
                logw_list.append(np.log(w))
                d_list.append(val)
    
    # Convert to arrays
    logw = np.array(logw_list)
    d = np.array(d_list)
    
    # Return empty arrays if no valid points
    if len(logw) == 0:
        return logw, d
    
    # Sort by logw ascending to ensure consistent ordering
    idx = np.argsort(logw)
    return logw[idx], d[idx]


def _dod_stats_ts(logw: np.ndarray, d: np.ndarray, prefix: str, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compute distance-of-distance statistics across windows for tail-shape features.
    
    Expects logw and d to be sorted by logw ascending. Defensively sorts
    to ensure correct delta_wmin_wmax, ratio_wmin_wmax, and curvature.
    
    Args:
        logw: Log of window sizes
        d: Feature values
        prefix: Feature name prefix
        eps: Small constant for numerical stability
        
    Returns:
        Dictionary of DoD features
    """
    features = {}
    
    # Initialize all features with NaN
    features[f'{prefix}_slope_logw'] = np.nan
    features[f'{prefix}_intercept_logw'] = np.nan
    features[f'{prefix}_range'] = np.nan
    features[f'{prefix}_std'] = np.nan
    features[f'{prefix}_cv'] = np.nan
    features[f'{prefix}_delta_wmin_wmax'] = np.nan
    features[f'{prefix}_ratio_wmin_wmax'] = np.nan
    features[f'{prefix}_curv_wA_wB_wC'] = np.nan
    
    if len(d) < 2:
        return features
    
    # Defensive sort to ensure correct ordering
    idx = np.argsort(logw)
    logw = logw[idx]
    d = d[idx]
    
    # Slope and intercept (least squares on d ~ logw)
    try:
        coeffs = np.polyfit(logw, d, deg=1)
        features[f'{prefix}_slope_logw'] = coeffs[0]
        features[f'{prefix}_intercept_logw'] = coeffs[1]
    except:
        pass
    
    # Range, std, cv
    features[f'{prefix}_range'] = np.max(d) - np.min(d)
    features[f'{prefix}_std'] = np.std(d)
    
    mean_abs = np.mean(np.abs(d))
    if mean_abs > eps:
        features[f'{prefix}_cv'] = np.std(d) / mean_abs
    
    # Delta and ratio using smallest and largest windows
    features[f'{prefix}_delta_wmin_wmax'] = d[-1] - d[0]
    
    if abs(d[0]) > eps:
        features[f'{prefix}_ratio_wmin_wmax'] = d[-1] / d[0]
    
    # Curvature using first 3 windows
    if len(d) >= 3:
        features[f'{prefix}_curv_wA_wB_wC'] = d[0] - 2 * d[1] + d[2]
    
    return features


def compute_scope_tail_shape_features(
    x0_scope: np.ndarray,
    x1_scope: np.ndarray,
    scope_name: str,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Compute tail-shape features for a given scope (window or full segment).
    
    Uses robust standardization per-segment, pooled-threshold exceedance metrics,
    and adaptive tail selection for stable tail-shape characterization.
    
    Args:
        x0_scope: Pre-break segment (boundary window or full)
        x1_scope: Post-break segment (boundary window or full)
        scope_name: "w{w}" or "full"
        eps: Small constant for numerical stability
        
    Returns:
        Dictionary of tail-shape features
    """
    features = {}
    
    # Clean inputs
    x0_clean = x0_scope[~np.isnan(x0_scope)]
    x1_clean = x1_scope[~np.isnan(x1_scope)]
    
    # Initialize all feature keys with NaN (using bl_ts_ prefix for tail-shape)
    for q in [90, 95]:
        features[f'bl_ts_hill_alpha0_q{q}_{scope_name}'] = np.nan
        features[f'bl_ts_hill_alpha1_q{q}_{scope_name}'] = np.nan
        features[f'bl_ts_hill_alpha_delta_q{q}_{scope_name}'] = np.nan
        features[f'bl_ts_hill_alpha_absdelta_q{q}_{scope_name}'] = np.nan
        features[f'bl_ts_hill_alpha_ndiff_q{q}_{scope_name}'] = np.nan
        features[f'bl_ts_hill_alpha_delta_q{q}_lower_{scope_name}'] = np.nan
        features[f'bl_ts_hill_alpha_absdelta_q{q}_lower_{scope_name}'] = np.nan
        features[f'bl_ts_hill_alpha_ndiff_q{q}_lower_{scope_name}'] = np.nan
    
    for t_str in ['t20', 't25', 't30']:
        features[f'bl_ts_p_hi_delta_{t_str}_{scope_name}'] = np.nan
        features[f'bl_ts_p_lo_delta_{t_str}_{scope_name}'] = np.nan
        features[f'bl_ts_mean_excess_hi_delta_{t_str}_{scope_name}'] = np.nan
        features[f'bl_ts_mean_excess_lo_delta_{t_str}_{scope_name}'] = np.nan
    
    features[f'bl_ts_qspace_hi_delta_{scope_name}'] = np.nan
    features[f'bl_ts_qspace_hi_absdelta_{scope_name}'] = np.nan
    features[f'bl_ts_qspace_hi_ndiff_{scope_name}'] = np.nan
    features[f'bl_ts_qspace_lo_delta_{scope_name}'] = np.nan
    features[f'bl_ts_qspace_lo_absdelta_{scope_name}'] = np.nan
    features[f'bl_ts_qspace_lo_ndiff_{scope_name}'] = np.nan
    features[f'bl_ts_asym_delta_{scope_name}'] = np.nan
    features[f'bl_ts_asym_absdelta_{scope_name}'] = np.nan
    features[f'bl_ts_asym_ndiff_{scope_name}'] = np.nan
    
    if len(x0_clean) < 10 or len(x1_clean) < 10:
        return features
    
    # Robust standardization per-segment
    z0, med0, scale0 = robust_zscore(x0_clean, eps=eps)
    z1, med1, scale1 = robust_zscore(x1_clean, eps=eps)
    
    # A) Hill estimator for tail heaviness (on standardized residuals)
    # Upper tail (q90 and q95 correspond to frac=0.10 and frac=0.05)
    alpha0_q90 = hill_estimator_robust(z0, frac=0.10, min_k=8)
    alpha1_q90 = hill_estimator_robust(z1, frac=0.10, min_k=8)
    alpha0_q95 = hill_estimator_robust(z0, frac=0.05, min_k=8)
    alpha1_q95 = hill_estimator_robust(z1, frac=0.05, min_k=8)
    
    features[f'bl_ts_hill_alpha0_q90_{scope_name}'] = alpha0_q90
    features[f'bl_ts_hill_alpha1_q90_{scope_name}'] = alpha1_q90
    if np.isfinite(alpha0_q90) and np.isfinite(alpha1_q90):
        delta = alpha1_q90 - alpha0_q90
        features[f'bl_ts_hill_alpha_delta_q90_{scope_name}'] = delta
        features[f'bl_ts_hill_alpha_absdelta_q90_{scope_name}'] = abs(delta)
        features[f'bl_ts_hill_alpha_ndiff_q90_{scope_name}'] = delta / (abs(alpha0_q90) + abs(alpha1_q90) + eps)
    
    features[f'bl_ts_hill_alpha0_q95_{scope_name}'] = alpha0_q95
    features[f'bl_ts_hill_alpha1_q95_{scope_name}'] = alpha1_q95
    if np.isfinite(alpha0_q95) and np.isfinite(alpha1_q95):
        delta = alpha1_q95 - alpha0_q95
        features[f'bl_ts_hill_alpha_delta_q95_{scope_name}'] = delta
        features[f'bl_ts_hill_alpha_absdelta_q95_{scope_name}'] = abs(delta)
        features[f'bl_ts_hill_alpha_ndiff_q95_{scope_name}'] = delta / (abs(alpha0_q95) + abs(alpha1_q95) + eps)
    
    # Lower tail (apply to -z)
    alpha0_q90_lower = hill_estimator_robust(-z0, frac=0.10, min_k=8)
    alpha1_q90_lower = hill_estimator_robust(-z1, frac=0.10, min_k=8)
    alpha0_q95_lower = hill_estimator_robust(-z0, frac=0.05, min_k=8)
    alpha1_q95_lower = hill_estimator_robust(-z1, frac=0.05, min_k=8)
    
    if np.isfinite(alpha0_q90_lower) and np.isfinite(alpha1_q90_lower):
        delta = alpha1_q90_lower - alpha0_q90_lower
        features[f'bl_ts_hill_alpha_delta_q90_lower_{scope_name}'] = delta
        features[f'bl_ts_hill_alpha_absdelta_q90_lower_{scope_name}'] = abs(delta)
        features[f'bl_ts_hill_alpha_ndiff_q90_lower_{scope_name}'] = delta / (abs(alpha0_q90_lower) + abs(alpha1_q90_lower) + eps)
    
    if np.isfinite(alpha0_q95_lower) and np.isfinite(alpha1_q95_lower):
        delta = alpha1_q95_lower - alpha0_q95_lower
        features[f'bl_ts_hill_alpha_delta_q95_lower_{scope_name}'] = delta
        features[f'bl_ts_hill_alpha_absdelta_q95_lower_{scope_name}'] = abs(delta)
        features[f'bl_ts_hill_alpha_ndiff_q95_lower_{scope_name}'] = delta / (abs(alpha0_q95_lower) + abs(alpha1_q95_lower) + eps)
    
    # B) Pooled threshold exceedance metrics (on standardized residuals)
    pooled_z = np.concatenate([z0, z1])
    
    for t_val, t_str in [(2.0, 't20'), (2.5, 't25'), (3.0, 't30')]:
        # Upper tail exceedances
        p0_hi = np.mean(z0 >= t_val)
        p1_hi = np.mean(z1 >= t_val)
        features[f'bl_ts_p_hi_delta_{t_str}_{scope_name}'] = p1_hi - p0_hi
        
        # Lower tail exceedances
        p0_lo = np.mean(z0 <= -t_val)
        p1_lo = np.mean(z1 <= -t_val)
        features[f'bl_ts_p_lo_delta_{t_str}_{scope_name}'] = p1_lo - p0_lo
        
        # Mean excess (how far beyond threshold)
        z0_exceed_hi = z0[z0 >= t_val] - t_val
        z1_exceed_hi = z1[z1 >= t_val] - t_val
        
        mean_excess_0_hi = np.mean(z0_exceed_hi) if len(z0_exceed_hi) > 0 else 0.0
        mean_excess_1_hi = np.mean(z1_exceed_hi) if len(z1_exceed_hi) > 0 else 0.0
        features[f'bl_ts_mean_excess_hi_delta_{t_str}_{scope_name}'] = mean_excess_1_hi - mean_excess_0_hi
        
        z0_exceed_lo = -t_val - z0[z0 <= -t_val]  # Excess magnitude
        z1_exceed_lo = -t_val - z1[z1 <= -t_val]
        
        mean_excess_0_lo = np.mean(z0_exceed_lo) if len(z0_exceed_lo) > 0 else 0.0
        mean_excess_1_lo = np.mean(z1_exceed_lo) if len(z1_exceed_lo) > 0 else 0.0
        features[f'bl_ts_mean_excess_lo_delta_{t_str}_{scope_name}'] = mean_excess_1_lo - mean_excess_0_lo
    
    # C) Quantile spacing ratios (on standardized residuals)
    try:
        # Get key quantiles
        q0_01, q0_05, q0_50, q0_95, q0_99 = np.quantile(z0, [0.01, 0.05, 0.50, 0.95, 0.99])
        q1_01, q1_05, q1_50, q1_95, q1_99 = np.quantile(z1, [0.01, 0.05, 0.50, 0.95, 0.99])
        
        # Upper tail spacing: (q99-q95)/(q95-q50)
        qspace_hi_0 = (q0_99 - q0_95) / (q0_95 - q0_50 + eps)
        qspace_hi_1 = (q1_99 - q1_95) / (q1_95 - q1_50 + eps)
        delta_hi = qspace_hi_1 - qspace_hi_0
        features[f'bl_ts_qspace_hi_delta_{scope_name}'] = delta_hi
        features[f'bl_ts_qspace_hi_absdelta_{scope_name}'] = abs(delta_hi)
        features[f'bl_ts_qspace_hi_ndiff_{scope_name}'] = delta_hi / (abs(qspace_hi_0) + abs(qspace_hi_1) + eps)
        
        # Lower tail spacing: (q50-q05)/(q05-q01)
        qspace_lo_0 = (q0_50 - q0_05) / (q0_05 - q0_01 + eps)
        qspace_lo_1 = (q1_50 - q1_05) / (q1_05 - q1_01 + eps)
        delta_lo = qspace_lo_1 - qspace_lo_0
        features[f'bl_ts_qspace_lo_delta_{scope_name}'] = delta_lo
        features[f'bl_ts_qspace_lo_absdelta_{scope_name}'] = abs(delta_lo)
        features[f'bl_ts_qspace_lo_ndiff_{scope_name}'] = delta_lo / (abs(qspace_lo_0) + abs(qspace_lo_1) + eps)
    except:
        pass  # Already initialized to NaN
    
    # D) Tail asymmetry: (q99-q50) - (q50-q01)
    try:
        q0_01, q0_50, q0_99 = np.quantile(z0, [0.01, 0.50, 0.99])
        q1_01, q1_50, q1_99 = np.quantile(z1, [0.01, 0.50, 0.99])
        
        asym0 = (q0_99 - q0_50) - (q0_50 - q0_01)
        asym1 = (q1_99 - q1_50) - (q1_50 - q1_01)
        delta_asym = asym1 - asym0
        
        features[f'bl_ts_asym_delta_{scope_name}'] = delta_asym
        features[f'bl_ts_asym_absdelta_{scope_name}'] = abs(delta_asym)
        features[f'bl_ts_asym_ndiff_{scope_name}'] = delta_asym / (abs(asym0) + abs(asym1) + eps)
    except:
        pass  # Already initialized to NaN
    
    return features


def compute_localization_ts_features(base: Dict[str, float], w: int, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compute localization features for tail-shape: boundary window vs full segment.
    
    Args:
        base: Feature dictionary with pre-computed tail-shape features
        w: Window size
        eps: Small constant for numerical stability
        
    Returns:
        Dictionary of localization features
    """
    loc_features = {}
    
    # Select key tail-shape metrics for localization
    metrics = [
        ('hill_alpha_delta_q95', f'bl_ts_hill_alpha_delta_q95_w{w}', 'bl_ts_hill_alpha_delta_q95_full'),
        ('qspace_hi_delta', f'bl_ts_qspace_hi_delta_w{w}', 'bl_ts_qspace_hi_delta_full'),
        ('qspace_lo_delta', f'bl_ts_qspace_lo_delta_w{w}', 'bl_ts_qspace_lo_delta_full'),
        ('asym_delta', f'bl_ts_asym_delta_w{w}', 'bl_ts_asym_delta_full')
    ]
    
    for metric_name, key_w, key_full in metrics:
        # Get values from base dictionary
        val_w = base.get(key_w, np.nan)
        val_full = base.get(key_full, np.nan)
        
        # Check if both are finite
        if np.isfinite(val_w) and np.isfinite(val_full):
            # 1) Signed difference
            loc_features[f'bl_ts_loc_diff_{metric_name}_w{w}'] = val_w - val_full
            
            # 2) Absolute difference
            loc_features[f'bl_ts_loc_absdiff_{metric_name}_w{w}'] = abs(val_w - val_full)
            
            # 3) Normalized difference
            loc_features[f'bl_ts_loc_ndiff_{metric_name}_w{w}'] = (val_w - val_full) / (abs(val_w) + abs(val_full) + eps)
            
            # 4) Ratio
            loc_features[f'bl_ts_loc_ratio_{metric_name}_w{w}'] = val_w / (val_full + eps)
        else:
            # If either is NaN, set all to NaN
            loc_features[f'bl_ts_loc_diff_{metric_name}_w{w}'] = np.nan
            loc_features[f'bl_ts_loc_absdiff_{metric_name}_w{w}'] = np.nan
            loc_features[f'bl_ts_loc_ndiff_{metric_name}_w{w}'] = np.nan
            loc_features[f'bl_ts_loc_ratio_{metric_name}_w{w}'] = np.nan
    
    return loc_features


def compute_boundary_tail_shape_features(
    x0: np.ndarray,
    x1: np.ndarray,
    windows: Tuple[int, ...] = (25, 50, 100, 250)
) -> Dict[str, float]:
    """
    Compute boundary-localized tail-shape change features.
    
    This function characterizes how the tail shape changes near the structural break,
    including tail heaviness (Hill estimator), quantile spacing ratios, asymmetry,
    localization (window vs full), and DoD statistics across windows.
    
    Args:
        x0: Pre-break segment
        x1: Post-break segment
        windows: Window sizes to analyze
        
    Returns:
        Dictionary of features
        
    Feature families (per scope):
        - bl_ts_hill_alpha*: Hill estimator for tail index (upper/lower, q90/q95)
        - bl_ts_p_hi/lo_delta_*: Exceedance probabilities
        - bl_ts_mean_excess_*: Mean magnitude of exceedances
        - bl_ts_qspace_*: Quantile spacing ratios (upper/lower)
        - bl_ts_asym_delta: Tail asymmetry shift
        - bl_ts_loc_*: Localization (window vs full)
        - bl_ts_dod_*: DoD statistics across windows
    """
    features = {}
    
    # Remove NaNs from full segments
    x0_full = x0[~np.isnan(x0)]
    x1_full = x1[~np.isnan(x1)]
    
    # Process each window size
    for w in windows:
        # Extract boundary windows
        x0b, x1b = extract_boundary_segments(x0, x1, w)
        
        # Compute features for this window
        window_feats = compute_scope_tail_shape_features(x0b, x1b, f"w{w}")
        features.update(window_feats)
    
    # Compute for full segments
    full_feats = compute_scope_tail_shape_features(x0_full, x1_full, "full")
    features.update(full_feats)
    
    # ===================================================================
    # LOCALIZATION FEATURES (window vs full)
    # ===================================================================
    for w in windows:
        loc_feats = compute_localization_ts_features(features, w)
        features.update(loc_feats)
    
    # ===================================================================
    # DoD FEATURES (statistics across windows)
    # ===================================================================
    eps = 1e-12
    
    # Select key metrics for DoD analysis
    dod_metrics = [
        ('bl_ts_hill_alpha_delta_q95_w{w}', 'bl_ts_dod_hill_alpha_delta_q95'),
        ('bl_ts_qspace_hi_delta_w{w}', 'bl_ts_dod_qspace_hi_delta'),
        ('bl_ts_qspace_lo_delta_w{w}', 'bl_ts_dod_qspace_lo_delta'),
        ('bl_ts_asym_delta_w{w}', 'bl_ts_dod_asym_delta')
    ]
    
    for key_pattern, prefix in dod_metrics:
        logw, d = _collect_window_series_ts(features, key_pattern, windows)
        dod_feats = _dod_stats_ts(logw, d, prefix, eps=eps)
        features.update(dod_feats)
    
    return features


if __name__ == "__main__":
    """Self-check: verify tail-shape features behave as expected."""
    
    print("=" * 70)
    print("BOUNDARY TAIL-SHAPE FEATURES - SELF-CHECK")
    print("=" * 70)
    
    # Test 1: Normal vs heavy-tail regime
    print("\nTest 1: Normal vs heavy-tail regime (x1 has spikes)")
    np.random.seed(42)
    x0_norm = np.random.randn(200)
    x1_tail = np.random.randn(200)
    
    # Add large positive spikes to x1 (makes tail heavier)
    spike_indices = np.random.choice(len(x1_tail), size=20, replace=False)
    x1_tail[spike_indices] += np.random.uniform(4, 8, size=20)
    
    features_tail = compute_boundary_tail_shape_features(x0_norm, x1_tail, windows=(50,))
    
    print(f"  bl_ts_hill_alpha0_q95_w50: {features_tail.get('bl_ts_hill_alpha0_q95_w50', np.nan):.4f}")
    print(f"  bl_ts_hill_alpha1_q95_w50: {features_tail.get('bl_ts_hill_alpha1_q95_w50', np.nan):.4f}")
    print(f"  bl_ts_hill_alpha_delta_q95_w50: {features_tail.get('bl_ts_hill_alpha_delta_q95_w50', np.nan):.4f}")
    print(f"  bl_ts_p_hi_delta_t25_w50: {features_tail.get('bl_ts_p_hi_delta_t25_w50', np.nan):.4f}")
    print(f"  bl_ts_p_hi_delta_t30_w50: {features_tail.get('bl_ts_p_hi_delta_t30_w50', np.nan):.4f}")
    print(f"  bl_ts_mean_excess_hi_delta_t25_w50: {features_tail.get('bl_ts_mean_excess_hi_delta_t25_w50', np.nan):.4f}")
    print(f"  bl_ts_qspace_hi_delta_w50: {features_tail.get('bl_ts_qspace_hi_delta_w50', np.nan):.4f}")
    print(f"  bl_ts_asym_delta_w50: {features_tail.get('bl_ts_asym_delta_w50', np.nan):.4f}")
    
    # Check localization features exist
    print(f"  bl_ts_loc_diff_hill_alpha_delta_q95_w50: {features_tail.get('bl_ts_loc_diff_hill_alpha_delta_q95_w50', np.nan):.4f}")
    
    # Check DoD features exist
    print(f"  bl_ts_dod_hill_alpha_delta_q95_slope_logw: {features_tail.get('bl_ts_dod_hill_alpha_delta_q95_slope_logw', np.nan):.4f}")
    
    # Assertions: heavy-tail regime should show increased exceedances and mean excess
    assert features_tail['bl_ts_p_hi_delta_t25_w50'] > 0, "Should have more exceedances at t=2.5 in heavy-tail regime"
    assert features_tail['bl_ts_mean_excess_hi_delta_t25_w50'] > 0, "Should have larger mean excess in heavy-tail regime"
    print("  ✓ Exceedance probability increased (heavy-tail detected)")
    print("  ✓ Mean excess increased (heavy-tail detected)")
    
    # Test 2: Pure location shift (x1 = x0 + constant)
    print("\nTest 2: Pure location shift (x1 = x0 + 1)")
    np.random.seed(123)
    x0_base = np.random.randn(200)
    x1_shifted = x0_base + 1.0  # Location shift only, no shape change
    
    features_shift = compute_boundary_tail_shape_features(x0_base, x1_shifted, windows=(50,))
    
    print(f"  bl_ts_hill_alpha_delta_q95_w50: {features_shift.get('bl_ts_hill_alpha_delta_q95_w50', np.nan):.4f}")
    print(f"  bl_ts_p_hi_delta_t25_w50: {features_shift.get('bl_ts_p_hi_delta_t25_w50', np.nan):.4f}")
    print(f"  bl_ts_mean_excess_hi_delta_t25_w50: {features_shift.get('bl_ts_mean_excess_hi_delta_t25_w50', np.nan):.4f}")
    print(f"  bl_ts_qspace_hi_delta_w50: {features_shift.get('bl_ts_qspace_hi_delta_w50', np.nan):.4f}")
    print(f"  bl_ts_asym_delta_w50: {features_shift.get('bl_ts_asym_delta_w50', np.nan):.4f}")
    
    # Standardized exceedance features should be near 0 under pure location shift
    # (since we compute on z-scores which remove location)
    assert abs(features_shift['bl_ts_p_hi_delta_t25_w50']) < 0.15, "Exceedance rate should be stable under location shift (standardized)"
    assert abs(features_shift['bl_ts_mean_excess_hi_delta_t25_w50']) < 0.5, "Mean excess should be stable under location shift (standardized)"
    assert abs(features_shift['bl_ts_qspace_hi_delta_w50']) < 1.0, "Qspace ratio should be stable under location shift (standardized)"
    print("  ✓ Standardized exceedance features stable under location shift")
    print("  ✓ Tail shape metrics stable under location shift")
    
    # Test 3: DoD with shuffled windows (test ordering robustness)
    print("\nTest 3: DoD features with shuffled window order")
    np.random.seed(789)
    x0_dod = np.random.randn(300)
    x1_dod = np.random.randn(300) + 1.0
    
    # Test with different window orders
    features_sorted = compute_boundary_tail_shape_features(x0_dod, x1_dod, windows=(25, 50, 100, 250))
    features_shuffled = compute_boundary_tail_shape_features(x0_dod, x1_dod, windows=(250, 25, 100, 50))
    
    # Verify DoD features are identical
    dod_key = 'bl_ts_dod_hill_alpha_delta_q95_delta_wmin_wmax'
    if dod_key in features_sorted and dod_key in features_shuffled:
        val_sorted = features_sorted[dod_key]
        val_shuffled = features_shuffled[dod_key]
        if np.isfinite(val_sorted) and np.isfinite(val_shuffled):
            assert np.isclose(val_sorted, val_shuffled, atol=1e-8), \
                f"DoD features should be identical: {val_sorted:.6f} vs {val_shuffled:.6f}"
            print(f"  {dod_key}: {val_sorted:.6f} (consistent across orderings)")
            print("  ✓ DoD features are window-order invariant")
    
    # Count features
    n_features = len(features_tail)
    n_loc_features = len([k for k in features_tail.keys() if 'loc_' in k])
    n_dod_features = len([k for k in features_tail.keys() if 'dod_' in k])
    print(f"\nFeature counts:")
    print(f"  Total: {n_features}")
    print(f"  Localization: {n_loc_features}")
    print(f"  DoD: {n_dod_features}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
