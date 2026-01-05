"""
CUSUM (Cumulative Sum) based features for structural break detection.

CUSUM is highly effective for detecting level shifts and characterizing
the shape of transitions. Used by multiple top solutions.

Key features from 6th place solution:
- Elbow detection (where does CUSUM change direction?)
- Shape categories (up-flat, down-flat, trend-up, etc.)
- Curvature/sharpness at boundary
- Wasserstein distance of CUSUM residuals
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats


def compute_cusum(x: np.ndarray, normalize=True) -> np.ndarray:
    """
    Compute cumulative sum of deviations from mean.
    
    Args:
        x: Input series
        normalize: Whether to normalize by std
        
    Returns:
        CUSUM series
    """
    if len(x) == 0:
        return np.array([])
    
    # Compute deviations from mean
    deviations = x - np.mean(x)
    
    # Cumulative sum
    cusum = np.cumsum(deviations)
    
    # Normalize by std
    if normalize and np.std(cusum) > 1e-10:
        cusum = cusum / np.std(cusum)
    
    return cusum


def detect_cusum_elbow(cusum_pre: np.ndarray, cusum_post: np.ndarray, 
                       window=10) -> Dict[str, float]:
    """
    Detect elbow pattern in CUSUM around the boundary.
    
    Args:
        cusum_pre: CUSUM values before break
        cusum_post: CUSUM values after break
        window: Window size around boundary to analyze
        
    Returns:
        Dictionary with elbow features
    """
    features = {}
    
    if len(cusum_pre) == 0 or len(cusum_post) == 0:
        return {'elbow_sharpness': 0.0, 'elbow_category': 0}
    
    # Get values near boundary
    boundary_pre = cusum_pre[-window:] if len(cusum_pre) >= window else cusum_pre
    boundary_post = cusum_post[:window] if len(cusum_post) >= window else cusum_post
    
    # Compute slopes
    if len(boundary_pre) > 1:
        slope_pre = (boundary_pre[-1] - boundary_pre[0]) / len(boundary_pre)
    else:
        slope_pre = 0.0
    
    if len(boundary_post) > 1:
        slope_post = (boundary_post[-1] - boundary_post[0]) / len(boundary_post)
    else:
        slope_post = 0.0
    
    # Elbow sharpness (change in slope)
    features['elbow_sharpness'] = np.abs(slope_post - slope_pre)
    
    # Curvature (second derivative approximation)
    if len(cusum_pre) > 2 and len(cusum_post) > 2:
        # Use last 3 points of pre and first 3 of post
        pre_last = cusum_pre[-3:]
        post_first = cusum_post[:3]
        combined = np.concatenate([pre_last, post_first])
        
        if len(combined) > 2:
            # Second differences
            first_diff = np.diff(combined)
            second_diff = np.diff(first_diff)
            # Curvature at boundary (middle of second_diff)
            if len(second_diff) > 0:
                features['elbow_curvature'] = np.abs(second_diff[len(second_diff)//2])
            else:
                features['elbow_curvature'] = 0.0
        else:
            features['elbow_curvature'] = 0.0
    else:
        features['elbow_curvature'] = 0.0
    
    # Categorize elbow shape
    # Categories: 0=flat, 1=up-flat, 2=down-flat, 3=trend-up, 4=trend-down, 5=V-shape, 6=inverted-V
    threshold = 0.05
    
    if np.abs(slope_pre) < threshold and np.abs(slope_post) < threshold:
        category = 0  # All flat
    elif np.abs(slope_pre) >= threshold and np.abs(slope_post) < threshold:
        category = 1 if slope_pre > 0 else 2  # Up-flat or down-flat
    elif np.abs(slope_pre) < threshold and np.abs(slope_post) >= threshold:
        category = 3 if slope_post > 0 else 4  # Flat-up or flat-down
    elif slope_pre > 0 and slope_post > 0:
        category = 3  # Trend up
    elif slope_pre < 0 and slope_post < 0:
        category = 4  # Trend down
    elif slope_pre > 0 and slope_post < 0:
        category = 6  # Inverted V
    else:  # slope_pre < 0 and slope_post > 0
        category = 5  # V shape
    
    features['elbow_category'] = category
    
    return features


def wasserstein_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute 1D Wasserstein distance (fast version)."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    return stats.wasserstein_distance(x, y)


def compute_cusum_features(x0: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """
    Compute CUSUM-based features.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        
    Returns:
        Dictionary of CUSUM features
    """
    features = {}
    
    # Compute CUSUM for each segment
    cusum_pre = compute_cusum(x0, normalize=True)
    cusum_post = compute_cusum(x1, normalize=True)
    
    # Also compute on combined series
    x_all = np.concatenate([x0, x1])
    cusum_global = compute_cusum(x_all, normalize=True)
    cusum_global_pre = cusum_global[:len(x0)]
    cusum_global_post = cusum_global[len(x0):]
    
    # Basic CUSUM statistics
    if len(cusum_pre) > 0:
        features['cusum_pre_final'] = cusum_pre[-1]
        features['cusum_pre_max'] = np.max(cusum_pre)
        features['cusum_pre_min'] = np.min(cusum_pre)
        features['cusum_pre_range'] = features['cusum_pre_max'] - features['cusum_pre_min']
        features['cusum_pre_std'] = np.std(cusum_pre)
    else:
        features['cusum_pre_final'] = 0.0
        features['cusum_pre_max'] = 0.0
        features['cusum_pre_min'] = 0.0
        features['cusum_pre_range'] = 0.0
        features['cusum_pre_std'] = 0.0
    
    if len(cusum_post) > 0:
        features['cusum_post_final'] = cusum_post[-1]
        features['cusum_post_max'] = np.max(cusum_post)
        features['cusum_post_min'] = np.min(cusum_post)
        features['cusum_post_range'] = features['cusum_post_max'] - features['cusum_post_min']
        features['cusum_post_std'] = np.std(cusum_post)
    else:
        features['cusum_post_final'] = 0.0
        features['cusum_post_max'] = 0.0
        features['cusum_post_min'] = 0.0
        features['cusum_post_range'] = 0.0
        features['cusum_post_std'] = 0.0
    
    # CUSUM changes
    features['cusum_range_diff'] = features['cusum_post_range'] - features['cusum_pre_range']
    features['cusum_std_diff'] = features['cusum_post_std'] - features['cusum_pre_std']
    
    # Global CUSUM jump at boundary
    if len(cusum_global_pre) > 0 and len(cusum_global_post) > 0:
        features['cusum_global_jump'] = cusum_global_post[0] - cusum_global_pre[-1]
        features['cusum_global_pre_final'] = cusum_global_pre[-1]
        features['cusum_global_post_first'] = cusum_global_post[0]
    else:
        features['cusum_global_jump'] = 0.0
        features['cusum_global_pre_final'] = 0.0
        features['cusum_global_post_first'] = 0.0
    
    # Elbow detection
    elbow_feats = detect_cusum_elbow(cusum_global_pre, cusum_global_post, window=20)
    features.update(elbow_feats)
    
    # CUSUM path statistics
    # Path length (total variation)
    if len(cusum_pre) > 1:
        features['cusum_pre_path_length'] = np.sum(np.abs(np.diff(cusum_pre)))
    else:
        features['cusum_pre_path_length'] = 0.0
    
    if len(cusum_post) > 1:
        features['cusum_post_path_length'] = np.sum(np.abs(np.diff(cusum_post)))
    else:
        features['cusum_post_path_length'] = 0.0
    
    features['cusum_path_length_ratio'] = (features['cusum_post_path_length'] / 
                                           (features['cusum_pre_path_length'] + 1e-8))
    
    # CUSUM residuals (deviations from linear trend)
    if len(cusum_pre) > 2:
        # Fit linear trend
        t_pre = np.arange(len(cusum_pre))
        slope_pre, intercept_pre = np.polyfit(t_pre, cusum_pre, 1)
        cusum_pre_trend = slope_pre * t_pre + intercept_pre
        cusum_pre_residuals = cusum_pre - cusum_pre_trend
    else:
        cusum_pre_residuals = cusum_pre
    
    if len(cusum_post) > 2:
        # Fit linear trend
        t_post = np.arange(len(cusum_post))
        slope_post, intercept_post = np.polyfit(t_post, cusum_post, 1)
        cusum_post_trend = slope_post * t_post + intercept_post
        cusum_post_residuals = cusum_post - cusum_post_trend
    else:
        cusum_post_residuals = cusum_post
    
    # Wasserstein distance of CUSUM residuals (from 6th place solution)
    if len(cusum_pre_residuals) > 0 and len(cusum_post_residuals) > 0:
        features['cusum_error_wasserstein'] = wasserstein_distance_1d(
            cusum_pre_residuals, cusum_post_residuals
        )
    else:
        features['cusum_error_wasserstein'] = 0.0
    
    # CUSUM volatility around boundary
    window = min(50, len(x0) // 4, len(x1) // 4)
    if window > 5:
        cusum_pre_boundary = cusum_global_pre[-window:]
        cusum_post_boundary = cusum_global_post[:window]
        
        features['cusum_boundary_vol_pre'] = np.std(np.diff(cusum_pre_boundary)) if len(cusum_pre_boundary) > 1 else 0.0
        features['cusum_boundary_vol_post'] = np.std(np.diff(cusum_post_boundary)) if len(cusum_post_boundary) > 1 else 0.0
        features['cusum_boundary_vol_ratio'] = (features['cusum_boundary_vol_post'] / 
                                                (features['cusum_boundary_vol_pre'] + 1e-8))
    else:
        features['cusum_boundary_vol_pre'] = 0.0
        features['cusum_boundary_vol_post'] = 0.0
        features['cusum_boundary_vol_ratio'] = 1.0
    
    return features


def compute_cusum_features_multiscale(x0: np.ndarray, x1: np.ndarray,
                                       windows=[50, 100, 250]) -> Dict[str, float]:
    """
    Compute CUSUM features at multiple scales.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        windows: List of window sizes
        
    Returns:
        Dictionary of CUSUM features at multiple scales
    """
    features = {}
    
    # Full-scale features
    full_feats = compute_cusum_features(x0, x1)
    for k, v in full_feats.items():
        features[f'{k}_full'] = v
    
    # Multi-scale features (boundary-focused)
    for w in windows:
        # Last w points of x0, first w points of x1
        x0_w = x0[-w:] if len(x0) >= w else x0
        x1_w = x1[:w] if len(x1) >= w else x1
        
        if len(x0_w) > 5 and len(x1_w) > 5:
            w_feats = compute_cusum_features(x0_w, x1_w)
            for k, v in w_feats.items():
                features[f'{k}_w{w}'] = v
    
    return features
