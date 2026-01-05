"""
Boundary-localized distribution distance features.

Compares distributions near the boundary between pre-break (x0) and post-break (x1) segments
using Wasserstein and Energy distances.

All features are leakage-free: computed per-series from x0 and x1 only.
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats as scipy_stats


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


def quantile_bounds(arr: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float]:
    """
    Compute quantile bounds for array.
    
    Args:
        arr: Input array
        q_low: Lower quantile (e.g., 0.05)
        q_high: Upper quantile (e.g., 0.95)
        
    Returns:
        (lower_bound, upper_bound) or (np.nan, np.nan) if empty
    """
    arr_clean = arr[~np.isnan(arr)]
    
    if len(arr_clean) == 0:
        return (np.nan, np.nan)
    
    lo = np.quantile(arr_clean, q_low)
    hi = np.quantile(arr_clean, q_high)
    
    return (lo, hi)


def signed_dod(a: float, b: float) -> float:
    """
    Compute signed difference of distances (DoD).
    
    Args:
        a: First distance
        b: Second distance
        
    Returns:
        a - b if both are finite, else np.nan
    """
    if np.isfinite(a) and np.isfinite(b):
        return a - b
    return np.nan


def _collect_window_series(
    features: Dict[str, float],
    key_pattern: str,
    windows: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect valid (finite) feature values across windows.
    
    Args:
        features: Feature dictionary
        key_pattern: Pattern like 'bl_wasserstein_z_w{w}'
        windows: Window sizes to check
        
    Returns:
        (logw, d): Arrays of log(window) and distance values for valid windows,
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


def _dod_stats(logw: np.ndarray, d: np.ndarray, prefix: str, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compute distance-of-distance statistics across windows.
    
    Expects logw and d to be sorted by logw ascending. Defensively sorts
    to ensure correct delta_wmin_wmax, ratio_wmin_wmax, and curvature.
    
    Args:
        logw: Log of window sizes
        d: Distance values
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
        # Use polyfit for least squares
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
        # (dA - dB) - (dB - dC) = dA - 2*dB + dC
        features[f'{prefix}_curv_wA_wB_wC'] = d[0] - 2 * d[1] + d[2]
    
    return features


def add_dod_ratio_ndiff(features: Dict[str, float], scope: str, eps: float = 1e-12) -> None:
    """
    Add ratio, normalized-difference, and log-ratio DoD features for a scope.
    
    Computes scale-free comparisons between Wasserstein and Energy distances,
    and between their winsorized variants.
    
    Args:
        features: Feature dictionary to update
        scope: Scope identifier (e.g., 'w25', 'w50', 'full')
        eps: Small constant for numerical stability
    """
    # Base distance keys
    k_wassz = f"bl_wasserstein_z_{scope}"
    k_enez = f"bl_energy_z_{scope}"
    
    # Winsorized distance keys
    k_wwassz = f"bl_tail_wins_wasserstein_z_{scope}"
    k_wenez = f"bl_tail_wins_energy_z_{scope}"
    
    # Get values (default to NaN if missing)
    wassz = features.get(k_wassz, np.nan)
    enez = features.get(k_enez, np.nan)
    wwassz = features.get(k_wwassz, np.nan)
    wenez = features.get(k_wenez, np.nan)
    
    # A) Ratio DoD: wassz / enez
    ratio_key = f"bl_dod_ratio_wassz_enez_{scope}"
    if ratio_key not in features:
        if np.isfinite(wassz) and np.isfinite(enez):
            features[ratio_key] = wassz / (enez + eps)
        else:
            features[ratio_key] = np.nan
    
    # B) Normalized difference DoD: (wassz - enez) / (|wassz| + |enez| + eps)
    ndiff_key = f"bl_dod_ndiff_wassz_enez_{scope}"
    if ndiff_key not in features:
        if np.isfinite(wassz) and np.isfinite(enez):
            features[ndiff_key] = (wassz - enez) / (abs(wassz) + abs(enez) + eps)
        else:
            features[ndiff_key] = np.nan
    
    # C) Log-ratio DoD: log((wassz + eps) / (enez + eps))
    logratio_key = f"bl_dod_logratio_wassz_enez_{scope}"
    if logratio_key not in features:
        if np.isfinite(wassz) and np.isfinite(enez):
            features[logratio_key] = np.log((wassz + eps) / (enez + eps))
        else:
            features[logratio_key] = np.nan
    
    # D) Ratio DoD for winsorized: wins_wassz / wins_enez
    ratio_wins_key = f"bl_dod_ratio_wins_wassz_wins_enez_{scope}"
    if ratio_wins_key not in features:
        if np.isfinite(wwassz) and np.isfinite(wenez):
            features[ratio_wins_key] = wwassz / (wenez + eps)
        else:
            features[ratio_wins_key] = np.nan
    
    # E) Normalized difference DoD for winsorized
    ndiff_wins_key = f"bl_dod_ndiff_wins_wassz_wins_enez_{scope}"
    if ndiff_wins_key not in features:
        if np.isfinite(wwassz) and np.isfinite(wenez):
            features[ndiff_wins_key] = (wwassz - wenez) / (abs(wwassz) + abs(wenez) + eps)
        else:
            features[ndiff_wins_key] = np.nan
    
    # F) Log-ratio DoD for winsorized
    logratio_wins_key = f"bl_dod_logratio_wins_wassz_wins_enez_{scope}"
    if logratio_wins_key not in features:
        if np.isfinite(wwassz) and np.isfinite(wenez):
            features[logratio_wins_key] = np.log((wwassz + eps) / (wenez + eps))
        else:
            features[logratio_wins_key] = np.nan


def compute_localization_features(base: Dict[str, float], w: int, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compute localization features: boundary window vs full segment distances.
    
    Measures how much the regime change is localized at the boundary versus
    being spread across the full time series.
    
    Args:
        base: Feature dictionary with pre-computed distances
        w: Window size
        eps: Small constant for numerical stability
        
    Returns:
        Dictionary of localization features
    """
    loc_features = {}
    
    # Define metrics to compute localization for
    metrics = [
        ('wassz', f'bl_wasserstein_z_w{w}', 'bl_wasserstein_z_full'),
        ('enez', f'bl_energy_z_w{w}', 'bl_energy_z_full'),
        ('wins_wassz', f'bl_tail_wins_wasserstein_z_w{w}', 'bl_tail_wins_wasserstein_z_full'),
        ('wins_enez', f'bl_tail_wins_energy_z_w{w}', 'bl_tail_wins_energy_z_full')
    ]
    
    for metric_name, key_w, key_full in metrics:
        # Get values from base dictionary
        d_w = base.get(key_w, np.nan)
        d_full = base.get(key_full, np.nan)
        
        # Check if both are finite
        if np.isfinite(d_w) and np.isfinite(d_full):
            # 1) Signed difference
            loc_features[f'bl_loc_diff_{metric_name}_w{w}'] = d_w - d_full
            
            # 2) Absolute difference
            loc_features[f'bl_loc_absdiff_{metric_name}_w{w}'] = abs(d_w - d_full)
            
            # 3) Normalized difference
            loc_features[f'bl_loc_ndiff_{metric_name}_w{w}'] = (d_w - d_full) / (abs(d_w) + abs(d_full) + eps)
            
            # 4) Ratio
            loc_features[f'bl_loc_ratio_{metric_name}_w{w}'] = d_w / (d_full + eps)
        else:
            # If either is NaN, set all to NaN
            loc_features[f'bl_loc_diff_{metric_name}_w{w}'] = np.nan
            loc_features[f'bl_loc_absdiff_{metric_name}_w{w}'] = np.nan
            loc_features[f'bl_loc_ndiff_{metric_name}_w{w}'] = np.nan
            loc_features[f'bl_loc_ratio_{metric_name}_w{w}'] = np.nan
    
    return loc_features


def select_tail(arr: np.ndarray, q: float = 0.9, side: str = "upper") -> np.ndarray:
    """
    Select tail region of distribution.
    
    Args:
        arr: Input array
        q: Quantile threshold (e.g., 0.9 for top 10%)
        side: "upper", "lower", or "both"
        
    Returns:
        Array of tail values (empty if <2 samples)
    """
    arr_clean = arr[~np.isnan(arr)]
    
    if len(arr_clean) < 2:
        return np.array([])
    
    if side == "upper":
        threshold = np.quantile(arr_clean, q)
        tail = arr_clean[arr_clean >= threshold]
    elif side == "lower":
        threshold = np.quantile(arr_clean, 1 - q)
        tail = arr_clean[arr_clean <= threshold]
    elif side == "both":
        threshold_lo = np.quantile(arr_clean, 1 - q)
        threshold_hi = np.quantile(arr_clean, q)
        tail = arr_clean[(arr_clean <= threshold_lo) | (arr_clean >= threshold_hi)]
    else:
        raise ValueError(f"Invalid side: {side}")
    
    if len(tail) < 2:
        return np.array([])
    
    return tail


def winsorize(arr: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    """
    Winsorize array by clipping to quantile bounds.
    
    Args:
        arr: Input array
        q_low: Lower quantile bound
        q_high: Upper quantile bound
        
    Returns:
        Winsorized array (empty if <2 samples)
    """
    arr_clean = arr[~np.isnan(arr)]
    
    if len(arr_clean) < 2:
        return np.array([])
    
    lo, hi = quantile_bounds(arr_clean, q_low, q_high)
    
    if np.isnan(lo) or np.isnan(hi):
        return np.array([])
    
    return np.clip(arr_clean, lo, hi)


def safe_distances(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Safely compute Wasserstein and Energy distances.
    
    Args:
        a, b: Input arrays
        
    Returns:
        (wasserstein, energy) or (np.nan, np.nan) if insufficient data
    """
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    
    try:
        wass = scipy_stats.wasserstein_distance(a, b)
    except:
        wass = np.nan
    
    try:
        energy = scipy_stats.energy_distance(a, b)
    except:
        energy = np.nan
    
    return (wass, energy)


def compute_tail_features(x0_scope: np.ndarray, x1_scope: np.ndarray, scope_name: str) -> Dict[str, float]:
    """
    Compute tail-restricted distance features for a given scope.
    
    Args:
        x0_scope: Pre-break segment (boundary window or full)
        x1_scope: Post-break segment (boundary window or full)
        scope_name: "w{w}" or "full"
        
    Returns:
        Dictionary of tail features
    """
    features = {}
    
    # Clean inputs
    x0_clean = x0_scope[~np.isnan(x0_scope)]
    x1_clean = x1_scope[~np.isnan(x1_scope)]
    
    if len(x0_clean) < 2 or len(x1_clean) < 2:
        # Fill all tail features with NaN
        for q_tail in [0.9, 0.95]:
            q_str = f"q{int(q_tail*100)}"
            for side in ["upper", "lower", "both"]:
                features[f'bl_tail_wasserstein_{q_str}_{side}_{scope_name}'] = np.nan
                features[f'bl_tail_energy_{q_str}_{side}_{scope_name}'] = np.nan
                features[f'bl_tail_wasserstein_z_{q_str}_{side}_{scope_name}'] = np.nan
                features[f'bl_tail_energy_z_{q_str}_{side}_{scope_name}'] = np.nan
            
            features[f'bl_tail_p_hi_delta_{q_str}_{scope_name}'] = np.nan
            features[f'bl_tail_p_lo_delta_{q_str}_{scope_name}'] = np.nan
            features[f'bl_tail_mean_excess_hi_delta_{q_str}_{scope_name}'] = np.nan
            features[f'bl_tail_mean_excess_lo_delta_{q_str}_{scope_name}'] = np.nan
        
        features[f'bl_tail_wins_wasserstein_{scope_name}'] = np.nan
        features[f'bl_tail_wins_energy_{scope_name}'] = np.nan
        features[f'bl_tail_wins_wasserstein_z_{scope_name}'] = np.nan
        features[f'bl_tail_wins_energy_z_{scope_name}'] = np.nan
        
        return features
    
    # A) Tail-only distances
    for q_tail in [0.9, 0.95]:
        q_str = f"q{int(q_tail*100)}"
        
        for side in ["upper", "lower", "both"]:
            # Select tails
            x0_tail = select_tail(x0_clean, q=q_tail, side=side)
            x1_tail = select_tail(x1_clean, q=q_tail, side=side)
            
            # Compute distances
            wass, energy = safe_distances(x0_tail, x1_tail)
            features[f'bl_tail_wasserstein_{q_str}_{side}_{scope_name}'] = wass
            features[f'bl_tail_energy_{q_str}_{side}_{scope_name}'] = energy
            
            # Standardized versions
            if len(x0_tail) >= 2 and len(x1_tail) >= 2:
                pooled_tail = np.concatenate([x0_tail, x1_tail])
                scale_tail = robust_scale(pooled_tail)
                
                features[f'bl_tail_wasserstein_z_{q_str}_{side}_{scope_name}'] = wass / scale_tail if not np.isnan(wass) else np.nan
                features[f'bl_tail_energy_z_{q_str}_{side}_{scope_name}'] = energy / scale_tail if not np.isnan(energy) else np.nan
            else:
                features[f'bl_tail_wasserstein_z_{q_str}_{side}_{scope_name}'] = np.nan
                features[f'bl_tail_energy_z_{q_str}_{side}_{scope_name}'] = np.nan
    
    # B) Winsorized distances
    x0_win = winsorize(x0_clean, q_low=0.05, q_high=0.95)
    x1_win = winsorize(x1_clean, q_low=0.05, q_high=0.95)
    
    wass_win, energy_win = safe_distances(x0_win, x1_win)
    features[f'bl_tail_wins_wasserstein_{scope_name}'] = wass_win
    features[f'bl_tail_wins_energy_{scope_name}'] = energy_win
    
    # Standardized winsorized
    if len(x0_win) >= 2 and len(x1_win) >= 2:
        pooled_win = np.concatenate([x0_win, x1_win])
        scale_win = robust_scale(pooled_win)
        
        features[f'bl_tail_wins_wasserstein_z_{scope_name}'] = wass_win / scale_win if not np.isnan(wass_win) else np.nan
        features[f'bl_tail_wins_energy_z_{scope_name}'] = energy_win / scale_win if not np.isnan(energy_win) else np.nan
    else:
        features[f'bl_tail_wins_wasserstein_z_{scope_name}'] = np.nan
        features[f'bl_tail_wins_energy_z_{scope_name}'] = np.nan
    
    # C) Tail mass / exceedance diagnostics
    for q_tail in [0.9, 0.95]:
        q_str = f"q{int(q_tail*100)}"
        
        # Pooled thresholds
        pooled = np.concatenate([x0_clean, x1_clean])
        hi = np.quantile(pooled, q_tail)
        lo = np.quantile(pooled, 1 - q_tail)
        
        # Upper tail probability shift
        p0_hi = np.mean(x0_clean >= hi) if len(x0_clean) > 0 else np.nan
        p1_hi = np.mean(x1_clean >= hi) if len(x1_clean) > 0 else np.nan
        features[f'bl_tail_p_hi_delta_{q_str}_{scope_name}'] = p1_hi - p0_hi if not np.isnan(p0_hi) else np.nan
        
        # Lower tail probability shift
        p0_lo = np.mean(x0_clean <= lo) if len(x0_clean) > 0 else np.nan
        p1_lo = np.mean(x1_clean <= lo) if len(x1_clean) > 0 else np.nan
        features[f'bl_tail_p_lo_delta_{q_str}_{scope_name}'] = p1_lo - p0_lo if not np.isnan(p0_lo) else np.nan
        
        # Mean exceedance (upper tail)
        x0_exceed_hi = x0_clean[x0_clean >= hi]
        x1_exceed_hi = x1_clean[x1_clean >= hi]
        
        if len(x0_exceed_hi) > 0 and len(x1_exceed_hi) > 0:
            features[f'bl_tail_mean_excess_hi_delta_{q_str}_{scope_name}'] = np.mean(x1_exceed_hi) - np.mean(x0_exceed_hi)
        else:
            features[f'bl_tail_mean_excess_hi_delta_{q_str}_{scope_name}'] = np.nan
        
        # Mean exceedance (lower tail)
        x0_exceed_lo = x0_clean[x0_clean <= lo]
        x1_exceed_lo = x1_clean[x1_clean <= lo]
        
        if len(x0_exceed_lo) > 0 and len(x1_exceed_lo) > 0:
            features[f'bl_tail_mean_excess_lo_delta_{q_str}_{scope_name}'] = np.mean(x1_exceed_lo) - np.mean(x0_exceed_lo)
        else:
            features[f'bl_tail_mean_excess_lo_delta_{q_str}_{scope_name}'] = np.nan
    
    return features


def compute_boundary_dist_features(
    x0: np.ndarray,
    x1: np.ndarray,
    windows: Tuple[int, ...] = (25, 50, 100, 250)
) -> Dict[str, float]:
    """
    Compute boundary-localized distribution distance features.
    
    Args:
        x0: Pre-break segment
        x1: Post-break segment
        windows: Window sizes to analyze
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Remove NaNs from full segments
    x0_full = x0[~np.isnan(x0)]
    x1_full = x1[~np.isnan(x1)]
    
    # Process each window size
    for w in windows:
        # Extract boundary windows
        x0b, x1b = extract_boundary_segments(x0, x1, w)
        
        # Check if we have enough samples
        if len(x0b) < 2 or len(x1b) < 2:
            # Fill with NaN for this window
            features[f'bl_wasserstein_w{w}'] = np.nan
            features[f'bl_energy_w{w}'] = np.nan
            features[f'bl_wasserstein_z_w{w}'] = np.nan
            features[f'bl_energy_z_w{w}'] = np.nan
            features[f'bl_mean_delta_w{w}'] = np.nan
            features[f'bl_median_delta_w{w}'] = np.nan
            continue
        
        # A) Wasserstein distance
        try:
            wass = scipy_stats.wasserstein_distance(x0b, x1b)
            features[f'bl_wasserstein_w{w}'] = wass
        except:
            features[f'bl_wasserstein_w{w}'] = np.nan
        
        # B) Energy distance
        try:
            energy = scipy_stats.energy_distance(x0b, x1b)
            features[f'bl_energy_w{w}'] = energy
        except:
            features[f'bl_energy_w{w}'] = np.nan
        
        # C) Standardized versions
        # Compute pooled robust scale
        pooled = np.concatenate([x0b, x1b])
        scale = robust_scale(pooled)
        
        try:
            wass_z = scipy_stats.wasserstein_distance(x0b, x1b) / scale
            features[f'bl_wasserstein_z_w{w}'] = wass_z
        except:
            features[f'bl_wasserstein_z_w{w}'] = np.nan
        
        try:
            energy_z = scipy_stats.energy_distance(x0b, x1b) / scale
            features[f'bl_energy_z_w{w}'] = energy_z
        except:
            features[f'bl_energy_z_w{w}'] = np.nan
        
        # D) Location shifts
        try:
            mean_delta = np.mean(x1b) - np.mean(x0b)
            features[f'bl_mean_delta_w{w}'] = mean_delta
        except:
            features[f'bl_mean_delta_w{w}'] = np.nan
        
        try:
            median_delta = np.median(x1b) - np.median(x0b)
            features[f'bl_median_delta_w{w}'] = median_delta
        except:
            features[f'bl_median_delta_w{w}'] = np.nan
        
        # E) Tail-restricted features
        tail_feats_w = compute_tail_features(x0b, x1b, f"w{w}")
        features.update(tail_feats_w)
        
        # F) Signed DoD features (after base z and tail z features exist)
        wz = features.get(f'bl_wasserstein_z_w{w}', np.nan)
        ez = features.get(f'bl_energy_z_w{w}', np.nan)
        features[f'bl_dod_diff_wassz_enez_w{w}'] = signed_dod(wz, ez)
        
        wwz = features.get(f'bl_tail_wins_wasserstein_z_w{w}', np.nan)
        wez = features.get(f'bl_tail_wins_energy_z_w{w}', np.nan)
        features[f'bl_dod_diff_wins_wassz_wins_enez_w{w}'] = signed_dod(wwz, wez)
        
        # G) Ratio, normalized-difference, and log-ratio DoD features
        add_dod_ratio_ndiff(features, f"w{w}")
    
    # Compute for full segments
    if len(x0_full) >= 2 and len(x1_full) >= 2:
        # A) Wasserstein distance
        try:
            wass_full = scipy_stats.wasserstein_distance(x0_full, x1_full)
            features['bl_wasserstein_full'] = wass_full
        except:
            features['bl_wasserstein_full'] = np.nan
        
        # B) Energy distance
        try:
            energy_full = scipy_stats.energy_distance(x0_full, x1_full)
            features['bl_energy_full'] = energy_full
        except:
            features['bl_energy_full'] = np.nan
        
        # C) Standardized versions
        pooled_full = np.concatenate([x0_full, x1_full])
        scale_full = robust_scale(pooled_full)
        
        try:
            wass_z_full = scipy_stats.wasserstein_distance(x0_full, x1_full) / scale_full
            features['bl_wasserstein_z_full'] = wass_z_full
        except:
            features['bl_wasserstein_z_full'] = np.nan
        
        try:
            energy_z_full = scipy_stats.energy_distance(x0_full, x1_full) / scale_full
            features['bl_energy_z_full'] = energy_z_full
        except:
            features['bl_energy_z_full'] = np.nan
        
        # D) Location shifts
        try:
            mean_delta_full = np.mean(x1_full) - np.mean(x0_full)
            features['bl_mean_delta_full'] = mean_delta_full
        except:
            features['bl_mean_delta_full'] = np.nan
        
        try:
            median_delta_full = np.median(x1_full) - np.median(x0_full)
            features['bl_median_delta_full'] = median_delta_full
        except:
            features['bl_median_delta_full'] = np.nan
        
        # E) Tail-restricted features for full segment
        tail_feats_full = compute_tail_features(x0_full, x1_full, "full")
        features.update(tail_feats_full)
        
        # F) Signed DoD features for full scope
        wz_full = features.get('bl_wasserstein_z_full', np.nan)
        ez_full = features.get('bl_energy_z_full', np.nan)
        features['bl_dod_diff_wassz_enez_full'] = signed_dod(wz_full, ez_full)
        
        wwz_full = features.get('bl_tail_wins_wasserstein_z_full', np.nan)
        wez_full = features.get('bl_tail_wins_energy_z_full', np.nan)
        features['bl_dod_diff_wins_wassz_wins_enez_full'] = signed_dod(wwz_full, wez_full)
        
        # G) Ratio, normalized-difference, and log-ratio DoD features
        add_dod_ratio_ndiff(features, "full")
    else:
        # Fill with NaN
        features['bl_wasserstein_full'] = np.nan
        features['bl_energy_full'] = np.nan
        features['bl_wasserstein_z_full'] = np.nan
        features['bl_energy_z_full'] = np.nan
        features['bl_mean_delta_full'] = np.nan
        features['bl_median_delta_full'] = np.nan
        features['bl_dod_diff_wassz_enez_full'] = np.nan
        features['bl_dod_diff_wins_wassz_wins_enez_full'] = np.nan
        # Add NaN for ratio/ndiff/logratio DoD features
        add_dod_ratio_ndiff(features, "full")
    
    # ===================================================================
    # DISTANCE-OF-DISTANCE (DoD) FEATURES
    # ===================================================================
    # Analyze how distances change across window sizes
    
    eps = 1e-12
    
    # A) DoD statistics for main distance metrics across windows
    for key_pattern, prefix in [
        ('bl_wasserstein_z_w{w}', 'bl_dod_wasserstein_z'),
        ('bl_energy_z_w{w}', 'bl_dod_energy_z'),
        ('bl_tail_wins_wasserstein_z_w{w}', 'bl_dod_wins_wasserstein_z'),
        ('bl_tail_wins_energy_z_w{w}', 'bl_dod_wins_energy_z')
    ]:
        logw, d = _collect_window_series(features, key_pattern, windows)
        dod_feats = _dod_stats(logw, d, prefix, eps=eps)
        features.update(dod_feats)
    
    # B) Cross-metric agreement features (per window and full)
    for w in windows:
        # Wasserstein vs Energy (standardized)
        wz_key = f'bl_wasserstein_z_w{w}'
        ez_key = f'bl_energy_z_w{w}'
        if wz_key in features and ez_key in features:
            wz = features[wz_key]
            ez = features[ez_key]
            if np.isfinite(wz) and np.isfinite(ez):
                absdiff = abs(wz - ez)
                features[f'bl_dod_absdiff_wassz_enez_w{w}'] = absdiff
                features[f'bl_dod_reldiff_wassz_enez_w{w}'] = absdiff / (abs(wz) + abs(ez) + eps)
            else:
                features[f'bl_dod_absdiff_wassz_enez_w{w}'] = np.nan
                features[f'bl_dod_reldiff_wassz_enez_w{w}'] = np.nan
        else:
            features[f'bl_dod_absdiff_wassz_enez_w{w}'] = np.nan
            features[f'bl_dod_reldiff_wassz_enez_w{w}'] = np.nan
        
        # Winsorized Wasserstein vs Energy (standardized)
        wwz_key = f'bl_tail_wins_wasserstein_z_w{w}'
        wez_key = f'bl_tail_wins_energy_z_w{w}'
        if wwz_key in features and wez_key in features:
            wwz = features[wwz_key]
            wez = features[wez_key]
            if np.isfinite(wwz) and np.isfinite(wez):
                absdiff = abs(wwz - wez)
                features[f'bl_dod_absdiff_wins_wassz_wins_enez_w{w}'] = absdiff
                features[f'bl_dod_reldiff_wins_wassz_wins_enez_w{w}'] = absdiff / (abs(wwz) + abs(wez) + eps)
            else:
                features[f'bl_dod_absdiff_wins_wassz_wins_enez_w{w}'] = np.nan
                features[f'bl_dod_reldiff_wins_wassz_wins_enez_w{w}'] = np.nan
        else:
            features[f'bl_dod_absdiff_wins_wassz_wins_enez_w{w}'] = np.nan
            features[f'bl_dod_reldiff_wins_wassz_wins_enez_w{w}'] = np.nan
    
    # Full scope cross-metric agreement
    wz_full = features.get('bl_wasserstein_z_full', np.nan)
    ez_full = features.get('bl_energy_z_full', np.nan)
    if np.isfinite(wz_full) and np.isfinite(ez_full):
        absdiff = abs(wz_full - ez_full)
        features['bl_dod_absdiff_wassz_enez_full'] = absdiff
        features['bl_dod_reldiff_wassz_enez_full'] = absdiff / (abs(wz_full) + abs(ez_full) + eps)
    else:
        features['bl_dod_absdiff_wassz_enez_full'] = np.nan
        features['bl_dod_reldiff_wassz_enez_full'] = np.nan
    
    wwz_full = features.get('bl_tail_wins_wasserstein_z_full', np.nan)
    wez_full = features.get('bl_tail_wins_energy_z_full', np.nan)
    if np.isfinite(wwz_full) and np.isfinite(wez_full):
        absdiff = abs(wwz_full - wez_full)
        features['bl_dod_absdiff_wins_wassz_wins_enez_full'] = absdiff
        features['bl_dod_reldiff_wins_wassz_wins_enez_full'] = absdiff / (abs(wwz_full) + abs(wez_full) + eps)
    else:
        features['bl_dod_absdiff_wins_wassz_wins_enez_full'] = np.nan
        features['bl_dod_reldiff_wins_wassz_wins_enez_full'] = np.nan
    
    # C) Outlier sensitivity features (raw z minus winsorized z)
    for w in windows:
        # Wasserstein
        wz_key = f'bl_wasserstein_z_w{w}'
        wwz_key = f'bl_tail_wins_wasserstein_z_w{w}'
        if wz_key in features and wwz_key in features:
            wz = features[wz_key]
            wwz = features[wwz_key]
            if np.isfinite(wz) and np.isfinite(wwz):
                features[f'bl_dod_outlier_sens_wassz_w{w}'] = wz - wwz
            else:
                features[f'bl_dod_outlier_sens_wassz_w{w}'] = np.nan
        else:
            features[f'bl_dod_outlier_sens_wassz_w{w}'] = np.nan
        
        # Energy
        ez_key = f'bl_energy_z_w{w}'
        wez_key = f'bl_tail_wins_energy_z_w{w}'
        if ez_key in features and wez_key in features:
            ez = features[ez_key]
            wez = features[wez_key]
            if np.isfinite(ez) and np.isfinite(wez):
                features[f'bl_dod_outlier_sens_enez_w{w}'] = ez - wez
            else:
                features[f'bl_dod_outlier_sens_enez_w{w}'] = np.nan
        else:
            features[f'bl_dod_outlier_sens_enez_w{w}'] = np.nan
    
    # Full scope outlier sensitivity
    if np.isfinite(wz_full) and np.isfinite(wwz_full):
        features['bl_dod_outlier_sens_wassz_full'] = wz_full - wwz_full
    else:
        features['bl_dod_outlier_sens_wassz_full'] = np.nan
    
    if np.isfinite(ez_full) and np.isfinite(wez_full):
        features['bl_dod_outlier_sens_enez_full'] = ez_full - wez_full
    else:
        features['bl_dod_outlier_sens_enez_full'] = np.nan
    
    # D) Tail-vs-bulk consistency features (optional, if tail features exist)
    for w in windows:
        for q_str in ['q90', 'q95']:
            # Wasserstein
            tail_key = f'bl_tail_wasserstein_z_{q_str}_both_w{w}'
            wins_key = f'bl_tail_wins_wasserstein_z_w{w}'
            if tail_key in features and wins_key in features:
                tail_val = features[tail_key]
                wins_val = features[wins_key]
                if np.isfinite(tail_val) and np.isfinite(wins_val):
                    features[f'bl_dod_tailbulk_wassz_{q_str}_both_w{w}'] = tail_val - wins_val
                else:
                    features[f'bl_dod_tailbulk_wassz_{q_str}_both_w{w}'] = np.nan
            
            # Energy
            tail_key = f'bl_tail_energy_z_{q_str}_both_w{w}'
            wins_key = f'bl_tail_wins_energy_z_w{w}'
            if tail_key in features and wins_key in features:
                tail_val = features[tail_key]
                wins_val = features[wins_key]
                if np.isfinite(tail_val) and np.isfinite(wins_val):
                    features[f'bl_dod_tailbulk_enez_{q_str}_both_w{w}'] = tail_val - wins_val
                else:
                    features[f'bl_dod_tailbulk_enez_{q_str}_both_w{w}'] = np.nan
    
    # Full scope tail-vs-bulk
    for q_str in ['q90', 'q95']:
        # Wasserstein
        tail_key = f'bl_tail_wasserstein_z_{q_str}_both_full'
        wins_key = 'bl_tail_wins_wasserstein_z_full'
        if tail_key in features and wins_key in features:
            tail_val = features[tail_key]
            wins_val = features[wins_key]
            if np.isfinite(tail_val) and np.isfinite(wins_val):
                features[f'bl_dod_tailbulk_wassz_{q_str}_both_full'] = tail_val - wins_val
            else:
                features[f'bl_dod_tailbulk_wassz_{q_str}_both_full'] = np.nan
        
        # Energy
        tail_key = f'bl_tail_energy_z_{q_str}_both_full'
        wins_key = 'bl_tail_wins_energy_z_full'
        if tail_key in features and wins_key in features:
            tail_val = features[tail_key]
            wins_val = features[wins_key]
            if np.isfinite(tail_val) and np.isfinite(wins_val):
                features[f'bl_dod_tailbulk_enez_{q_str}_both_full'] = tail_val - wins_val
            else:
                features[f'bl_dod_tailbulk_enez_{q_str}_both_full'] = np.nan
    
    # E) Localization features (boundary vs full) - compute after full features exist
    for w in windows:
        loc_feats = compute_localization_features(features, w)
        features.update(loc_feats)
    
    return features


if __name__ == "__main__":
    # Sanity check
    print("=" * 70)
    print("BOUNDARY DISTRIBUTION DISTANCE FEATURES - SELF-CHECK")
    print("=" * 70)
    
    # Create synthetic data: x1 shifted by +1
    np.random.seed(42)
    x0 = np.random.randn(200) + 0.0
    x1 = np.random.randn(200) + 1.0  # Shifted distribution
    
    print("\nTest 1: Normal case (x1 shifted by +1)")
    features = compute_boundary_dist_features(x0, x1, windows=(25, 50))
    
    print(f"  bl_wasserstein_w25: {features['bl_wasserstein_w25']:.4f}")
    print(f"  bl_energy_w25: {features['bl_energy_w25']:.4f}")
    print(f"  bl_wasserstein_z_w25: {features['bl_wasserstein_z_w25']:.4f}")
    print(f"  bl_energy_z_w25: {features['bl_energy_z_w25']:.4f}")
    print(f"  bl_mean_delta_w25: {features['bl_mean_delta_w25']:.4f}")
    print(f"  bl_median_delta_w25: {features['bl_median_delta_w25']:.4f}")
    print(f"  bl_wasserstein_full: {features['bl_wasserstein_full']:.4f}")
    print(f"  bl_energy_full: {features['bl_energy_full']:.4f}")
    
    # Assertions
    assert features['bl_wasserstein_w25'] > 0, "Wasserstein should be positive"
    assert features['bl_energy_w25'] > 0, "Energy should be positive"
    assert features['bl_wasserstein_z_w25'] > 0, "Standardized Wasserstein should be positive"
    assert features['bl_energy_z_w25'] > 0, "Standardized Energy should be positive"
    assert 0.5 < features['bl_mean_delta_w25'] < 1.5, "Mean delta should be around 1.0"
    assert 0.5 < features['bl_median_delta_w25'] < 1.5, "Median delta should be around 1.0"
    
    # Check signed DoD features exist
    assert 'bl_dod_diff_wassz_enez_w25' in features, "Signed DoD feature should exist"
    assert np.isfinite(features['bl_dod_diff_wassz_enez_w25']), "Signed DoD should be finite"
    
    # Check ratio/ndiff/logratio DoD features
    print(f"  bl_dod_ratio_wassz_enez_w25: {features.get('bl_dod_ratio_wassz_enez_w25', np.nan):.4f}")
    print(f"  bl_dod_ndiff_wassz_enez_w25: {features.get('bl_dod_ndiff_wassz_enez_w25', np.nan):.4f}")
    print(f"  bl_dod_logratio_wassz_enez_w25: {features.get('bl_dod_logratio_wassz_enez_w25', np.nan):.4f}")
    assert np.isfinite(features['bl_dod_ratio_wassz_enez_w25']), "Ratio DoD should be finite"
    assert np.isfinite(features['bl_dod_ndiff_wassz_enez_w25']), "Ndiff DoD should be finite"
    assert np.isfinite(features['bl_dod_logratio_wassz_enez_w25']), "Logratio DoD should be finite"
    
    # Check localization features
    print(f"  bl_loc_diff_wassz_w25: {features.get('bl_loc_diff_wassz_w25', np.nan):.4f}")
    print(f"  bl_loc_ratio_wassz_w25: {features.get('bl_loc_ratio_wassz_w25', np.nan):.4f}")
    assert np.isfinite(features.get('bl_loc_diff_wassz_w25', np.nan)), "Localization diff should be finite"
    assert np.isfinite(features.get('bl_loc_ratio_wassz_w25', np.nan)), "Localization ratio should be finite"
    
    print("\n✓ Test 1 passed")
    
    # Test 2: Short segments (window larger than data)
    print("\nTest 2: Short segments (window > len)")
    x0_short = np.random.randn(10)
    x1_short = np.random.randn(10)
    features_short = compute_boundary_dist_features(x0_short, x1_short, windows=(25, 50))
    
    print(f"  bl_wasserstein_w25: {features_short['bl_wasserstein_w25']}")
    print(f"  bl_wasserstein_w50: {features_short['bl_wasserstein_w50']}")
    
    # When window > len, should still compute (using full segment)
    assert not np.isnan(features_short['bl_wasserstein_w25']), "Should compute with available data"
    
    print("\n✓ Test 2 passed")
    
    # Test 3: Very short segments (< 2 samples)
    print("\nTest 3: Very short segments (< 2 samples)")
    x0_tiny = np.array([1.0])
    x1_tiny = np.array([2.0])
    features_tiny = compute_boundary_dist_features(x0_tiny, x1_tiny, windows=(25,))
    
    print(f"  bl_wasserstein_w25: {features_tiny['bl_wasserstein_w25']}")
    assert np.isnan(features_tiny['bl_wasserstein_w25']), "Should return NaN for insufficient data"
    
    print("\n✓ Test 3 passed")
    
    # Test 4: With NaNs in data
    print("\nTest 4: Data with NaNs")
    x0_nan = np.array([1, 2, np.nan, 3, 4, 5, np.nan, 6, 7, 8] * 10)
    x1_nan = np.array([2, 3, np.nan, 4, 5, 6, np.nan, 7, 8, 9] * 10)
    features_nan = compute_boundary_dist_features(x0_nan, x1_nan, windows=(25,))
    
    print(f"  bl_wasserstein_w25: {features_nan['bl_wasserstein_w25']:.4f}")
    assert not np.isnan(features_nan['bl_wasserstein_w25']), "Should handle NaNs gracefully"
    
    print("\n✓ Test 4 passed")
    
    # Test 5: Heavy-tail regime change
    print("\nTest 5: Heavy-tail regime (x1 has upper tail spikes)")
    np.random.seed(123)
    x0_norm = np.random.randn(200)
    x1_tail = np.random.randn(200)
    # Add large positive spikes to top 10% of x1
    spike_indices = np.random.choice(len(x1_tail), size=20, replace=False)
    x1_tail[spike_indices] += np.random.uniform(3, 6, size=20)
    
    features_tail = compute_boundary_dist_features(x0_norm, x1_tail, windows=(25,))
    
    print(f"  bl_wasserstein_w25: {features_tail['bl_wasserstein_w25']:.4f}")
    print(f"  bl_tail_wasserstein_q95_upper_w25: {features_tail['bl_tail_wasserstein_q95_upper_w25']:.4f}")
    print(f"  bl_tail_energy_q95_upper_w25: {features_tail['bl_tail_energy_q95_upper_w25']:.4f}")
    print(f"  bl_tail_p_hi_delta_q95_w25: {features_tail['bl_tail_p_hi_delta_q95_w25']:.4f}")
    print(f"  bl_tail_mean_excess_hi_delta_q95_w25: {features_tail['bl_tail_mean_excess_hi_delta_q95_w25']:.4f}")
    print(f"  bl_tail_wins_wasserstein_w25: {features_tail['bl_tail_wins_wasserstein_w25']:.4f}")
    
    # Tail-specific distances should be larger than baseline
    assert features_tail['bl_tail_wasserstein_q95_upper_w25'] > 0, "Upper tail distance should be positive"
    assert features_tail['bl_tail_p_hi_delta_q95_w25'] > 0, "Upper tail probability should increase"
    
    # Winsorized should be smaller (less influenced by outliers)
    if not np.isnan(features_tail['bl_tail_wins_wasserstein_w25']):
        assert features_tail['bl_tail_wins_wasserstein_w25'] < features_tail['bl_wasserstein_w25'], \
            "Winsorized should reduce outlier influence"
    
    # Check signed DoD for winsorized features
    wwz = features_tail.get('bl_tail_wins_wasserstein_z_w25', np.nan)
    wez = features_tail.get('bl_tail_wins_energy_z_w25', np.nan)
    if np.isfinite(wwz) and np.isfinite(wez):
        assert 'bl_dod_diff_wins_wassz_wins_enez_w25' in features_tail, "Winsorized signed DoD should exist"
        assert np.isfinite(features_tail['bl_dod_diff_wins_wassz_wins_enez_w25']), "Winsorized signed DoD should be finite"
    
    # Check localization features for tail case
    print(f"  bl_loc_diff_wassz_w25: {features_tail.get('bl_loc_diff_wassz_w25', np.nan):.4f}")
    print(f"  bl_loc_ratio_wins_wassz_w25: {features_tail.get('bl_loc_ratio_wins_wassz_w25', np.nan):.4f}")
    
    print("\n✓ Test 5 passed")
    
    # Test 6: DoD (distance-of-distance) features
    print("\nTest 6: Distance-of-distance (DoD) features")
    np.random.seed(456)
    x0_dod = np.random.randn(300)
    x1_dod = np.random.randn(300) + 1.5  # Clear shift
    
    features_dod = compute_boundary_dist_features(x0_dod, x1_dod, windows=(25, 50, 100, 250))
    
    # Check DoD statistics for wasserstein_z
    print(f"  bl_dod_wasserstein_z_slope_logw: {features_dod.get('bl_dod_wasserstein_z_slope_logw', np.nan):.4f}")
    print(f"  bl_dod_wasserstein_z_range: {features_dod.get('bl_dod_wasserstein_z_range', np.nan):.4f}")
    print(f"  bl_dod_wasserstein_z_cv: {features_dod.get('bl_dod_wasserstein_z_cv', np.nan):.4f}")
    
    # Check cross-metric agreement
    print(f"  bl_dod_absdiff_wassz_enez_w25: {features_dod.get('bl_dod_absdiff_wassz_enez_w25', np.nan):.4f}")
    print(f"  bl_dod_reldiff_wassz_enez_w25: {features_dod.get('bl_dod_reldiff_wassz_enez_w25', np.nan):.4f}")
    
    # Check outlier sensitivity
    print(f"  bl_dod_outlier_sens_wassz_w25: {features_dod.get('bl_dod_outlier_sens_wassz_w25', np.nan):.4f}")
    
    # Check tail-vs-bulk consistency
    print(f"  bl_dod_tailbulk_wassz_q90_both_w25: {features_dod.get('bl_dod_tailbulk_wassz_q90_both_w25', np.nan):.4f}")
    
    # Validate DoD features are computed
    assert np.isfinite(features_dod['bl_dod_wasserstein_z_slope_logw']), "DoD slope should be finite"
    assert np.isfinite(features_dod['bl_dod_wasserstein_z_range']), "DoD range should be finite"
    assert np.isfinite(features_dod['bl_dod_absdiff_wassz_enez_w25']), "Cross-metric agreement should be finite"
    
    # Count DoD features
    dod_keys = [k for k in features_dod.keys() if 'dod' in k]
    print(f"  Total DoD features: {len(dod_keys)}")
    
    print("\n✓ Test 6 passed")
    
    # Test 7: DoD with shuffled windows (test sorting fix)
    print("\nTest 7: DoD features with shuffled window order")
    np.random.seed(789)
    x0_shuffle = np.random.randn(300)
    x1_shuffle = np.random.randn(300) + 1.0
    
    # Intentionally pass windows in non-sorted order
    features_shuffle = compute_boundary_dist_features(x0_shuffle, x1_shuffle, windows=(250, 25, 100, 50))
    
    # Verify DoD delta and ratio are consistent with sorted interpretation
    # delta_wmin_wmax should be: distance[w250] - distance[w25]
    # ratio_wmin_wmax should be: distance[w250] / distance[w25]
    d25 = features_shuffle['bl_wasserstein_z_w25']
    d250 = features_shuffle['bl_wasserstein_z_w250']
    
    delta_computed = features_shuffle['bl_dod_wasserstein_z_delta_wmin_wmax']
    ratio_computed = features_shuffle['bl_dod_wasserstein_z_ratio_wmin_wmax']
    
    print(f"  bl_wasserstein_z_w25: {d25:.4f}")
    print(f"  bl_wasserstein_z_w250: {d250:.4f}")
    print(f"  bl_dod_wasserstein_z_delta_wmin_wmax: {delta_computed:.4f}")
    print(f"  Expected delta (w250 - w25): {(d250 - d25):.4f}")
    print(f"  bl_dod_wasserstein_z_ratio_wmin_wmax: {ratio_computed:.4f}")
    print(f"  Expected ratio (w250 / w25): {(d250 / (d25 + 1e-12)):.4f}")
    
    # Assertions to verify correctness
    assert np.isfinite(delta_computed), "Delta should be finite"
    assert np.isfinite(ratio_computed), "Ratio should be finite"
    assert np.isclose(delta_computed, d250 - d25, atol=1e-8), "Delta should match sorted window interpretation"
    assert np.isclose(ratio_computed, d250 / (d25 + 1e-12), atol=1e-8), "Ratio should match sorted window interpretation"
    
    print("\n✓ Test 7 passed")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)