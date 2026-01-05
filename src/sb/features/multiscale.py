"""
Multi-scale feature extraction.

Computes the SAME features at different window sizes around the break boundary.
This captures both global and local structural changes without adding new feature families.

Why multi-scale helps out-of-sample:
- Some breaks are immediate (captured by small windows)
- Some breaks are gradual (captured by large windows)
- Ensemble of scales is more robust than single scale
"""

import numpy as np
from typing import Dict
from .. import config, preprocessing
from . import dist, dynamics
from .model_based import ar_features


def compute_windowed_features(
    x0: np.ndarray,
    x1: np.ndarray,
    window_size: int,
    suffix: str,
    include_spectral: bool = False
) -> Dict[str, float]:
    """
    Compute features on a window near the break boundary.
    
    Takes the LAST window_size points from x0 and FIRST window_size points from x1.
    This focuses on the region immediately around the presumed break.
    
    Args:
        x0: Pre-break values (full)
        x1: Post-break values (full)
        window_size: Number of points to take from each side
        suffix: String to append to feature names (e.g., "_w50")
        
    Returns:
        Dictionary of windowed features with suffixed names
    """
    # Extract windows near boundary
    x0_window = x0[-window_size:] if len(x0) >= window_size else x0
    x1_window = x1[:window_size] if len(x1) >= window_size else x1
    
    # Need minimum data for meaningful features
    if len(x0_window) < 10 or len(x1_window) < 10:
        # Return NaN features if window too small
        return {
            f"delta_q_slope{suffix}": np.nan,
            f"median_shift{suffix}": np.nan,
            f"iqr_ratio{suffix}": np.nan,
            f"delta_entropy{suffix}": np.nan,
            f"vol_slope_post{suffix}": np.nan,
            f"delta_kalman_var{suffix}": np.nan,
            f"energy{suffix}": np.nan,
            f"wasserstein{suffix}": np.nan,
            f"q10_delta{suffix}": np.nan,
            f"q50_delta{suffix}": np.nan,
            f"q90_delta{suffix}": np.nan,
            f"mad_ratio{suffix}": np.nan,
            f"iqr_ratio_robust{suffix}": np.nan,
            f"acf1_shift{suffix}": np.nan,
            f"ar1_phi_pre{suffix}": np.nan,
            f"ar1_phi_post{suffix}": np.nan,
            f"delta_ar1_phi{suffix}": np.nan,
            f"ar1_resid_var_pre{suffix}": np.nan,
            f"ar1_resid_var_post{suffix}": np.nan,
            f"delta_resid_var{suffix}": np.nan,
            f"ar1_rmse_pre{suffix}": np.nan,
            f"ar1_rmse_post{suffix}": np.nan,
            f"delta_rmse{suffix}": np.nan,
        }
    
    # Apply robust scaling
    x0_scaled = preprocessing.robust_scale(x0_window)
    x1_scaled = preprocessing.robust_scale(x1_window)
    
    # Compute same features as base
    features = {}
    
    # Distribution shape features
    quant_feats = dist.quantile_features(x0_scaled, x1_scaled)
    for k, v in quant_feats.items():
        features[f"{k}{suffix}"] = v
    
    features[f"delta_entropy{suffix}"] = dist.entropy_change(x0_scaled, x1_scaled)
    
    # Dynamics features
    vol_feats = dynamics.volatility_features(x0_scaled, x1_scaled)
    for k, v in vol_feats.items():
        features[f"{k}{suffix}"] = v
    
    # Two-sample statistics (boundary-focused)
    features[f"energy{suffix}"] = dist.energy_distance_1d(x0_scaled, x1_scaled)
    features[f"wasserstein{suffix}"] = dist.wasserstein_1d(x0_scaled, x1_scaled)
    
    # Quantile deltas
    q_deltas = dist.quantile_deltas(x0_scaled, x1_scaled, qs=[0.1, 0.5, 0.9])
    for k, v in q_deltas.items():
        features[f"{k}{suffix}"] = v
    
    # Scale shift
    scale_feats = dist.scale_shift(x0_scaled, x1_scaled)
    features[f"mad_ratio{suffix}"] = scale_feats["mad_ratio"]
    features[f"iqr_ratio_robust{suffix}"] = scale_feats["iqr_ratio"]
    
    # Autocorrelation shift
    features[f"acf1_shift{suffix}"] = dist.acf1_shift(x0_window, x1_window)
    
    # AR(1) predictability features
    ar_feats = ar_features.ar1_features(x0_window, x1_window)
    for k, v in ar_feats.items():
        features[f"{k}{suffix}"] = v
    
    # Spectral features (optional, deltas only to reduce feature count)
    if include_spectral:
        from . import spectral
        spec_feats = spectral.spectral_features_deltas_only(x0_window, x1_window)
        for k, v in spec_feats.items():
            features[f"{k}{suffix}"] = v
    
    return features


def compute_multiscale_features(
    x0: np.ndarray, 
    x1: np.ndarray,
    include_spectral: bool = False,
    include_wavelet: bool = False
) -> Dict[str, float]:
    """
    Compute features at multiple scales:
    - Full segments (original features)
    - Boundary-focused windows of different sizes
    
    This provides both global and local perspectives on the break.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        include_spectral: Whether to include spectral features at each scale
        include_wavelet: Whether to include wavelet features (full + one boundary window)
        
    Returns:
        Dictionary with all features across all scales
    """
    features = {}
    
    # Scale 1: Full segments (baseline features)
    # Apply robust scaling
    x0_scaled_full = preprocessing.robust_scale(x0)
    x1_scaled_full = preprocessing.robust_scale(x1)
    
    # Distribution shape
    base_feats = dist.quantile_features(x0_scaled_full, x1_scaled_full)
    features.update(base_feats)
    features["delta_entropy"] = dist.entropy_change(x0_scaled_full, x1_scaled_full)
    
    # Dynamics
    vol_feats = dynamics.volatility_features(x0_scaled_full, x1_scaled_full)
    features.update(vol_feats)
    
    # Two-sample statistics (full scale)
    features["energy"] = dist.energy_distance_1d(x0_scaled_full, x1_scaled_full)
    features["wasserstein"] = dist.wasserstein_1d(x0_scaled_full, x1_scaled_full)
    
    # Quantile deltas
    q_deltas = dist.quantile_deltas(x0_scaled_full, x1_scaled_full, qs=[0.1, 0.5, 0.9])
    features.update(q_deltas)
    
    # Scale shift
    scale_feats = dist.scale_shift(x0_scaled_full, x1_scaled_full)
    features["mad_ratio"] = scale_feats["mad_ratio"]
    features["iqr_ratio_robust"] = scale_feats["iqr_ratio"]
    
    # Autocorrelation shift
    features["acf1_shift"] = dist.acf1_shift(x0, x1)
    
    # AR(1) predictability features (full scale)
    ar_feats = ar_features.ar1_features(x0, x1)
    features.update(ar_feats)
    
    # Spectral features (full scale, optional - includes base + v2)
    if include_spectral:
        from . import spectral
        spec_feats = spectral.spectral_features_all(x0, x1, include_v2=True)
        features.update(spec_feats)
    
    # Wavelet features (full scale, optional)
    if include_wavelet:
        from . import wavelet
        wav_feats = wavelet.wavelet_features(x0, x1)
        features.update(wav_feats)
    
    # Scale 2+: Windowed features around boundary (deltas only for spectral)
    for window_size in config.MULTI_SCALE_WINDOWS:
        suffix = f"_w{window_size}"
        windowed_feats = compute_windowed_features(x0, x1, window_size, suffix, include_spectral=include_spectral)
        features.update(windowed_feats)
    
    # Add one boundary-focused wavelet window (largest window for stability)
    if include_wavelet and len(config.MULTI_SCALE_WINDOWS) > 0:
        # Use largest window size for best DWT stability
        boundary_window = max(config.MULTI_SCALE_WINDOWS)
        suffix = f"_w{boundary_window}"
        
        # Extract boundary windows
        x0_window = x0[-boundary_window:] if len(x0) >= boundary_window else x0
        x1_window = x1[:boundary_window] if len(x1) >= boundary_window else x1
        
        # Compute wavelet features on boundary window
        wav_boundary_feats = wavelet.wavelet_features(x0_window, x1_window)
        for k, v in wav_boundary_feats.items():
            features[f"{k}{suffix}"] = v
    
    return features
