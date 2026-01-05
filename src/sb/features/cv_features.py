"""
Coefficient of Variation (CV) features - The "magic" feature from winning solutions.

Multiple top teams independently discovered that CV acts as a regime detector,
providing 5-10% AUC boost. The data has different generation regimes that need
to be identified via CV thresholds.

Key insight: CV = std / mean provides unit-less comparison across series,
enabling the model to identify "easy negative" regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_cv_features(x0: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """
    Compute Coefficient of Variation features.
    
    The CV is std/mean, providing a unit-less measure of relative variability.
    Winners found this creates natural regime boundaries in the data.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        
    Returns:
        Dictionary of CV features
    """
    features = {}
    
    # Combine for global CV
    x_all = np.concatenate([x0, x1])
    
    # Basic CV for each segment
    def safe_cv(x):
        """Compute CV safely, handling zero mean."""
        if len(x) == 0:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if np.abs(mean) < 1e-10:
            # Mean too close to zero, use alternative
            return std  # Just return std as proxy
        return std / np.abs(mean)
    
    cv_global = safe_cv(x_all)
    cv_pre = safe_cv(x0)
    cv_post = safe_cv(x1)
    
    features['cv_global'] = cv_global
    features['cv_pre'] = cv_pre
    features['cv_post'] = cv_post
    
    # CV changes (key signals)
    features['cv_diff'] = cv_post - cv_pre
    features['cv_abs_diff'] = np.abs(cv_post - cv_pre)
    features['cv_ratio'] = cv_post / (cv_pre + 1e-8)
    features['cv_log_ratio'] = np.log((cv_post + 1e-8) / (cv_pre + 1e-8))
    
    # Regime indicators (from winning solutions)
    # Winners found thresholds at ~0.03, 0.04, 0.20, 0.27
    features['cv_global_low'] = float(cv_global < 0.05)  # Very low CV regime
    features['cv_global_easy_neg'] = float(0.198 <= cv_global <= 0.20)  # "Easy negatives" regime
    features['cv_global_high'] = float(cv_global > 0.27)  # High CV regime
    
    # CV stability (is it consistent pre/post?)
    features['cv_stability'] = 1.0 / (np.abs(cv_post - cv_pre) + 1e-8)
    
    # Mean-normalized differences (like CV but for differences)
    mean_global = np.mean(x_all)
    if np.abs(mean_global) > 1e-10:
        features['mean_norm_diff'] = (np.mean(x1) - np.mean(x0)) / np.abs(mean_global)
    else:
        features['mean_norm_diff'] = 0.0
    
    # NOTE: Rolling CV features removed - too slow for 10K series
    # Winners didn't use rolling CV, just global/pre/post
    features['cv_rolling_pre_mean'] = cv_pre
    features['cv_rolling_pre_std'] = 0.0
    features['cv_rolling_post_mean'] = cv_post
    features['cv_rolling_post_std'] = 0.0
    
    # Signal-to-noise ratio (alternative formulation)
    features['snr_pre'] = np.abs(np.mean(x0)) / (np.std(x0) + 1e-8)
    features['snr_post'] = np.abs(np.mean(x1)) / (np.std(x1) + 1e-8)
    features['snr_diff'] = features['snr_post'] - features['snr_pre']
    features['snr_ratio'] = features['snr_post'] / (features['snr_pre'] + 1e-8)
    
    # Interaction with variance (the magic feature from Chinese team)
    # magic = (std^2) / mean = std * cv
    std_global = np.std(x_all)
    mean_global_abs = np.abs(mean_global)
    if mean_global_abs > 1e-10:
        features['cv_std_interaction'] = std_global * cv_global  # std^2 / mean
        features['cv_var_over_mean'] = (std_global ** 2) / mean_global_abs
    else:
        features['cv_std_interaction'] = 0.0
        features['cv_var_over_mean'] = 0.0
    
    return features


def compute_cv_features_multiscale(x0: np.ndarray, x1: np.ndarray, 
                                    windows=[50, 100, 250]) -> Dict[str, float]:
    """
    Compute CV features at multiple scales (boundary windows).
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        windows: List of window sizes
        
    Returns:
        Dictionary of CV features at multiple scales
    """
    features = {}
    
    # Full-scale features
    full_feats = compute_cv_features(x0, x1)
    for k, v in full_feats.items():
        features[f'{k}_full'] = v
    
    # Multi-scale features (boundary-focused)
    for w in windows:
        # Last w points of x0, first w points of x1
        x0_w = x0[-w:] if len(x0) >= w else x0
        x1_w = x1[:w] if len(x1) >= w else x1
        
        if len(x0_w) > 0 and len(x1_w) > 0:
            w_feats = compute_cv_features(x0_w, x1_w)
            for k, v in w_feats.items():
                features[f'{k}_w{w}'] = v
    
    return features
