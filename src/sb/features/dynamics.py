"""
Day 2 features: Transition dynamics (your edge).
Captures how the series evolves after the break, not just static differences.
"""

import numpy as np
import pandas as pd
from .. import config


def rolling_var_slope(x: np.ndarray, window: int = None) -> float:
    """
    Post-break volatility response.
    
    Instead of comparing variance levels, measure how variance *evolves*
    after the break. This captures delayed regime effects.
    
    Why this matters:
    - Captures delayed regime effects
    - Very underused in literature
    - Hard to Spearman-match
    
    Args:
        x: Post-break values (x1)
        window: Rolling window size (default from config)
        
    Returns:
        Absolute slope of rolling variance over time
    """
    if window is None:
        window = config.ROLLING_WINDOW
    
    if len(x) < window * config.MIN_WINDOW_MULTIPLIER:
        return 0.0
    
    rv = pd.Series(x).rolling(window).var().dropna()
    
    if len(rv) < 2:
        return 0.0
    
    t = np.arange(len(rv))
    slope = np.polyfit(t, rv.values, 1)[0]
    
    return abs(slope)


def kalman_level_variance(x: np.ndarray) -> float:
    """
    Simple Kalman response signal (no heavy tuning).
    
    Local level model, but we care about change in inferred noise,
    not likelihood. Acts like a proxy for process noise shift.
    
    This is a lightweight Kalman approximation using first differences
    as a proxy for innovation variance.
    
    Args:
        x: Input values
        
    Returns:
        Variance of first differences (process noise proxy)
    """
    x = np.asarray(x)
    
    if len(x) < 2:
        return 0.0
    
    diffs = np.diff(x, n=config.DIFF_ORDER)
    return np.var(diffs)


def kalman_variance_change(x0: np.ndarray, x1: np.ndarray) -> float:
    """
    Change in Kalman-approximated process variance.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        
    Returns:
        Absolute change in process variance proxy
    """
    q0 = kalman_level_variance(x0)
    q1 = kalman_level_variance(x1)
    
    return abs(q1 - q0)


def volatility_features(x0: np.ndarray, x1: np.ndarray) -> dict:
    """
    Combined volatility-based features for Day 2.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        
    Returns:
        Dictionary with:
        - vol_slope_post: Post-break volatility evolution
        - delta_kalman_var: Change in process noise proxy
    """
    return {
        "vol_slope_post": rolling_var_slope(x1),
        "delta_kalman_var": kalman_variance_change(x0, x1),
    }
