"""
Preprocessing utilities for robust normalization.
"""

import numpy as np
from . import config


def robust_scale(x: np.ndarray) -> np.ndarray:
    """
    Robust normalization using median and MAD (Median Absolute Deviation).
    
    This avoids scale-driven correlation with classic tests by using
    robust statistics instead of mean/std.
    
    Args:
        x: Input array
        
    Returns:
        Scaled array: (x - median) / MAD
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + config.MAD_EPSILON
    return (x - med) / mad


def clip_outliers(x: np.ndarray, n_mad: float = 5.0) -> np.ndarray:
    """
    Clip extreme outliers based on MAD.
    
    Args:
        x: Input array
        n_mad: Number of MAD units to use as threshold
        
    Returns:
        Clipped array
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + config.MAD_EPSILON
    
    lower = med - n_mad * mad
    upper = med + n_mad * mad
    
    return np.clip(x, lower, upper)


def remove_linear_trend(x: np.ndarray) -> np.ndarray:
    """
    Remove linear trend from series.
    
    Args:
        x: Input array
        
    Returns:
        Detrended array
    """
    if len(x) < 2:
        return x
    
    t = np.arange(len(x))
    coeffs = np.polyfit(t, x, 1)
    trend = np.polyval(coeffs, t)
    
    return x - trend
