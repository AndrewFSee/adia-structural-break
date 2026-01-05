"""
Day 1 features: Distribution shape (not mean/variance).
These are intentionally non-standard to avoid correlation with classic tests.
"""

import numpy as np
from typing import Dict
from .. import config


def quantile_features(x0: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """
    Distribution shape change features using quantiles.
    
    Captures shape change, not level. This decorrelates from:
    - KS test
    - t-test
    - Classic variance tests
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        
    Returns:
        Dictionary with three features:
        - delta_q_slope: Change in distribution slope (deformation)
        - median_shift: Robust location shift
        - iqr_ratio: Change in scale via interquartile range
    """
    qs = config.QUANTILES
    q0 = np.quantile(x0, qs)
    q1 = np.quantile(x1, qs)
    
    # Shape change, not level
    slope0 = q0[-1] - q0[0]  # 90th - 10th percentile
    slope1 = q1[-1] - q1[0]
    
    return {
        "delta_q_slope": abs(slope1 - slope0),
        "median_shift": abs(q1[2] - q0[2]),  # q1[2] is median (50th percentile)
        "iqr_ratio": (q1[3] - q1[1]) / (q0[3] - q0[1] + config.MAD_EPSILON),
    }


def entropy(x: np.ndarray, bins: int = None) -> float:
    """
    Shannon entropy of the empirical distribution.
    
    Cheap, effective, and decorrelates nicely from variance-based detectors.
    
    Args:
        x: Input values
        bins: Number of histogram bins (default from config)
        
    Returns:
        Entropy value (non-negative)
    """
    if bins is None:
        bins = config.ENTROPY_BINS
    
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zero bins
    
    return -np.sum(hist * np.log(hist))


def entropy_change(x0: np.ndarray, x1: np.ndarray) -> float:
    """
    Change in entropy between pre and post segments.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        
    Returns:
        Absolute entropy change
    """
    ent0 = entropy(x0)
    ent1 = entropy(x1)
    return abs(ent1 - ent0)


def energy_distance_1d(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Compute energy distance between two 1D distributions.
    
    Energy distance is a metric between distributions based on expected
    distances between observations. More sensitive to distribution shape
    changes than traditional two-sample tests.
    
    E(X,Y) = 2*E|X-Y| - E|X-X'| - E|Y-Y'|
    
    Args:
        pre: Pre-break values
        post: Post-break values
        
    Returns:
        Energy distance (non-negative scalar)
    """
    n_pre = len(pre)
    n_post = len(post)
    
    if n_pre == 0 or n_post == 0:
        return 0.0
    
    # E|X-Y|: expected distance between samples from different distributions
    cross_term = np.mean(np.abs(pre[:, None] - post[None, :]))
    
    # E|X-X'|: expected distance within pre
    if n_pre > 1:
        pre_term = np.mean(np.abs(pre[:, None] - pre[None, :]))
    else:
        pre_term = 0.0
    
    # E|Y-Y'|: expected distance within post
    if n_post > 1:
        post_term = np.mean(np.abs(post[:, None] - post[None, :]))
    else:
        post_term = 0.0
    
    energy = 2.0 * cross_term - pre_term - post_term
    
    return max(0.0, energy)  # Ensure non-negative due to numerical precision


def wasserstein_1d(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance).
    
    Wasserstein-1 distance is the area between CDFs. Fast and exact
    computation for 1D case using sorted arrays.
    
    Args:
        pre: Pre-break values
        post: Post-break values
        
    Returns:
        Wasserstein distance (non-negative scalar)
    """
    if len(pre) == 0 or len(post) == 0:
        return 0.0
    
    # Sort both arrays
    pre_sorted = np.sort(pre)
    post_sorted = np.sort(post)
    
    # Compute empirical CDFs at all unique points
    all_values = np.concatenate([pre_sorted, post_sorted])
    
    # Compute CDFs
    cdf_pre = np.searchsorted(pre_sorted, all_values, side='right') / len(pre)
    cdf_post = np.searchsorted(post_sorted, all_values, side='right') / len(post)
    
    # Wasserstein-1 is integral of |CDF1 - CDF2|
    # Approximate by summing over all points
    deltas = np.diff(all_values, prepend=all_values[0])
    wasserstein = np.sum(np.abs(cdf_pre - cdf_post) * deltas)
    
    return wasserstein


def quantile_deltas(pre: np.ndarray, post: np.ndarray, 
                   qs: list = None) -> Dict[str, float]:
    """
    Compute quantile differences at multiple levels.
    
    Returns differences for each quantile: |Q_post(p) - Q_pre(p)|
    Useful for detecting shifts at different parts of the distribution.
    
    Args:
        pre: Pre-break values
        post: Post-break values
        qs: Quantile levels (default: [0.1, 0.5, 0.9])
        
    Returns:
        Dictionary with quantile delta features
    """
    if qs is None:
        qs = [0.1, 0.5, 0.9]
    
    if len(pre) == 0 or len(post) == 0:
        return {f"q{int(q*100)}_delta": 0.0 for q in qs}
    
    q_pre = np.quantile(pre, qs)
    q_post = np.quantile(post, qs)
    
    deltas = np.abs(q_post - q_pre)
    
    return {f"q{int(q*100)}_delta": float(delta) 
            for q, delta in zip(qs, deltas)}


def scale_shift(pre: np.ndarray, post: np.ndarray) -> Dict[str, float]:
    """
    Compute scale shifts using robust measures (MAD and IQR).
    
    Returns both MAD ratio and IQR ratio to capture scale changes
    without being sensitive to outliers or location shifts.
    
    Args:
        pre: Pre-break values
        post: Post-break values
        
    Returns:
        Dictionary with 'mad_ratio' and 'iqr_ratio'
    """
    if len(pre) == 0 or len(post) == 0:
        return {"mad_ratio": 1.0, "iqr_ratio": 1.0}
    
    # MAD (Median Absolute Deviation)
    mad_pre = np.median(np.abs(pre - np.median(pre)))
    mad_post = np.median(np.abs(post - np.median(post)))
    
    mad_ratio = (mad_post + config.MAD_EPSILON) / (mad_pre + config.MAD_EPSILON)
    
    # IQR (Interquartile Range)
    q_pre = np.quantile(pre, [0.25, 0.75])
    q_post = np.quantile(post, [0.25, 0.75])
    
    iqr_pre = q_pre[1] - q_pre[0]
    iqr_post = q_post[1] - q_post[0]
    
    iqr_ratio = (iqr_post + config.MAD_EPSILON) / (iqr_pre + config.MAD_EPSILON)
    
    return {
        "mad_ratio": float(mad_ratio),
        "iqr_ratio": float(iqr_ratio)
    }


def acf1_shift(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Compute change in lag-1 autocorrelation.
    
    Measures change in short-term temporal dependence structure.
    Useful for detecting regime changes in time series dynamics.
    
    Args:
        pre: Pre-break values
        post: Post-break values
        
    Returns:
        Absolute difference in lag-1 autocorrelation
    """
    def compute_acf1(x: np.ndarray) -> float:
        """Helper to compute lag-1 autocorrelation."""
        if len(x) < 2:
            return 0.0
        
        # Demean
        x_centered = x - np.mean(x)
        
        # Variance
        var = np.sum(x_centered ** 2)
        
        if var < 1e-10:
            return 0.0
        
        # Lag-1 autocovariance
        acov1 = np.sum(x_centered[:-1] * x_centered[1:])
        
        # ACF = autocovariance / variance
        acf1 = acov1 / var
        
        return acf1
    
    acf1_pre = compute_acf1(pre)
    acf1_post = compute_acf1(post)
    
    return abs(acf1_post - acf1_pre)
