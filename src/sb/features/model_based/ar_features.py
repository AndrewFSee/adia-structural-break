"""
Predictability-shift features based on AR(1) dynamics.

These features measure changes in time series predictability across the break:
- AR(1) coefficient (phi): persistence/mean-reversion strength
- Residual variance: unpredictable component
- One-step-ahead RMSE: forecast accuracy

If predictability changes, it suggests a structural break in the generating process.
"""

import numpy as np
from typing import Dict, Tuple


def fit_ar1_ols(x: np.ndarray) -> Tuple[float, float]:
    """
    Fit AR(1) model using ordinary least squares.
    
    Model: x[t] = phi * x[t-1] + epsilon[t]
    
    Args:
        x: Time series values
        
    Returns:
        (phi, resid_var): AR(1) coefficient and residual variance
    """
    if len(x) < 3:
        return 0.0, 0.0
    
    # Prepare lagged data
    y = x[1:]      # x[t]
    X = x[:-1]     # x[t-1]
    
    # OLS: phi = cov(x[t], x[t-1]) / var(x[t-1])
    cov_xy = np.mean((X - X.mean()) * (y - y.mean()))
    var_x = np.var(X, ddof=1)
    
    if var_x < 1e-10:
        return 0.0, np.var(y, ddof=1)
    
    phi = cov_xy / var_x
    
    # Compute residuals
    predictions = phi * X
    residuals = y - predictions
    resid_var = np.var(residuals, ddof=1)
    
    return phi, resid_var


def fit_ar1_yule_walker(x: np.ndarray) -> Tuple[float, float]:
    """
    Fit AR(1) model using Yule-Walker equations.
    
    This is equivalent to OLS for AR(1) but based on autocorrelation.
    More numerically stable for near-unit-root processes.
    
    Args:
        x: Time series values
        
    Returns:
        (phi, resid_var): AR(1) coefficient and residual variance
    """
    if len(x) < 3:
        return 0.0, 0.0
    
    # Demean
    x_centered = x - np.mean(x)
    
    # Compute lag-0 and lag-1 autocovariances
    gamma_0 = np.var(x_centered, ddof=1)
    
    if gamma_0 < 1e-10:
        return 0.0, gamma_0
    
    gamma_1 = np.mean(x_centered[:-1] * x_centered[1:])
    
    # Yule-Walker: phi = gamma_1 / gamma_0
    phi = gamma_1 / gamma_0
    
    # Residual variance: sigma^2 = gamma_0 * (1 - phi^2)
    resid_var = gamma_0 * (1.0 - phi**2)
    resid_var = max(0.0, resid_var)  # Ensure non-negative
    
    return phi, resid_var


def compute_walk_forward_rmse(x: np.ndarray, phi: float) -> float:
    """
    Compute walk-forward one-step-ahead RMSE for AR(1) model.
    
    For each time point t, predict x[t] using x[t-1] and measure error.
    This gives a realistic measure of forecast accuracy.
    
    Args:
        x: Time series values
        phi: AR(1) coefficient
        
    Returns:
        RMSE of one-step-ahead forecasts
    """
    if len(x) < 3:
        return 0.0
    
    # One-step-ahead predictions: x_hat[t] = phi * x[t-1]
    predictions = phi * x[:-1]
    actuals = x[1:]
    
    errors = actuals - predictions
    rmse = np.sqrt(np.mean(errors**2))
    
    return rmse


def ar1_features(pre: np.ndarray, post: np.ndarray, 
                method: str = "yule-walker") -> Dict[str, float]:
    """
    Compute AR(1) predictability-shift features.
    
    Compares autoregressive dynamics between pre-break and post-break segments:
    - AR(1) coefficient (phi): persistence/mean-reversion
    - Residual variance: unpredictable noise
    - One-step-ahead RMSE: forecast accuracy
    
    Args:
        pre: Pre-break values
        post: Post-break values
        method: "ols" or "yule-walker" (default: yule-walker)
        
    Returns:
        Dictionary with AR(1) features
    """
    # Choose fitting method
    if method == "ols":
        fit_func = fit_ar1_ols
    else:
        fit_func = fit_ar1_yule_walker
    
    # Fit AR(1) to pre and post
    phi_pre, resid_var_pre = fit_func(pre)
    phi_post, resid_var_post = fit_func(post)
    
    # Compute walk-forward RMSEs
    rmse_pre = compute_walk_forward_rmse(pre, phi_pre)
    rmse_post = compute_walk_forward_rmse(post, phi_post)
    
    return {
        # AR(1) coefficients
        "ar1_phi_pre": phi_pre,
        "ar1_phi_post": phi_post,
        "delta_ar1_phi": abs(phi_post - phi_pre),
        
        # Residual variances
        "ar1_resid_var_pre": resid_var_pre,
        "ar1_resid_var_post": resid_var_post,
        "delta_resid_var": abs(resid_var_post - resid_var_pre),
        
        # Forecast accuracy
        "ar1_rmse_pre": rmse_pre,
        "ar1_rmse_post": rmse_post,
        "delta_rmse": abs(rmse_post - rmse_pre),
    }


def ar1_phi_shift(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Quick function: absolute change in AR(1) coefficient.
    
    Args:
        pre: Pre-break values
        post: Post-break values
        
    Returns:
        Absolute change in AR(1) phi
    """
    phi_pre, _ = fit_ar1_yule_walker(pre)
    phi_post, _ = fit_ar1_yule_walker(post)
    return abs(phi_post - phi_pre)


def ar1_resid_var_shift(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Quick function: absolute change in AR(1) residual variance.
    
    Args:
        pre: Pre-break values
        post: Post-break values
        
    Returns:
        Absolute change in residual variance
    """
    _, var_pre = fit_ar1_yule_walker(pre)
    _, var_post = fit_ar1_yule_walker(post)
    return abs(var_post - var_pre)


def ar1_rmse_shift(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Quick function: absolute change in one-step-ahead RMSE.
    
    Args:
        pre: Pre-break values
        post: Post-break values
        
    Returns:
        Absolute change in forecast RMSE
    """
    phi_pre, _ = fit_ar1_yule_walker(pre)
    phi_post, _ = fit_ar1_yule_walker(post)
    
    rmse_pre = compute_walk_forward_rmse(pre, phi_pre)
    rmse_post = compute_walk_forward_rmse(post, phi_post)
    
    return abs(rmse_post - rmse_pre)
