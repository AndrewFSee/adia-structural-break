"""
AR/Kalman features for structural break detection with strict leakage prevention.

CRITICAL ANTI-LEAKAGE RULES:
1. Preprocessing (standardization, winsorization) uses ONLY pre-segment statistics
2. AR/Kalman parameters estimated ONLY on pre-segment
3. Post-segment is evaluated using pre-segment parameters (break = distribution shift)
4. Window features must respect boundary (no cross-contamination)
5. Global transforms (rank normalization) must be fit INSIDE CV folds only

Common pitfalls marked with ⚠️ warnings in comments.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
import warnings


def split_pre_post(df_one_id: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a single series into pre-break and post-break segments.
    
    Args:
        df_one_id: DataFrame for one id with columns [value, period]
        
    Returns:
        pre_values: Pre-break segment (period==0)
        post_values: Post-break segment (period==1)
    """
    pre_values = df_one_id[df_one_id['period'] == 0]['value'].values
    post_values = df_one_id[df_one_id['period'] == 1]['value'].values
    
    return pre_values, post_values


def robust_preprocess(
    pre: np.ndarray,
    post: np.ndarray,
    demean: bool = True,
    standardize: bool = True,
    winsorize_quantiles: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Robust preprocessing using ONLY pre-segment statistics.
    
    ⚠️ ANTI-LEAKAGE: All statistics computed from PRE only!
    
    Args:
        pre: Pre-break values
        post: Post-break values
        demean: Whether to subtract pre-segment mean
        standardize: Whether to divide by pre-segment std
        winsorize_quantiles: If provided, clip to (lower, upper) quantiles of PRE
        
    Returns:
        pre_processed: Processed pre values
        post_processed: Processed post values (using PRE statistics!)
        stats: Dictionary of statistics used (for diagnostics)
    """
    if len(pre) < 2:
        # Too short, return as-is
        return pre.copy(), post.copy(), {}
    
    stats_dict = {}
    
    # Step 1: Winsorization using PRE quantiles only
    # ⚠️ Computing quantiles on POST would be leakage!
    if winsorize_quantiles is not None:
        lower_q, upper_q = winsorize_quantiles
        lower_val = np.quantile(pre, lower_q)
        upper_val = np.quantile(pre, upper_q)
        
        pre_clipped = np.clip(pre, lower_val, upper_val)
        post_clipped = np.clip(post, lower_val, upper_val)  # Apply PRE limits to POST
        
        stats_dict['winsorize_lower'] = lower_val
        stats_dict['winsorize_upper'] = upper_val
    else:
        pre_clipped = pre.copy()
        post_clipped = post.copy()
    
    # Step 2: De-mean using PRE mean only
    # ⚠️ Using post mean would be leakage!
    if demean:
        pre_mean = np.mean(pre_clipped)
        pre_demeaned = pre_clipped - pre_mean
        post_demeaned = post_clipped - pre_mean  # Use PRE mean on POST
        
        stats_dict['mean'] = pre_mean
    else:
        pre_demeaned = pre_clipped
        post_demeaned = post_clipped
    
    # Step 3: Standardize using PRE std only
    # ⚠️ Using post std would be leakage!
    if standardize:
        pre_std = np.std(pre_demeaned, ddof=1)
        
        if pre_std < 1e-8:
            # Constant series, skip standardization
            stats_dict['std'] = pre_std
            stats_dict['constant'] = True
            return pre_demeaned, post_demeaned, stats_dict
        
        pre_standardized = pre_demeaned / pre_std
        post_standardized = post_demeaned / pre_std  # Use PRE std on POST
        
        stats_dict['std'] = pre_std
        stats_dict['constant'] = False
    else:
        pre_standardized = pre_demeaned
        post_standardized = post_demeaned
    
    return pre_standardized, post_standardized, stats_dict


def fit_ar1_yule_walker(x: np.ndarray) -> Dict[str, float]:
    """
    Fit AR(1) model using Yule-Walker equations.
    
    Model: x[t] = phi * x[t-1] + epsilon[t]
    
    Args:
        x: Time series values
        
    Returns:
        Dictionary with phi, sigma_eps, resid_var, rmse
    """
    if len(x) < 3:
        return {
            'phi': np.nan,
            'sigma_eps': np.nan,
            'resid_var': np.nan,
            'rmse': np.nan
        }
    
    # Demean
    x_centered = x - np.mean(x)
    
    # Compute autocovariances
    gamma_0 = np.var(x_centered, ddof=1)
    
    if gamma_0 < 1e-10:
        # Constant series
        return {
            'phi': 0.0,
            'sigma_eps': 0.0,
            'resid_var': 0.0,
            'rmse': 0.0
        }
    
    gamma_1 = np.mean(x_centered[:-1] * x_centered[1:])
    
    # Yule-Walker estimate
    phi = gamma_1 / gamma_0
    
    # Residual variance
    resid_var = gamma_0 * (1.0 - phi**2)
    resid_var = max(0.0, resid_var)
    
    sigma_eps = np.sqrt(resid_var)
    
    # Walk-forward RMSE
    predictions = phi * x[:-1]
    actuals = x[1:]
    rmse = np.sqrt(np.mean((actuals - predictions)**2))
    
    return {
        'phi': phi,
        'sigma_eps': sigma_eps,
        'resid_var': resid_var,
        'rmse': rmse
    }


def ar1_cross_prediction_error(pre: np.ndarray, post: np.ndarray,
                               phi_pre: float) -> Dict[str, float]:
    """
    Apply PRE-fitted AR(1) model to POST segment.
    
    ⚠️ ANTI-LEAKAGE: Uses ONLY pre-segment parameters!
    This is the key break detection signal: does the pre-model fit the post data?
    
    Args:
        pre: Pre-break values (used for mean)
        post: Post-break values
        phi_pre: AR(1) coefficient from pre-segment
        
    Returns:
        Dictionary with cross-prediction errors
    """
    if len(post) < 2:
        return {
            'rmse_post_pred_by_pre': np.nan,
            'resid_var_post_pred_by_pre': np.nan
        }
    
    # ⚠️ Use PRE mean for predictions, not POST mean!
    pre_mean = np.mean(pre)
    
    # Center post using pre mean
    post_centered = post - pre_mean
    
    # One-step predictions using pre phi
    predictions = phi_pre * post_centered[:-1]
    actuals = post_centered[1:]
    
    residuals = actuals - predictions
    rmse = np.sqrt(np.mean(residuals**2))
    resid_var = np.var(residuals, ddof=1)
    
    return {
        'rmse_post_pred_by_pre': rmse,
        'resid_var_post_pred_by_pre': resid_var
    }


def simple_kalman_local_level(x: np.ndarray, q: float = None, r: float = None) -> Dict[str, float]:
    """
    Simple local level Kalman filter: x[t] = level[t] + noise
    
    State equation: level[t] = level[t-1] + w[t], w ~ N(0, q)
    Observation: x[t] = level[t] + v[t], v ~ N(0, r)
    
    ⚠️ ANTI-LEAKAGE: Estimate q, r from PRE only!
    
    Args:
        x: Time series values
        q: Process noise variance (estimated if None)
        r: Observation noise variance (estimated if None)
        
    Returns:
        Dictionary with innovations variance and one-step-ahead RMSE
    """
    if len(x) < 3:
        return {
            'kalman_innov_var': np.nan,
            'kalman_rmse': np.nan,
            'kalman_q': np.nan,
            'kalman_r': np.nan
        }
    
    # Simple heuristic parameter estimation if not provided
    if q is None or r is None:
        # Use variance decomposition heuristic
        total_var = np.var(x, ddof=1)
        diff_var = np.var(np.diff(x), ddof=1)
        
        # Heuristic: r = var(diff) / 2, q = total_var - r
        r_est = diff_var / 2.0
        q_est = max(total_var - r_est, r_est * 0.1)  # At least 10% of r
        
        q = q_est
        r = r_est
    
    # Kalman filter forward pass
    n = len(x)
    level = np.zeros(n)
    P = np.zeros(n)  # Error covariance
    innovations = np.zeros(n - 1)
    
    # Initialize
    level[0] = x[0]
    P[0] = r
    
    for t in range(1, n):
        # Prediction
        level_pred = level[t-1]
        P_pred = P[t-1] + q
        
        # Innovation
        innov = x[t] - level_pred
        innovations[t-1] = innov
        
        # Innovation variance
        S = P_pred + r
        
        # Kalman gain
        K = P_pred / S
        
        # Update
        level[t] = level_pred + K * innov
        P[t] = (1 - K) * P_pred
    
    # Statistics
    innov_var = np.var(innovations, ddof=1)
    rmse = np.sqrt(np.mean(innovations**2))
    
    return {
        'kalman_innov_var': innov_var,
        'kalman_rmse': rmse,
        'kalman_q': q,
        'kalman_r': r
    }


def kalman_cross_prediction_error(post: np.ndarray, q_pre: float, r_pre: float) -> Dict[str, float]:
    """
    Apply PRE-fitted Kalman parameters to POST segment.
    
    ⚠️ ANTI-LEAKAGE: Uses ONLY pre-segment parameters (q, r)!
    
    Args:
        post: Post-break values
        q_pre: Process noise from pre-segment
        r_pre: Observation noise from pre-segment
        
    Returns:
        Dictionary with post-segment innovations under pre-parameters
    """
    post_results = simple_kalman_local_level(post, q=q_pre, r=r_pre)
    
    return {
        'kalman_innov_var_post_pred_by_pre': post_results['kalman_innov_var'],
        'kalman_rmse_post_pred_by_pre': post_results['kalman_rmse']
    }


def extract_windowed_features(pre: np.ndarray, post: np.ndarray,
                              window_size: int, suffix: str) -> Dict[str, float]:
    """
    Extract AR/Kalman features from boundary-focused windows.
    
    ⚠️ ANTI-LEAKAGE: Take last N from PRE, first N from POST - no cross-contamination!
    
    Args:
        pre: Full pre-break segment
        post: Full post-break segment
        window_size: Number of points to take from each side
        suffix: Feature name suffix (e.g., "_w50")
        
    Returns:
        Dictionary of windowed features
    """
    # Extract windows near boundary
    # ⚠️ Must not mix: use LAST points of pre, FIRST points of post
    pre_window = pre[-window_size:] if len(pre) >= window_size else pre
    post_window = post[:window_size] if len(post) >= window_size else post
    
    if len(pre_window) < 3 or len(post_window) < 3:
        # Too small for meaningful features
        return {
            f'ar1_phi_pre{suffix}': np.nan,
            f'ar1_phi_post{suffix}': np.nan,
            f'delta_ar1_phi{suffix}': np.nan,
            f'ar1_rmse_pre{suffix}': np.nan,
            f'rmse_post_pred_by_pre{suffix}': np.nan,
            f'delta_ar1_rmse{suffix}': np.nan,
        }
    
    # Preprocess (use window's own pre stats)
    pre_proc, post_proc, _ = robust_preprocess(pre_window, post_window,
                                                demean=True, standardize=True)
    
    # AR features
    ar_pre = fit_ar1_yule_walker(pre_proc)
    ar_post = fit_ar1_yule_walker(post_proc)
    cross_pred = ar1_cross_prediction_error(pre_proc, post_proc, ar_pre['phi'])
    
    return {
        f'ar1_phi_pre{suffix}': ar_pre['phi'],
        f'ar1_phi_post{suffix}': ar_post['phi'],
        f'delta_ar1_phi{suffix}': abs(ar_post['phi'] - ar_pre['phi']),
        f'ar1_rmse_pre{suffix}': ar_pre['rmse'],
        f'rmse_post_pred_by_pre{suffix}': cross_pred['rmse_post_pred_by_pre'],
        f'delta_ar1_rmse{suffix}': abs(ar_post['rmse'] - ar_pre['rmse']),
    }


def compute_ar_kalman_features_single(pre: np.ndarray, post: np.ndarray,
                                     window_sizes: list = None,
                                     fast_mode: bool = False) -> Dict[str, float]:
    """
    Compute all AR/Kalman features for a single time series.
    
    ⚠️ ANTI-LEAKAGE GUARANTEED: All preprocessing uses ONLY pre-segment statistics.
    
    Args:
        pre: Pre-break values
        post: Post-break values
        window_sizes: List of window sizes for boundary features (default: [25, 50, 100])
        fast_mode: If True, skip some expensive features
        
    Returns:
        Dictionary of all features
    """
    if window_sizes is None:
        window_sizes = [25, 50, 100]
    
    features = {}
    
    # Handle edge cases
    if len(pre) < 3 or len(post) < 3:
        # Return NaN features
        base_nans = {
            'ar1_phi_pre': np.nan, 'ar1_phi_post': np.nan, 'delta_ar1_phi': np.nan,
            'ar1_resid_var_pre': np.nan, 'ar1_resid_var_post': np.nan,
            'log_resid_var_ratio': np.nan,
            'ar1_rmse_pre': np.nan, 'ar1_rmse_post': np.nan, 'delta_ar1_rmse': np.nan,
            'rmse_post_pred_by_pre': np.nan, 'resid_var_post_pred_by_pre': np.nan,
            'kalman_innov_var_pre': np.nan, 'kalman_innov_var_post': np.nan,
            'log_kalman_innov_ratio': np.nan,
            'kalman_rmse_pre': np.nan, 'kalman_rmse_post_pred_by_pre': np.nan,
        }
        features.update(base_nans)
        
        # Window features
        for w in window_sizes:
            suffix = f'_w{w}'
            features.update({
                f'ar1_phi_pre{suffix}': np.nan,
                f'delta_ar1_phi{suffix}': np.nan,
                f'delta_ar1_rmse{suffix}': np.nan,
            })
        
        return features
    
    # Robust preprocessing (ONLY using pre stats!)
    pre_proc, post_proc, prep_stats = robust_preprocess(
        pre, post,
        demean=True,
        standardize=True,
        winsorize_quantiles=(0.01, 0.99)  # Computed on PRE only!
    )
    
    # Check if series is constant
    if prep_stats.get('constant', False):
        # Return zeros/nans for constant series
        return {k: 0.0 if 'phi' in k or 'ratio' in k else np.nan 
                for k in features.keys()}
    
    # === AR(1) Features ===
    ar_pre = fit_ar1_yule_walker(pre_proc)
    ar_post = fit_ar1_yule_walker(post_proc)
    cross_pred = ar1_cross_prediction_error(pre_proc, post_proc, ar_pre['phi'])
    
    features['ar1_phi_pre'] = ar_pre['phi']
    features['ar1_phi_post'] = ar_post['phi']
    features['delta_ar1_phi'] = abs(ar_post['phi'] - ar_pre['phi'])
    
    features['ar1_resid_var_pre'] = ar_pre['resid_var']
    features['ar1_resid_var_post'] = ar_post['resid_var']
    
    # Log ratio (robust to scale)
    if ar_pre['resid_var'] > 1e-10 and ar_post['resid_var'] > 1e-10:
        features['log_resid_var_ratio'] = np.log(ar_post['resid_var'] / ar_pre['resid_var'])
    else:
        features['log_resid_var_ratio'] = 0.0
    
    features['ar1_rmse_pre'] = ar_pre['rmse']
    features['ar1_rmse_post'] = ar_post['rmse']
    features['delta_ar1_rmse'] = abs(ar_post['rmse'] - ar_pre['rmse'])
    
    features['rmse_post_pred_by_pre'] = cross_pred['rmse_post_pred_by_pre']
    features['resid_var_post_pred_by_pre'] = cross_pred['resid_var_post_pred_by_pre']
    
    # === Kalman Features ===
    if not fast_mode:
        kalman_pre = simple_kalman_local_level(pre_proc)
        q_pre = kalman_pre['kalman_q']
        r_pre = kalman_pre['kalman_r']
        
        kalman_post = simple_kalman_local_level(post_proc)
        kalman_cross = kalman_cross_prediction_error(post_proc, q_pre, r_pre)
        
        features['kalman_innov_var_pre'] = kalman_pre['kalman_innov_var']
        features['kalman_innov_var_post'] = kalman_post['kalman_innov_var']
        
        # Log ratio
        if kalman_pre['kalman_innov_var'] > 1e-10 and kalman_post['kalman_innov_var'] > 1e-10:
            features['log_kalman_innov_ratio'] = np.log(
                kalman_post['kalman_innov_var'] / kalman_pre['kalman_innov_var']
            )
        else:
            features['log_kalman_innov_ratio'] = 0.0
        
        features['kalman_rmse_pre'] = kalman_pre['kalman_rmse']
        features['kalman_rmse_post_pred_by_pre'] = kalman_cross['kalman_rmse_post_pred_by_pre']
    
    # === Windowed Features ===
    for window_size in window_sizes:
        suffix = f'_w{window_size}'
        window_feats = extract_windowed_features(pre, post, window_size, suffix)
        features.update(window_feats)
    
    return features


def compute_ar_kalman_features(df: pd.DataFrame,
                               window_sizes: list = None,
                               fast_mode: bool = False) -> pd.DataFrame:
    """
    Compute AR/Kalman features for all series in dataset.
    
    Args:
        df: DataFrame with MultiIndex (id, time) and columns [value, period]
        window_sizes: List of window sizes for boundary features
        fast_mode: If True, skip expensive features
        
    Returns:
        DataFrame with index=id and columns=features
    """
    if window_sizes is None:
        window_sizes = [25, 50, 100]
    
    feature_list = []
    ids = []
    
    for series_id in df.index.get_level_values(0).unique():
        series_data = df.loc[series_id]
        pre, post = split_pre_post(series_data)
        
        features = compute_ar_kalman_features_single(pre, post, window_sizes, fast_mode)
        feature_list.append(features)
        ids.append(series_id)
    
    feature_df = pd.DataFrame(feature_list, index=ids)
    feature_df.index.name = 'id'
    
    return feature_df
