"""
AR/Kalman structural break features with strict leakage prevention.

KEY ANTI-LEAKAGE PRINCIPLE:
All models and preprocessing are fit ONLY on PRE segment, then applied to POST.
This ensures features are "pre-model → post-score" style, capturing break signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import linalg
from joblib import Parallel, delayed
import warnings


def robust_preprocess(
    pre: np.ndarray,
    post: np.ndarray,
    winsorize_quantiles: Tuple[float, float] = (0.01, 0.99)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Preprocess using ONLY PRE statistics.
    
    ⚠️ ANTI-LEAKAGE: Winsorization and standardization fit on PRE only!
    
    Args:
        pre: Pre-segment values
        post: Post-segment values
        winsorize_quantiles: (lower, upper) quantiles for clipping
        
    Returns:
        pre_processed: Processed pre values
        post_processed: Processed post values (using PRE stats!)
        stats: Dictionary of preprocessing statistics
    """
    # Winsorize using PRE quantiles only
    q_low, q_high = np.quantile(pre, winsorize_quantiles)
    pre_wins = np.clip(pre, q_low, q_high)
    post_wins = np.clip(post, q_low, q_high)
    
    # Standardize using PRE mean/std only
    pre_mean = np.mean(pre_wins)
    pre_std = np.std(pre_wins)
    
    if pre_std < 1e-10:
        # Constant series - return zeros
        return np.zeros_like(pre), np.zeros_like(post), {
            'pre_mean': pre_mean,
            'pre_std': 0.0,
            'q_low': q_low,
            'q_high': q_high
        }
    
    pre_processed = (pre_wins - pre_mean) / pre_std
    post_processed = (post_wins - pre_mean) / pre_std
    
    stats = {
        'pre_mean': pre_mean,
        'pre_std': pre_std,
        'q_low': q_low,
        'q_high': q_high
    }
    
    return pre_processed, post_processed, stats


def fit_ar1(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit AR(1) model: x_t = phi * x_{t-1} + epsilon_t
    
    Uses Yule-Walker equations for robust estimation.
    
    Args:
        x: Time series values (should be preprocessed)
        
    Returns:
        phi: AR(1) coefficient
        resid_var: Residual variance
        rmse: Root mean squared error of 1-step predictions
    """
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    
    # Yule-Walker estimation
    acf_0 = np.var(x)
    acf_1 = np.correlate(x[:-1] - np.mean(x), x[1:] - np.mean(x), mode='valid')[0] / (len(x) - 1)
    
    if acf_0 < 1e-10:
        return 0.0, 0.0, 0.0
    
    phi = acf_1 / acf_0
    
    # Clip phi to stable range
    phi = np.clip(phi, -0.999, 0.999)
    
    # Compute residuals
    pred = phi * x[:-1]
    resid = x[1:] - pred
    resid_var = np.var(resid)
    rmse = np.sqrt(np.mean(resid**2))
    
    return phi, resid_var, rmse


def ar1_cross_prediction_error(
    pre: np.ndarray,
    post: np.ndarray
) -> Tuple[float, float]:
    """
    Apply PRE-fitted AR(1) to POST segment.
    
    ⚠️ ANTI-LEAKAGE: Uses PRE model on POST data!
    
    Args:
        pre: Pre-segment (preprocessed)
        post: Post-segment (preprocessed)
        
    Returns:
        rmse: RMSE of cross-predictions
        resid_var: Residual variance of cross-predictions
    """
    if len(pre) < 3 or len(post) < 2:
        return np.nan, np.nan
    
    # Fit on PRE
    phi_pre, _, _ = fit_ar1(pre)
    
    if np.isnan(phi_pre):
        return np.nan, np.nan
    
    # Apply to POST
    pred = phi_pre * post[:-1]
    resid = post[1:] - pred
    resid_var = np.var(resid)
    rmse = np.sqrt(np.mean(resid**2))
    
    return rmse, resid_var


def fit_ar2(x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit AR(2) model: x_t = phi1 * x_{t-1} + phi2 * x_{t-2} + epsilon_t
    
    Uses least squares with stability check.
    
    Args:
        x: Time series values (should be preprocessed)
        
    Returns:
        phi1, phi2: AR(2) coefficients
        resid_var: Residual variance
        rmse: Root mean squared error
    """
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan
    
    # Design matrix
    X_mat = np.column_stack([x[1:-1], x[:-2]])
    y_vec = x[2:]
    
    try:
        # Least squares with small ridge for numerical stability
        XtX = X_mat.T @ X_mat + 1e-6 * np.eye(2)
        Xty = X_mat.T @ y_vec
        phi = linalg.solve(XtX, Xty, assume_a='pos')
        
        phi1, phi2 = phi
        
        # Stability check: roots outside unit circle
        # Characteristic polynomial: 1 - phi1*z - phi2*z^2 = 0
        roots = np.roots([1, -phi1, -phi2])
        if np.any(np.abs(roots) >= 0.999):
            # Unstable - fall back to AR(1)
            phi1, resid_var, rmse = fit_ar1(x)
            return phi1, 0.0, resid_var, rmse
        
        # Compute residuals
        pred = phi1 * x[1:-1] + phi2 * x[:-2]
        resid = x[2:] - pred
        resid_var = np.var(resid)
        rmse = np.sqrt(np.mean(resid**2))
        
        return phi1, phi2, resid_var, rmse
        
    except:
        return np.nan, np.nan, np.nan, np.nan


def compute_ar_features(
    pre_values: np.ndarray,
    post_values: np.ndarray,
    window_sizes: List[int] = [25, 50, 100]
) -> Dict[str, float]:
    """
    Compute AR-based structural break features.
    
    ⚠️ ANTI-LEAKAGE: All models fit on PRE only!
    
    Args:
        pre_values: Pre-segment values (raw)
        post_values: Post-segment values (raw)
        window_sizes: Boundary window sizes
        
    Returns:
        Dictionary of AR features
    """
    features = {}
    
    # Preprocess using PRE stats only
    pre_proc, post_proc, _ = robust_preprocess(pre_values, post_values)
    
    # Full-segment AR(1) features
    phi_pre, resid_var_pre, rmse_pre = fit_ar1(pre_proc)
    phi_post, resid_var_post, rmse_post = fit_ar1(post_proc)
    
    features['ar1_phi_pre'] = phi_pre
    features['ar1_phi_post'] = phi_post
    features['ar1_delta_phi'] = phi_post - phi_pre if not np.isnan(phi_pre) and not np.isnan(phi_post) else np.nan
    features['ar1_resid_var_pre'] = resid_var_pre
    features['ar1_resid_var_post'] = resid_var_post
    features['ar1_log_resid_var_ratio'] = np.log(resid_var_post / (resid_var_pre + 1e-10)) if resid_var_pre > 1e-10 else np.nan
    features['ar1_rmse_pre'] = rmse_pre
    features['ar1_rmse_post'] = rmse_post
    
    # Cross-prediction: apply PRE model to POST
    rmse_cross, resid_var_cross = ar1_cross_prediction_error(pre_proc, post_proc)
    features['ar1_rmse_cross_pred'] = rmse_cross
    features['ar1_resid_var_cross_pred'] = resid_var_cross
    features['ar1_delta_rmse_cross'] = rmse_cross - rmse_pre if not np.isnan(rmse_cross) and not np.isnan(rmse_pre) else np.nan
    
    # AR(2) features (if enough data)
    if len(pre_proc) >= 5:
        phi1_pre, phi2_pre, resid_var2_pre, rmse2_pre = fit_ar2(pre_proc)
        features['ar2_phi1_pre'] = phi1_pre
        features['ar2_phi2_pre'] = phi2_pre
        features['ar2_resid_var_pre'] = resid_var2_pre
        features['ar2_rmse_pre'] = rmse2_pre
    else:
        features['ar2_phi1_pre'] = np.nan
        features['ar2_phi2_pre'] = np.nan
        features['ar2_resid_var_pre'] = np.nan
        features['ar2_rmse_pre'] = np.nan
    
    # Window-based features
    for w in window_sizes:
        if len(pre_proc) >= w and len(post_proc) >= w:
            # Last w of PRE, first w of POST
            pre_win = pre_proc[-w:]
            post_win = post_proc[:w]
            
            phi_pre_w, resid_var_pre_w, rmse_pre_w = fit_ar1(pre_win)
            phi_post_w, resid_var_post_w, rmse_post_w = fit_ar1(post_win)
            
            features[f'ar1_phi_pre_w{w}'] = phi_pre_w
            features[f'ar1_phi_post_w{w}'] = phi_post_w
            features[f'ar1_delta_phi_w{w}'] = phi_post_w - phi_pre_w if not np.isnan(phi_pre_w) and not np.isnan(phi_post_w) else np.nan
            features[f'ar1_rmse_pre_w{w}'] = rmse_pre_w
            
            # Cross-prediction for window
            rmse_cross_w, _ = ar1_cross_prediction_error(pre_win, post_win)
            features[f'ar1_rmse_cross_w{w}'] = rmse_cross_w
        else:
            features[f'ar1_phi_pre_w{w}'] = np.nan
            features[f'ar1_phi_post_w{w}'] = np.nan
            features[f'ar1_delta_phi_w{w}'] = np.nan
            features[f'ar1_rmse_pre_w{w}'] = np.nan
            features[f'ar1_rmse_cross_w{w}'] = np.nan
    
    return features


def simple_kalman_local_level(
    x: np.ndarray,
    process_var: float = 0.1,
    obs_var: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simple local level Kalman filter.
    
    Model:
        level_t = level_{t-1} + w_t,  w_t ~ N(0, process_var)
        x_t = level_t + v_t,          v_t ~ N(0, obs_var)
    
    Args:
        x: Observations
        process_var: Process noise variance (Q)
        obs_var: Observation noise variance (R)
        
    Returns:
        filtered_states: Filtered level estimates
        innovations: One-step prediction errors
        innovation_var: Variance of innovations
    """
    n = len(x)
    
    # Initialize
    level = np.zeros(n)
    P = np.zeros(n)  # State variance
    innovations = np.zeros(n)
    
    # Initial state
    level[0] = x[0]
    P[0] = obs_var
    
    for t in range(1, n):
        # Predict
        level_pred = level[t-1]
        P_pred = P[t-1] + process_var
        
        # Innovation
        innov = x[t] - level_pred
        innov_var = P_pred + obs_var
        
        # Kalman gain
        K = P_pred / innov_var
        
        # Update
        level[t] = level_pred + K * innov
        P[t] = (1 - K) * P_pred
        
        innovations[t] = innov
    
    innovation_var = np.var(innovations[1:])  # Skip first (no prediction)
    
    return level, innovations, innovation_var


def fit_kalman_params(
    x: np.ndarray,
    process_var_grid: np.ndarray = np.logspace(-3, 0, 5)
) -> Tuple[float, float, float]:
    """
    Fit Kalman filter parameters via simple grid search.
    
    Deterministic: uses fixed grid and selects best by log-likelihood.
    
    Args:
        x: Observations (preprocessed)
        process_var_grid: Grid of process variance values to try
        
    Returns:
        best_process_var: Best process noise variance
        obs_var: Fixed observation noise (=1 after preprocessing)
        innovation_var: Innovation variance with best params
    """
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    
    obs_var = 1.0  # Fixed after standardization
    
    best_nll = np.inf
    best_process_var = process_var_grid[0]
    best_innov_var = np.nan
    
    for pv in process_var_grid:
        _, innovations, innov_var = simple_kalman_local_level(x, pv, obs_var)
        
        # Negative log-likelihood (simplified)
        nll = 0.5 * np.sum(innovations[1:]**2 / innov_var) + 0.5 * (len(x) - 1) * np.log(innov_var)
        
        if nll < best_nll:
            best_nll = nll
            best_process_var = pv
            best_innov_var = innov_var
    
    return best_process_var, obs_var, best_innov_var


def kalman_cross_prediction_error(
    pre: np.ndarray,
    post: np.ndarray,
    process_var_grid: np.ndarray = np.logspace(-3, 0, 5)
) -> Tuple[float, float]:
    """
    Apply PRE-fitted Kalman filter to POST segment.
    
    ⚠️ ANTI-LEAKAGE: Uses PRE parameters on POST data!
    
    Args:
        pre: Pre-segment (preprocessed)
        post: Post-segment (preprocessed)
        process_var_grid: Grid for parameter fitting
        
    Returns:
        rmse: RMSE of one-step predictions on POST
        innovation_var: Innovation variance on POST
    """
    if len(pre) < 3 or len(post) < 2:
        return np.nan, np.nan
    
    # Fit params on PRE
    process_var_pre, obs_var_pre, _ = fit_kalman_params(pre, process_var_grid)
    
    if np.isnan(process_var_pre):
        return np.nan, np.nan
    
    # Apply to POST
    _, innovations, innov_var = simple_kalman_local_level(post, process_var_pre, obs_var_pre)
    rmse = np.sqrt(np.mean(innovations[1:]**2))
    
    return rmse, innov_var


def compute_kalman_features(
    pre_values: np.ndarray,
    post_values: np.ndarray,
    window_sizes: List[int] = [25, 50, 100]
) -> Dict[str, float]:
    """
    Compute Kalman filter structural break features.
    
    ⚠️ ANTI-LEAKAGE: All parameters fit on PRE only!
    
    Args:
        pre_values: Pre-segment values (raw)
        post_values: Post-segment values (raw)
        window_sizes: Boundary window sizes
        
    Returns:
        Dictionary of Kalman features
    """
    features = {}
    
    # Preprocess using PRE stats only
    pre_proc, post_proc, _ = robust_preprocess(pre_values, post_values)
    
    # Full-segment Kalman features
    process_var_pre, obs_var_pre, innov_var_pre = fit_kalman_params(pre_proc)
    _, _, innov_var_post = simple_kalman_local_level(post_proc, process_var_pre, obs_var_pre)
    
    features['kf_process_var_pre'] = process_var_pre
    features['kf_innov_var_pre'] = innov_var_pre
    features['kf_innov_var_post'] = innov_var_post
    features['kf_log_innov_var_ratio'] = np.log(innov_var_post / (innov_var_pre + 1e-10)) if innov_var_pre > 1e-10 else np.nan
    
    # Cross-prediction: apply PRE params to POST
    rmse_cross, innov_var_cross = kalman_cross_prediction_error(pre_proc, post_proc)
    features['kf_rmse_cross_pred'] = rmse_cross
    features['kf_innov_var_cross_pred'] = innov_var_cross
    
    # Window-based features
    for w in window_sizes:
        if len(pre_proc) >= w and len(post_proc) >= w:
            pre_win = pre_proc[-w:]
            post_win = post_proc[:w]
            
            pv_pre_w, ov_pre_w, iv_pre_w = fit_kalman_params(pre_win)
            _, _, iv_post_w = simple_kalman_local_level(post_win, pv_pre_w, ov_pre_w)
            
            features[f'kf_innov_var_pre_w{w}'] = iv_pre_w
            features[f'kf_innov_var_post_w{w}'] = iv_post_w
            features[f'kf_log_innov_var_ratio_w{w}'] = np.log(iv_post_w / (iv_pre_w + 1e-10)) if iv_pre_w > 1e-10 else np.nan
            
            # Cross-prediction for window
            rmse_cross_w, _ = kalman_cross_prediction_error(pre_win, post_win)
            features[f'kf_rmse_cross_w{w}'] = rmse_cross_w
        else:
            features[f'kf_innov_var_pre_w{w}'] = np.nan
            features[f'kf_innov_var_post_w{w}'] = np.nan
            features[f'kf_log_innov_var_ratio_w{w}'] = np.nan
            features[f'kf_rmse_cross_w{w}'] = np.nan
    
    return features


def fit_local_linear_trend(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit local linear trend: x_t ≈ level + slope * t
    
    Uses robust least squares.
    
    Args:
        x: Time series values (preprocessed)
        
    Returns:
        level: Intercept
        slope: Slope coefficient
        rmse: Fit quality
    """
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    
    t = np.arange(len(x))
    
    # Least squares
    X_mat = np.column_stack([np.ones(len(x)), t])
    
    try:
        params = linalg.lstsq(X_mat, x)[0]
        level, slope = params
        
        pred = level + slope * t
        resid = x - pred
        rmse = np.sqrt(np.mean(resid**2))
        
        return level, slope, rmse
    except:
        return np.nan, np.nan, np.nan


def compute_local_trend_features(
    pre_values: np.ndarray,
    post_values: np.ndarray,
    window_sizes: List[int] = [25, 50, 100]
) -> Dict[str, float]:
    """
    Compute local trend structural break features.
    
    ⚠️ ANTI-LEAKAGE: Trend fit on PRE only!
    
    Args:
        pre_values: Pre-segment values (raw)
        post_values: Post-segment values (raw)
        window_sizes: Boundary window sizes
        
    Returns:
        Dictionary of trend features
    """
    features = {}
    
    # Preprocess using PRE stats only
    pre_proc, post_proc, _ = robust_preprocess(pre_values, post_values)
    
    # Full-segment trend features
    level_pre, slope_pre, rmse_pre = fit_local_linear_trend(pre_proc)
    level_post, slope_post, rmse_post = fit_local_linear_trend(post_proc)
    
    features['trend_slope_pre'] = slope_pre
    features['trend_slope_post'] = slope_post
    features['trend_delta_slope'] = slope_post - slope_pre if not np.isnan(slope_pre) and not np.isnan(slope_post) else np.nan
    features['trend_rmse_pre'] = rmse_pre
    features['trend_rmse_post'] = rmse_post
    
    # Cross-prediction: apply PRE trend to POST
    if not np.isnan(level_pre) and not np.isnan(slope_pre):
        t_post = np.arange(len(post_proc))
        pred_post = level_pre + slope_pre * t_post
        resid_post = post_proc - pred_post
        rmse_cross = np.sqrt(np.mean(resid_post**2))
        mean_error = np.mean(resid_post)
        
        features['trend_rmse_cross'] = rmse_cross
        features['trend_mean_error_cross'] = mean_error
    else:
        features['trend_rmse_cross'] = np.nan
        features['trend_mean_error_cross'] = np.nan
    
    # Window-based features
    for w in window_sizes:
        if len(pre_proc) >= w and len(post_proc) >= w:
            pre_win = pre_proc[-w:]
            post_win = post_proc[:w]
            
            level_pre_w, slope_pre_w, rmse_pre_w = fit_local_linear_trend(pre_win)
            level_post_w, slope_post_w, rmse_post_w = fit_local_linear_trend(post_win)
            
            features[f'trend_slope_pre_w{w}'] = slope_pre_w
            features[f'trend_slope_post_w{w}'] = slope_post_w
            features[f'trend_delta_slope_w{w}'] = slope_post_w - slope_pre_w if not np.isnan(slope_pre_w) and not np.isnan(slope_post_w) else np.nan
            
            # Cross-prediction for window
            if not np.isnan(level_pre_w) and not np.isnan(slope_pre_w):
                t_post_w = np.arange(len(post_win))
                pred_post_w = level_pre_w + slope_pre_w * t_post_w
                rmse_cross_w = np.sqrt(np.mean((post_win - pred_post_w)**2))
                features[f'trend_rmse_cross_w{w}'] = rmse_cross_w
            else:
                features[f'trend_rmse_cross_w{w}'] = np.nan
        else:
            features[f'trend_slope_pre_w{w}'] = np.nan
            features[f'trend_slope_post_w{w}'] = np.nan
            features[f'trend_delta_slope_w{w}'] = np.nan
            features[f'trend_rmse_cross_w{w}'] = np.nan
    
    return features


def extract_features_for_id(
    df_one_id: pd.DataFrame,
    window_sizes: List[int] = [25, 50, 100]
) -> Dict[str, float]:
    """
    Extract all AR/Kalman features for a single time series.
    
    Args:
        df_one_id: DataFrame for one series ID (MultiIndex with time, or regular)
        window_sizes: Boundary window sizes
        
    Returns:
        Dictionary of all features
    """
    # Split into pre/post
    pre_mask = df_one_id['period'] == 0
    post_mask = df_one_id['period'] == 1
    
    pre_values = df_one_id.loc[pre_mask, 'value'].values
    post_values = df_one_id.loc[post_mask, 'value'].values
    
    if len(pre_values) < 3 or len(post_values) < 2:
        # Insufficient data - return NaNs
        return {}
    
    # Compute all feature families
    features = {}
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        ar_feats = compute_ar_features(pre_values, post_values, window_sizes)
        kf_feats = compute_kalman_features(pre_values, post_values, window_sizes)
        trend_feats = compute_local_trend_features(pre_values, post_values, window_sizes)
        
        features.update(ar_feats)
        features.update(kf_feats)
        features.update(trend_feats)
    
    return features


def extract_features(
    X: pd.DataFrame,
    window_sizes: List[int] = [25, 50, 100],
    n_jobs: int = 1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract AR/Kalman features for all time series.
    
    ⚠️ ANTI-LEAKAGE: All preprocessing and model fitting use PRE segment only!
    
    Args:
        X: DataFrame with MultiIndex (id, time) and columns [value, period]
        window_sizes: Boundary window sizes
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Whether to print progress
        
    Returns:
        DataFrame indexed by id with all features
    """
    if verbose:
        print(f"Extracting AR/Kalman features for {X.index.get_level_values(0).nunique():,} series...")
        print(f"Window sizes: {window_sizes}")
        print(f"Parallel jobs: {n_jobs}")
    
    # Get unique IDs
    if isinstance(X.index, pd.MultiIndex):
        unique_ids = X.index.get_level_values(0).unique()
    else:
        unique_ids = X['id'].unique()
    
    # Extract features in parallel
    def process_one(series_id):
        if isinstance(X.index, pd.MultiIndex):
            df_one = X.loc[series_id]
        else:
            df_one = X[X['id'] == series_id]
        
        return series_id, extract_features_for_id(df_one, window_sizes)
    
    if n_jobs == 1:
        results = [process_one(sid) for sid in unique_ids]
    else:
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_one)(sid) for sid in unique_ids
        )
    
    # Convert to DataFrame
    feature_dicts = {sid: feats for sid, feats in results}
    features_df = pd.DataFrame.from_dict(feature_dicts, orient='index')
    features_df.index.name = 'id'
    
    if verbose:
        print(f"✅ Extracted {features_df.shape[1]} features for {len(features_df):,} series")
        n_nan = features_df.isna().sum().sum()
        print(f"   NaN count: {n_nan:,} ({n_nan / features_df.size * 100:.1f}%)")
    
    return features_df
