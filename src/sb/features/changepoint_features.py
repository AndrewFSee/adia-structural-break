"""
Bayesian change point detection features using ruptures library.

Uses multiple algorithms to detect change points and extract features:
- Pelt (Pruned Exact Linear Time)
- Binary Segmentation
- Window-based detection
- Bottom-Up hierarchical

These provide complementary views of structural breaks.
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from typing import Tuple


def split_pre_post(df_one_id) -> Tuple[np.ndarray, np.ndarray]:
    """Split a single series into pre-break and post-break segments."""
    if isinstance(df_one_id, pd.Series):
        df_one_id = df_one_id.to_frame()
    
    if 'period' in df_one_id.columns and 'value' in df_one_id.columns:
        pre_values = df_one_id[df_one_id['period'] == 0]['value'].values
        post_values = df_one_id[df_one_id['period'] == 1]['value'].values
    elif 'period' in df_one_id.index.names:
        if 'value' in df_one_id.columns:
            pre_values = df_one_id[df_one_id.index.get_level_values('period') == 0]['value'].values
            post_values = df_one_id[df_one_id.index.get_level_values('period') == 1]['value'].values
        else:
            df_reset = df_one_id.reset_index()
            pre_values = df_reset[df_reset['period'] == 0]['value'].values
            post_values = df_reset[df_reset['period'] == 1]['value'].values
    else:
        df_reset = df_one_id.reset_index()
        if 'period' in df_reset.columns and 'value' in df_reset.columns:
            pre_values = df_reset[df_reset['period'] == 0]['value'].values
            post_values = df_reset[df_reset['period'] == 1]['value'].values
        else:
            raise ValueError(f"Cannot find 'period' and 'value' columns")
    
    return pre_values, post_values


def iter_series_data(df: pd.DataFrame):
    """Iterate over series in a DataFrame."""
    if isinstance(df.index, pd.MultiIndex):
        for series_id in df.index.get_level_values(0).unique():
            yield series_id, df.loc[series_id]
    else:
        for series_id, series_data in df.groupby('id', sort=False):
            yield series_id, series_data


def detect_changepoints_pelt(series, model='rbf', penalty=3.0):
    """
    Detect change points using Pelt algorithm.
    
    Args:
        series: Time series array
        model: Cost model ('l1', 'l2', 'rbf', 'linear', 'normal', 'ar')
        penalty: Penalty value (higher = fewer change points)
    
    Returns:
        Dictionary with change point features
    """
    features = {}
    
    try:
        # Add timeout protection - if series is too long, skip
        if len(series) > 500:
            features[f'pelt_{model}_n_changepoints'] = 0
            features[f'pelt_{model}_first_cp_pos'] = 0.5
            features[f'pelt_{model}_avg_distance'] = 1.0
            return features
        
        algo = rpt.Pelt(model=model, min_size=2).fit(series)
        result = algo.predict(pen=penalty)
        
        # Number of change points detected
        n_changepoints = len(result) - 1  # Last point is end of series
        features[f'pelt_{model}_n_changepoints'] = n_changepoints
        
        if n_changepoints > 0:
            # Position of first change point (normalized)
            features[f'pelt_{model}_first_cp_pos'] = result[0] / len(series)
            
            # Average distance between change points
            if n_changepoints > 1:
                distances = np.diff(result)
                features[f'pelt_{model}_avg_distance'] = np.mean(distances) / len(series)
            else:
                features[f'pelt_{model}_avg_distance'] = 1.0
        else:
            features[f'pelt_{model}_first_cp_pos'] = 0.5
            features[f'pelt_{model}_avg_distance'] = 1.0
            
    except Exception as e:
        features[f'pelt_{model}_n_changepoints'] = 0
        features[f'pelt_{model}_first_cp_pos'] = 0.5
        features[f'pelt_{model}_avg_distance'] = 1.0
    
    return features


def detect_changepoints_binseg(series, model='l2', n_bkps=1):
    """
    Detect change points using Binary Segmentation.
    
    Args:
        series: Time series array
        model: Cost model
        n_bkps: Number of breakpoints to detect
    
    Returns:
        Dictionary with change point features
    """
    features = {}
    
    try:
        algo = rpt.Binseg(model=model, min_size=2).fit(series)
        result = algo.predict(n_bkps=n_bkps)
        
        if len(result) > 0:
            # Position of detected breakpoint (normalized)
            features[f'binseg_{model}_cp_pos'] = result[0] / len(series)
            
            # Distance from midpoint (how centered is the break?)
            features[f'binseg_{model}_cp_from_mid'] = abs(result[0] / len(series) - 0.5)
        else:
            features[f'binseg_{model}_cp_pos'] = 0.5
            features[f'binseg_{model}_cp_from_mid'] = 0.0
            
    except Exception as e:
        features[f'binseg_{model}_cp_pos'] = 0.5
        features[f'binseg_{model}_cp_from_mid'] = 0.0
    
    return features


def detect_changepoints_window(series, model='l2', width=50):
    """
    Detect change points using Window-based method.
    
    Args:
        series: Time series array
        model: Cost model
        width: Window width
    
    Returns:
        Dictionary with change point features
    """
    features = {}
    
    try:
        # Adjust width if series is too short
        width = min(width, len(series) // 3)
        if width < 5:
            width = 5
        
        algo = rpt.Window(width=width, model=model).fit(series)
        result = algo.predict(n_bkps=1)
        
        if len(result) > 0:
            # Position of detected breakpoint
            features[f'window_{model}_cp_pos'] = result[0] / len(series)
        else:
            features[f'window_{model}_cp_pos'] = 0.5
            
    except Exception as e:
        features[f'window_{model}_cp_pos'] = 0.5
    
    return features


def compute_segment_statistics(series, changepoint_pos):
    """
    Compute statistics before and after a detected change point.
    
    Args:
        series: Time series array
        changepoint_pos: Position of change point (0-1)
    
    Returns:
        Dictionary with segment comparison features
    """
    features = {}
    
    try:
        cp_idx = int(changepoint_pos * len(series))
        if cp_idx <= 0:
            cp_idx = 1
        if cp_idx >= len(series):
            cp_idx = len(series) - 1
        
        seg1 = series[:cp_idx]
        seg2 = series[cp_idx:]
        
        if len(seg1) > 0 and len(seg2) > 0:
            # Mean difference
            features['cp_mean_diff'] = np.mean(seg2) - np.mean(seg1)
            features['cp_mean_ratio'] = np.mean(seg2) / (np.mean(seg1) + 1e-8)
            
            # Variance difference
            features['cp_var_diff'] = np.var(seg2) - np.var(seg1)
            features['cp_var_ratio'] = np.var(seg2) / (np.var(seg1) + 1e-8)
            
            # Range difference
            features['cp_range_diff'] = (np.max(seg2) - np.min(seg2)) - (np.max(seg1) - np.min(seg1))
        else:
            features['cp_mean_diff'] = 0
            features['cp_mean_ratio'] = 1
            features['cp_var_diff'] = 0
            features['cp_var_ratio'] = 1
            features['cp_range_diff'] = 0
            
    except Exception as e:
        features['cp_mean_diff'] = 0
        features['cp_mean_ratio'] = 1
        features['cp_var_diff'] = 0
        features['cp_var_ratio'] = 1
        features['cp_range_diff'] = 0
    
    return features


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Bayesian change point detection features.
    
    Uses multiple algorithms (Pelt, Binary Segmentation, Window) with
    different cost models to detect structural breaks and extract features.
    
    Args:
        df: Multi-index DataFrame with (series_id, period) index
    
    Returns:
        DataFrame with change point detection features
    """
    features = []
    ids = []
    
    for series_id, series_data in iter_series_data(df):
        pre, post = split_pre_post(series_data)
        
        # Concatenate pre and post for full series analysis
        full_series = np.concatenate([pre, post])
        
        feat = {}
        
        # Only use fast l2 model, skip slow rbf
        pelt_features = detect_changepoints_pelt(full_series, model='l2', penalty=3.0)
        feat.update(pelt_features)
        
        # Binary Segmentation - faster than Pelt
        binseg_features = detect_changepoints_binseg(full_series, model='l2', n_bkps=1)
        feat.update(binseg_features)
        
        # Window-based detection - only one window size
        window_features = detect_changepoints_window(full_series, model='l2', width=50)
        feat.update(window_features)
        
        # Segment statistics using known boundary
        # We know the boundary is at len(pre), so compare detected vs actual
        actual_cp_pos = len(pre) / len(full_series)
        feat['cp_actual_pos'] = actual_cp_pos
        
        # Compute statistics at actual boundary
        seg_stats = compute_segment_statistics(full_series, actual_cp_pos)
        feat.update(seg_stats)
        
        # Detection accuracy features
        # How close did each algorithm get to the actual boundary?
        error_features = {}
        for key in list(feat.keys()):
            if 'cp_pos' in key and key != 'cp_actual_pos':
                detected_pos = feat[key]
                error_features[f'{key}_error'] = abs(detected_pos - actual_cp_pos)
        feat.update(error_features)
        
        features.append(feat)
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)
