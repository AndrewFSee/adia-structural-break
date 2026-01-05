"""
Wavelet transform features for structural break detection.

Wavelets capture both time and frequency information at different scales,
making them effective for detecting changes in time series structure.

Based on research showing wavelets are powerful for change point detection.
"""

import numpy as np
import pandas as pd
import pywt
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


def wavelet_energy(coeffs):
    """Compute energy (sum of squares) of wavelet coefficients."""
    return np.sum(coeffs**2)


def wavelet_entropy(coeffs):
    """Compute entropy of wavelet coefficient distribution."""
    coeffs_sq = coeffs**2
    energy = np.sum(coeffs_sq)
    if energy == 0:
        return 0
    probs = coeffs_sq / energy
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log(probs))


def compute_wavelet_features(series, wavelet='db4', level=3):
    """
    Compute wavelet decomposition features for a time series.
    
    Args:
        series: 1D array of time series values
        wavelet: Wavelet family ('db4', 'sym4', 'coif2', etc.)
        level: Decomposition level
    
    Returns:
        Dictionary of wavelet features
    """
    features = {}
    
    try:
        # Ensure series is long enough for decomposition
        min_len = 2**level
        if len(series) < min_len:
            # Pad if too short
            series = np.pad(series, (0, min_len - len(series)), mode='edge')
        
        # Decompose
        coeffs = pywt.wavedec(series, wavelet, level=level)
        
        # Features for each level
        for i, coeff in enumerate(coeffs):
            level_name = 'a' if i == 0 else f'd{i}'
            
            # Energy at this level
            features[f'wavelet_{level_name}_energy'] = wavelet_energy(coeff)
            
            # Standard deviation
            features[f'wavelet_{level_name}_std'] = np.std(coeff)
            
            # Mean absolute value
            features[f'wavelet_{level_name}_mean_abs'] = np.mean(np.abs(coeff))
            
            # Entropy
            features[f'wavelet_{level_name}_entropy'] = wavelet_entropy(coeff)
        
        # Total energy
        total_energy = sum(wavelet_energy(c) for c in coeffs)
        features['wavelet_total_energy'] = total_energy
        
        # Energy distribution (percentage at each level)
        for i, coeff in enumerate(coeffs):
            level_name = 'a' if i == 0 else f'd{i}'
            level_energy = wavelet_energy(coeff)
            features[f'wavelet_{level_name}_energy_pct'] = level_energy / (total_energy + 1e-8)
        
    except Exception as e:
        # If wavelet transform fails, return zeros
        for i in range(level + 1):
            level_name = 'a' if i == 0 else f'd{i}'
            features[f'wavelet_{level_name}_energy'] = 0
            features[f'wavelet_{level_name}_std'] = 0
            features[f'wavelet_{level_name}_mean_abs'] = 0
            features[f'wavelet_{level_name}_entropy'] = 0
            features[f'wavelet_{level_name}_energy_pct'] = 0
        features['wavelet_total_energy'] = 0
    
    return features


def extract_features(df: pd.DataFrame, wavelet='db4', level=3) -> pd.DataFrame:
    """
    Extract wavelet-based features for structural break detection.
    
    Computes wavelet decomposition for pre and post segments and creates
    comparison features:
    - Energy ratios (post/pre at each level)
    - Energy differences (post-pre at each level)
    - Entropy changes
    - Energy distribution changes
    
    Args:
        df: Multi-index DataFrame with (series_id, period) index
        wavelet: Wavelet family to use
        level: Decomposition level
    
    Returns:
        DataFrame with wavelet features
    """
    features = []
    ids = []
    
    for series_id, series_data in iter_series_data(df):
        pre, post = split_pre_post(series_data)
        
        # Compute wavelet features for pre and post
        pre_features = compute_wavelet_features(pre, wavelet, level)
        post_features = compute_wavelet_features(post, wavelet, level)
        
        feat = {}
        
        # Comparison features
        for key in pre_features.keys():
            pre_val = pre_features[key]
            post_val = post_features[key]
            
            # Ratio
            feat[f'{key}_ratio'] = post_val / (pre_val + 1e-8)
            
            # Difference
            feat[f'{key}_diff'] = post_val - pre_val
            
            # Absolute change
            feat[f'{key}_abs_change'] = abs(post_val - pre_val)
        
        # Energy shift features (change in energy distribution)
        for i in range(level + 1):
            level_name = 'a' if i == 0 else f'd{i}'
            pre_pct = pre_features[f'wavelet_{level_name}_energy_pct']
            post_pct = post_features[f'wavelet_{level_name}_energy_pct']
            feat[f'wavelet_{level_name}_energy_shift'] = abs(post_pct - pre_pct)
        
        features.append(feat)
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)
