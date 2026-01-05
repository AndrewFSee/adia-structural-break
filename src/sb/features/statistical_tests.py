"""
Statistical hypothesis tests and effect size features.

Based on winning solutions:
- 10th place: Anderson-Darling, Cohen's d, IQR ratios, variance ratios
- Brazilian team: F-test, Levene, KS tests
- 2nd place: Rolling statistics
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def split_pre_post(df_one_id) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a single series into pre-break and post-break segments.
    
    Args:
        df_one_id: DataFrame or Series for one series (can be MultiIndex or regular format)
        
    Returns:
        pre_values: Pre-break segment (period==0)
        post_values: Post-break segment (period==1)
    """
    # Convert Series to DataFrame if needed
    if isinstance(df_one_id, pd.Series):
        df_one_id = df_one_id.to_frame()
    
    # Try to find 'period' and 'value' columns, checking both columns and index
    if 'period' in df_one_id.columns and 'value' in df_one_id.columns:
        # Standard case: both are columns
        pre_values = df_one_id[df_one_id['period'] == 0]['value'].values
        post_values = df_one_id[df_one_id['period'] == 1]['value'].values
    elif 'period' in df_one_id.index.names:
        # period is in the index
        if 'value' in df_one_id.columns:
            pre_values = df_one_id[df_one_id.index.get_level_values('period') == 0]['value'].values
            post_values = df_one_id[df_one_id.index.get_level_values('period') == 1]['value'].values
        else:
            # Reset index and try again
            df_reset = df_one_id.reset_index()
            pre_values = df_reset[df_reset['period'] == 0]['value'].values
            post_values = df_reset[df_reset['period'] == 1]['value'].values
    else:
        # Last resort: reset index and look for period
        df_reset = df_one_id.reset_index()
        if 'period' in df_reset.columns and 'value' in df_reset.columns:
            pre_values = df_reset[df_reset['period'] == 0]['value'].values
            post_values = df_reset[df_reset['period'] == 1]['value'].values
        else:
            # Debug: print what we actually have
            raise ValueError(
                f"Cannot find 'period' and 'value' columns.\n"
                f"Original columns: {df_one_id.columns.tolist()}\n"
                f"Original index names: {df_one_id.index.names}\n"
                f"After reset_index columns: {df_reset.columns.tolist()}\n"
                f"DataFrame shape: {df_one_id.shape}"
            )
    
    return pre_values, post_values


def iter_series_data(df: pd.DataFrame):
    """
    Iterate over series in a DataFrame, handling both MultiIndex and regular format.
    
    Yields:
        (series_id, series_data) tuples
    """
    if isinstance(df.index, pd.MultiIndex):
        for series_id in df.index.get_level_values(0).unique():
            yield series_id, df.loc[series_id]
    else:
        # Regular format with 'id' column
        for series_id, series_data in df.groupby('id', sort=False):
            yield series_id, series_data


def compute_anderson_darling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anderson-Darling test - more powerful than KS for tail differences.
    
    Returns statistic (higher = more different distributions).
    """
    features = []
    ids = []
    
    for series_id, series_data in iter_series_data(df):
        pre, post = split_pre_post(series_data)
        
        # Anderson-Darling k-sample test
        result = stats.anderson_ksamp([pre, post])
        
        features.append({
            'anderson_darling_stat': result.statistic,
        })
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)


def compute_cohens_d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cohen's d - standardized effect size for mean difference.
    
    Formula: (mean_post - mean_pre) / pooled_std
    Interpretation:
    - |d| < 0.2: small effect
    - 0.2 <= |d| < 0.5: medium effect  
    - |d| >= 0.5: large effect
    """
    features = []
    ids = []
    eps = 1e-12
    
    for series_id, series_data in iter_series_data(df):
        pre, post = split_pre_post(series_data)
        
        mean_pre = np.mean(pre)
        mean_post = np.mean(post)
        var_pre = np.var(pre, ddof=1)
        var_post = np.var(post, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt((var_pre + var_post) / 2) + eps
        
        cohens_d = (mean_post - mean_pre) / pooled_std
        
        # Also compute standardized jump using pre std only
        std_jump = (mean_post - mean_pre) / (np.sqrt(var_pre) + eps)
        
        features.append({
            'cohens_d': cohens_d,
            'std_jump': std_jump,
        })
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)


def compute_variance_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variance ratios at multiple windows (from 10th place).
    
    Compares variance of differences (differenced series) before/after break.
    """
    features = []
    ids = []
    eps = 1e-12
    
    for series_id, series_data in iter_series_data(df):
        pre, post = split_pre_post(series_data)
        
        # Cumulative sum
        cumsum_full = np.cumsum(np.concatenate([pre, post]))
        pre_cumsum = cumsum_full[:len(pre)]
        post_cumsum = cumsum_full[len(pre):]
        
        feat = {}
        
        # Variance of differences ratio at multiple windows
        for w in [5, 20, 100]:
            # Get window
            w_pre = pre[-w:] if len(pre) >= w else pre
            w_post = post[:w] if len(post) >= w else post
            
            # Variance of differences
            if len(w_pre) > 1:
                vard_pre = np.var(np.diff(w_pre), ddof=1) + eps
            else:
                vard_pre = eps
                
            if len(w_post) > 1:
                vard_post = np.var(np.diff(w_post), ddof=1) + eps
            else:
                vard_post = eps
            
            feat[f'vard_ratio_w{w}'] = vard_post / vard_pre
        
        # Same for cumsum
        for w in [20, 100]:
            w_pre_cs = pre_cumsum[-w:] if len(pre_cumsum) >= w else pre_cumsum
            w_post_cs = post_cumsum[:w] if len(post_cumsum) >= w else post_cumsum
            
            if len(w_pre_cs) > 1:
                vard_pre_cs = np.var(np.diff(w_pre_cs), ddof=1) + eps
            else:
                vard_pre_cs = eps
                
            if len(w_post_cs) > 1:
                vard_post_cs = np.var(np.diff(w_post_cs), ddof=1) + eps
            else:
                vard_post_cs = eps
            
            feat[f'vard_ratio_cumsum_w{w}'] = vard_post_cs / vard_pre_cs
        
        features.append(feat)
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)


def compute_iqr_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR ratios at multiple windows (from 10th place).
    
    IQR = Q75 - Q25 (interquartile range, robust to outliers)
    """
    features = []
    ids = []
    eps = 1e-12
    
    for series_id, series_data in iter_series_data(df):
        pre, post = split_pre_post(series_data)
        
        feat = {}
        
        # IQR ratios at multiple windows
        for w in [20, 50, 100, 200]:
            w_pre = pre[-w:] if len(pre) >= w else pre
            w_post = post[:w] if len(post) >= w else post
            
            if len(w_pre) >= 2:
                q25_pre, q75_pre = np.percentile(w_pre, [25, 75])
                iqr_pre = (q75_pre - q25_pre) + eps
            else:
                iqr_pre = eps
                
            if len(w_post) >= 2:
                q25_post, q75_post = np.percentile(w_post, [25, 75])
                iqr_post = (q75_post - q25_post) + eps
            else:
                iqr_post = eps
            
            feat[f'iqr_ratio_w{w}'] = iqr_post / iqr_pre
        
        # IQR range difference (90th - 10th percentile)
        if len(pre) >= 10:
            q90_pre, q10_pre = np.percentile(pre, [90, 10])
        else:
            q90_pre = q10_pre = np.mean(pre)
            
        if len(post) >= 10:
            q90_post, q10_post = np.percentile(post, [90, 10])
        else:
            q90_post = q10_post = np.mean(post)
        
        feat['d_iqr_range'] = (q90_post - q10_post) - (q90_pre - q10_pre)
        
        features.append(feat)
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)


def compute_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    F-test, Levene's test, KS test (from Brazilian team and 2nd place).
    """
    features = []
    ids = []
    
    for series_id, series_data in iter_series_data(df):
        pre, post = split_pre_post(series_data)
        
        # F-test for variance
        if len(pre) > 1 and len(post) > 1:
            var_pre = np.var(pre, ddof=1)
            var_post = np.var(post, ddof=1)
            
            if var_pre > 0 and var_post > 0:
                f_stat = var_pre / var_post
                f_p = 2 * min(
                    stats.f.cdf(f_stat, len(pre)-1, len(post)-1),
                    1 - stats.f.cdf(f_stat, len(pre)-1, len(post)-1)
                )
            else:
                f_stat = f_p = 0
        else:
            f_stat = f_p = 0
        
        # Levene's test (robust variance test)
        if len(pre) >= 3 and len(post) >= 3:
            levene_stat, levene_p = stats.levene(pre, post)
        else:
            levene_stat = levene_p = 0
        
        # KS test (distribution comparison)
        if len(pre) >= 2 and len(post) >= 2:
            ks_stat, ks_p = stats.ks_2samp(pre, post)
        else:
            ks_stat = ks_p = 0
        
        features.append({
            'f_test_stat': f_stat,
            'f_test_p': f_p,
            'levene_stat': levene_stat,
            'levene_p': levene_p,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
        })
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)


def compute_rolling_stats(df: pd.DataFrame, window: int = 16) -> pd.DataFrame:
    """
    Rolling mean and rolling std (from 2nd place).
    
    Captures local trends and volatility changes.
    """
    features = []
    ids = []
    
    for series_id, series_data in iter_series_data(df):
        # Need to check if period is in columns or index
        if 'period' in series_data.columns:
            values = series_data['value'].values
            n_pre = (series_data['period'] == 0).sum()
        else:
            # Reset index to access period
            series_data_reset = series_data.reset_index()
            values = series_data_reset['value'].values
            n_pre = (series_data_reset['period'] == 0).sum()
        
        # Rolling mean
        rolling_mean = pd.Series(values).rolling(window, min_periods=1).mean().values
        
        # Rolling std
        rolling_std = pd.Series(values).rolling(window, min_periods=1).std().values
        
        # Get pre/post split (n_pre already computed above)
        rm_pre = rolling_mean[:n_pre]
        rm_post = rolling_mean[n_pre:]
        rs_pre = rolling_std[:n_pre]
        rs_post = rolling_std[n_pre:]
        
        features.append({
            f'rolling_mean_w{window}_pre_mean': np.mean(rm_pre),
            f'rolling_mean_w{window}_post_mean': np.mean(rm_post),
            f'rolling_mean_w{window}_diff': np.mean(rm_post) - np.mean(rm_pre),
            f'rolling_std_w{window}_pre_mean': np.mean(rs_pre),
            f'rolling_std_w{window}_post_mean': np.mean(rs_post),
            f'rolling_std_w{window}_diff': np.mean(rs_post) - np.mean(rs_pre),
        })
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)


def compute_statistical_test_features(
    df: pd.DataFrame,
    use_anderson: bool = True,
    use_cohens_d: bool = True,
    use_variance_ratios: bool = True,
    use_iqr_ratios: bool = True,
    use_hypothesis_tests: bool = True,
    use_rolling_stats: bool = True,
) -> pd.DataFrame:
    """
    Compute all statistical test features.
    
    Args:
        df: Multi-index DataFrame with (id, period) index
        use_anderson: Anderson-Darling test
        use_cohens_d: Cohen's d effect size
        use_variance_ratios: Variance of differences ratios
        use_iqr_ratios: IQR ratios at multiple windows
        use_hypothesis_tests: F-test, Levene, KS
        use_rolling_stats: Rolling mean/std
    
    Returns:
        DataFrame with statistical test features
    """
    feature_dfs = []
    
    if use_anderson:
        feature_dfs.append(compute_anderson_darling(df))
    
    if use_cohens_d:
        feature_dfs.append(compute_cohens_d(df))
    
    if use_variance_ratios:
        feature_dfs.append(compute_variance_ratios(df))
    
    if use_iqr_ratios:
        feature_dfs.append(compute_iqr_ratios(df))
    
    if use_hypothesis_tests:
        feature_dfs.append(compute_hypothesis_tests(df))
    
    if use_rolling_stats:
        feature_dfs.append(compute_rolling_stats(df, window=16))
    
    if not feature_dfs:
        raise ValueError("At least one feature type must be enabled")
    
    return pd.concat(feature_dfs, axis=1)
