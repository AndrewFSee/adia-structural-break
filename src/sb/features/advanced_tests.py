"""
Advanced statistical tests for structural break detection.

These are more powerful tests that complement the existing statistical_tests.py:
- Cramér-von Mises: More powerful than KS for all distribution differences
- Energy statistic: Distance-based, powerful for complex differences
- Wilcoxon rank-sum: Robust non-parametric test
- Mood's test: Specialized for scale/variance differences
- Tail features: Quantile ratios, extreme values, kurtosis
- Spectral features: Frequency domain analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import cramervonmises_2samp, energy_distance, ranksums, mood, kurtosis
from typing import Tuple


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
            raise ValueError(f"Cannot find 'period' and 'value' columns")
    
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


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract advanced statistical test features.
    
    Args:
        df: Multi-index DataFrame with (series_id, period) index
            and 'value' column
    
    Returns:
        DataFrame with advanced statistical features
    """
    features = []
    ids = []
    
    for series_id, series_data in iter_series_data(df):
        pre, post = split_pre_post(series_data)
        
        feat = {}
        
        # Cramér-von Mises test (more powerful than KS)
        try:
            cvm = cramervonmises_2samp(pre, post)
            feat['cvm_stat'] = cvm.statistic
            feat['cvm_pval'] = cvm.pvalue
        except:
            feat['cvm_stat'] = 0
            feat['cvm_pval'] = 1
        
        # Energy statistic (distance-based)
        try:
            feat['energy_dist'] = energy_distance(pre, post)
        except:
            feat['energy_dist'] = 0
        
        # Wilcoxon rank-sum (robust non-parametric)
        try:
            wstat, wpval = ranksums(pre, post)
            feat['wilcoxon_stat'] = wstat
            feat['wilcoxon_pval'] = wpval
        except:
            feat['wilcoxon_stat'] = 0
            feat['wilcoxon_pval'] = 1
        
        # Mood's test (scale/variance)
        try:
            mstat, mpval = mood(pre, post)
            feat['mood_stat'] = mstat
            feat['mood_pval'] = mpval
        except:
            feat['mood_stat'] = 0
            feat['mood_pval'] = 1
        
        # Tail behavior features
        try:
            # Quantile ratios (Q95/Q5)
            pre_q95 = np.percentile(pre, 95)
            pre_q05 = np.percentile(pre, 5)
            post_q95 = np.percentile(post, 95)
            post_q05 = np.percentile(post, 5)
            
            pre_tail_ratio = pre_q95 / (pre_q05 + 1e-8)
            post_tail_ratio = post_q95 / (post_q05 + 1e-8)
            
            feat['tail_q95_q05_ratio'] = post_tail_ratio / (pre_tail_ratio + 1e-8)
            feat['tail_q95_q05_diff'] = post_tail_ratio - pre_tail_ratio
            
            # Kurtosis change (tail heaviness)
            pre_kurt = kurtosis(pre)
            post_kurt = kurtosis(post)
            feat['tail_kurtosis_diff'] = post_kurt - pre_kurt
            feat['tail_kurtosis_ratio'] = post_kurt / (pre_kurt + 1e-8) if pre_kurt != 0 else 1
            
            # Extreme values (> 3 std)
            pre_mean, pre_std = np.mean(pre), np.std(pre)
            post_mean, post_std = np.mean(post), np.std(post)
            
            pre_extremes = np.sum(np.abs(pre - pre_mean) > 3*pre_std)
            post_extremes = np.sum(np.abs(post - post_mean) > 3*post_std)
            
            feat['tail_extremes_ratio'] = post_extremes / (pre_extremes + 1)
            feat['tail_extremes_diff'] = post_extremes - pre_extremes
            
        except:
            feat['tail_q95_q05_ratio'] = 1
            feat['tail_q95_q05_diff'] = 0
            feat['tail_kurtosis_diff'] = 0
            feat['tail_kurtosis_ratio'] = 1
            feat['tail_extremes_ratio'] = 1
            feat['tail_extremes_diff'] = 0
        
        # Spectral features (FFT)
        try:
            from scipy.fft import fft
            
            # FFT magnitude spectra
            pre_fft = np.abs(fft(pre))[:len(pre)//2]
            post_fft = np.abs(fft(post))[:len(post)//2]
            
            # Normalize to make them comparable
            pre_fft = pre_fft / (np.sum(pre_fft) + 1e-8)
            post_fft = post_fft / (np.sum(post_fft) + 1e-8)
            
            # Ensure same length for comparison
            min_len = min(len(pre_fft), len(post_fft))
            pre_fft = pre_fft[:min_len]
            post_fft = post_fft[:min_len]
            
            # L1 distance between spectra
            feat['spectral_l1_distance'] = np.sum(np.abs(pre_fft - post_fft))
            
            # KL divergence (with smoothing)
            pre_fft_smooth = pre_fft + 1e-8
            post_fft_smooth = post_fft + 1e-8
            feat['spectral_kl_divergence'] = np.sum(
                pre_fft_smooth * np.log(pre_fft_smooth / post_fft_smooth)
            )
            
            # Dominant frequency shift
            pre_dominant = np.argmax(pre_fft)
            post_dominant = np.argmax(post_fft)
            feat['spectral_dominant_freq_shift'] = abs(post_dominant - pre_dominant) / min_len
            
            # Energy in low vs high frequencies
            split = min_len // 4
            pre_low_energy = np.sum(pre_fft[:split]**2)
            post_low_energy = np.sum(post_fft[:split]**2)
            feat['spectral_low_freq_ratio'] = post_low_energy / (pre_low_energy + 1e-8)
            
        except:
            feat['spectral_l1_distance'] = 0
            feat['spectral_kl_divergence'] = 0
            feat['spectral_dominant_freq_shift'] = 0
            feat['spectral_low_freq_ratio'] = 1
        
        # Permutation entropy (complexity measure)
        try:
            def permutation_entropy(series, m=3, tau=1):
                """Calculate permutation entropy."""
                from itertools import permutations as iterperms
                
                if len(series) < m * tau:
                    return 0
                
                perms = list(iterperms(range(m)))
                perm_counts = {p: 0 for p in perms}
                
                for i in range(len(series) - (m-1)*tau):
                    idx = [i + j*tau for j in range(m)]
                    pattern = tuple(np.argsort([series[j] for j in idx]))
                    if pattern in perm_counts:
                        perm_counts[pattern] += 1
                
                total = sum(perm_counts.values())
                if total == 0:
                    return 0
                    
                probs = [c/total for c in perm_counts.values() if c > 0]
                entropy = -sum(p * np.log(p) for p in probs)
                return entropy
            
            pe_pre = permutation_entropy(pre, m=3)
            pe_post = permutation_entropy(post, m=3)
            
            feat['perm_entropy_pre'] = pe_pre
            feat['perm_entropy_post'] = pe_post
            feat['perm_entropy_ratio'] = pe_post / (pe_pre + 1e-8)
            feat['perm_entropy_diff'] = pe_post - pe_pre
            
        except:
            feat['perm_entropy_pre'] = 0
            feat['perm_entropy_post'] = 0
            feat['perm_entropy_ratio'] = 1
            feat['perm_entropy_diff'] = 0
        
        features.append(feat)
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)
