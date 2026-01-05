"""
Minimal spectral features for structural break detection.

Top-6 spectral family: entropy and low/high frequency balance.
All features are leakage-safe: preprocessing uses PRE segment statistics only.
"""

import numpy as np
from typing import Dict


def safe_standardize_pre_only(
    x0: np.ndarray,
    x1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize both segments using PRE statistics only.
    
    ⚠️ ANTI-LEAKAGE: Uses PRE mean/std for both segments!
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        
    Returns:
        x0_scaled, x1_scaled (or zeros if std too small)
    """
    if len(x0) == 0:
        return x0, x1
    
    mean_pre = np.mean(x0)
    std_pre = np.std(x0)
    
    if std_pre < 1e-10:
        # Constant series - return zeros
        return np.zeros_like(x0), np.zeros_like(x1)
    
    x0_scaled = (x0 - mean_pre) / std_pre
    x1_scaled = (x1 - mean_pre) / std_pre
    
    return x0_scaled, x1_scaled


def compute_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized power spectral density using Hann-windowed periodogram.
    
    Args:
        x: Time series (preprocessed)
        
    Returns:
        freqs: Frequency bins (normalized, 0 to 0.5)
        psd: Power spectral density (normalized to sum to 1)
    """
    n = len(x)
    
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    x_windowed = x * window
    
    # Compute rFFT (real-valued input)
    fft_vals = np.fft.rfft(x_windowed)
    
    # Power spectral density (magnitude squared)
    psd = np.abs(fft_vals) ** 2
    
    # Normalize to sum to 1 (treat as probability distribution)
    psd = psd / (np.sum(psd) + 1e-12)
    
    # Frequency bins (0 to 0.5 for real signal)
    freqs = np.fft.rfftfreq(n)
    
    return freqs, psd


def spectral_entropy(psd: np.ndarray) -> float:
    """
    Compute normalized Shannon entropy of PSD.
    
    Higher entropy = more uniform spectrum (white noise)
    Lower entropy = more concentrated spectrum (periodic signal)
    
    Args:
        psd: Power spectral density (normalized to sum to 1)
        
    Returns:
        Spectral entropy (0 to 1)
    """
    psd_safe = psd + 1e-12  # Avoid log(0)
    entropy = -np.sum(psd_safe * np.log(psd_safe))
    
    # Normalize by maximum possible entropy
    max_entropy = np.log(len(psd))
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def log_low_high_ratio(freqs: np.ndarray, psd: np.ndarray, cutoff: float = 0.08) -> float:
    """
    Compute log ratio of low-band to high-band power.
    
    Positive = low frequencies dominate (trending/smooth behavior)
    Negative = high frequencies dominate (noisy/erratic behavior)
    
    Args:
        freqs: Frequency bins
        psd: Power spectral density (normalized)
        cutoff: Frequency cutoff between low and high (default: 0.08)
        
    Returns:
        Log ratio log(low/high)
    """
    low_mask = freqs <= cutoff
    high_mask = freqs > cutoff
    
    low_power = np.sum(psd[low_mask]) + 1e-12
    high_power = np.sum(psd[high_mask]) + 1e-12
    
    return np.log(low_power / high_power)


def spectral_peak_ratio(psd: np.ndarray) -> float:
    """
    Ratio of strongest non-DC peak to total power.
    
    High ratio = strong periodicity at one frequency
    Low ratio = distributed/noisy spectrum
    
    Args:
        psd: Power spectral density (normalized)
        
    Returns:
        Peak ratio (0 to 1)
    """
    if len(psd) < 2:
        return 0.0
    
    # Skip DC component (index 0)
    peak_power = np.max(psd[1:])
    return peak_power


def spectral_flatness(psd: np.ndarray) -> float:
    """
    Spectral flatness: geometric mean / arithmetic mean of PSD.
    
    Near 1.0 = white noise (flat spectrum)
    Near 0.0 = tonal/periodic (concentrated spectrum)
    
    Args:
        psd: Power spectral density (normalized)
        
    Returns:
        Flatness (0 to 1)
    """
    psd_safe = psd + 1e-12
    
    # Geometric mean
    log_mean = np.mean(np.log(psd_safe))
    geom_mean = np.exp(log_mean)
    
    # Arithmetic mean
    arith_mean = np.mean(psd_safe)
    
    return geom_mean / arith_mean if arith_mean > 0 else 0.0


def spectral_flux(psd: np.ndarray) -> float:
    """
    Mean squared difference between adjacent PSD bins.
    
    High flux = rapid spectral variation
    Low flux = smooth spectrum
    
    Args:
        psd: Power spectral density (normalized)
        
    Returns:
        Mean squared flux
    """
    if len(psd) < 2:
        return 0.0
    
    diff = np.diff(psd)
    return np.mean(diff ** 2)


def spectral_rolloff(freqs: np.ndarray, psd: np.ndarray, threshold: float = 0.5) -> float:
    """
    Frequency below which threshold% of total power is contained.
    
    Low rolloff = power concentrated at low frequencies
    High rolloff = power spread to high frequencies
    
    Args:
        freqs: Frequency bins
        psd: Power spectral density (normalized)
        threshold: Cumulative power threshold (0 to 1)
        
    Returns:
        Rolloff frequency (0 to 0.5)
    """
    cumsum = np.cumsum(psd)
    idx = np.where(cumsum >= threshold)[0]
    
    if len(idx) == 0:
        return freqs[-1]  # All power at end
    
    return freqs[idx[0]]


def spectral_bandwidth(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Bandwidth around spectral centroid (sqrt of weighted variance).
    
    Wide bandwidth = power spread across frequencies
    Narrow bandwidth = power concentrated near centroid
    
    Args:
        freqs: Frequency bins
        psd: Power spectral density (normalized)
        
    Returns:
        Bandwidth (frequency units)
    """
    # Spectral centroid (weighted mean)
    centroid = np.sum(freqs * psd)
    
    # Weighted variance
    variance = np.sum(((freqs - centroid) ** 2) * psd)
    
    return np.sqrt(variance)


def high_freq_power(freqs: np.ndarray, psd: np.ndarray, cutoff: float = 0.15) -> float:
    """
    Power above high-frequency cutoff.
    
    High value = noisy/erratic behavior
    Low value = smooth/trending behavior
    
    Args:
        freqs: Frequency bins
        psd: Power spectral density (normalized)
        cutoff: Frequency cutoff (default: 0.15)
        
    Returns:
        High-frequency power (0 to 1)
    """
    high_mask = freqs > cutoff
    return np.sum(psd[high_mask])


def spectral_features(x0: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """
    Compute minimal spectral feature family (top-6).
    
    ⚠️ ANTI-LEAKAGE: Preprocessing uses PRE segment statistics only!
    
    Features computed:
    - log_low_high_pre, log_low_high_post, delta_log_low_high
    - spec_entropy_pre, spec_entropy_post, delta_spec_entropy
    
    Args:
        x0: Pre-break values (raw)
        x1: Post-break values (raw)
        
    Returns:
        Dictionary of 6 spectral features
    """
    # Handle short sequences gracefully
    if len(x0) < 8 or len(x1) < 8:
        return {
            'log_low_high_pre': 0.0,
            'log_low_high_post': 0.0,
            'delta_log_low_high': 0.0,
            'spec_entropy_pre': 0.0,
            'spec_entropy_post': 0.0,
            'delta_spec_entropy': 0.0,
        }
    
    # Standardize using PRE statistics only (leakage-safe)
    x0_scaled, x1_scaled = safe_standardize_pre_only(x0, x1)
    
    # Compute PSDs
    freqs_pre, psd_pre = compute_psd(x0_scaled)
    freqs_post, psd_post = compute_psd(x1_scaled)
    
    # Log low/high ratio
    low_high_pre = log_low_high_ratio(freqs_pre, psd_pre, cutoff=0.08)
    low_high_post = log_low_high_ratio(freqs_post, psd_post, cutoff=0.08)
    
    # Spectral entropy
    entropy_pre = spectral_entropy(psd_pre)
    entropy_post = spectral_entropy(psd_post)
    
    # Return exactly 6 features
    return {
        'log_low_high_pre': low_high_pre,
        'log_low_high_post': low_high_post,
        'delta_log_low_high': low_high_post - low_high_pre,
        'spec_entropy_pre': entropy_pre,
        'spec_entropy_post': entropy_post,
        'delta_spec_entropy': entropy_post - entropy_pre,
    }


def spectral_features_v2(x0: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """
    Compute spectral v2 feature family (6 additional features).
    
    ⚠️ ANTI-LEAKAGE: Preprocessing uses PRE segment statistics only!
    
    Features computed (all deltas):
    - delta_peak_ratio: Change in strongest peak power ratio
    - delta_flatness: Change in spectral flatness (tonality)
    - delta_flux: Change in spectral flux (bin-to-bin variation)
    - delta_rolloff50: Change in 50% rolloff frequency
    - delta_bandwidth: Change in bandwidth around centroid
    - delta_hf_power: Change in high-frequency power (>0.15)
    
    Args:
        x0: Pre-break values (raw)
        x1: Post-break values (raw)
        
    Returns:
        Dictionary of 6 spectral v2 features
    """
    # Handle short sequences gracefully
    if len(x0) < 16 or len(x1) < 16:
        return {
            'delta_peak_ratio': 0.0,
            'delta_flatness': 0.0,
            'delta_flux': 0.0,
            'delta_rolloff50': 0.0,
            'delta_bandwidth': 0.0,
            'delta_hf_power': 0.0,
        }
    
    # Standardize using PRE statistics only (leakage-safe)
    x0_scaled, x1_scaled = safe_standardize_pre_only(x0, x1)
    
    # Compute PSDs
    freqs_pre, psd_pre = compute_psd(x0_scaled)
    freqs_post, psd_post = compute_psd(x1_scaled)
    
    # Compute v2 features for both segments
    peak_pre = spectral_peak_ratio(psd_pre)
    peak_post = spectral_peak_ratio(psd_post)
    
    flat_pre = spectral_flatness(psd_pre)
    flat_post = spectral_flatness(psd_post)
    
    flux_pre = spectral_flux(psd_pre)
    flux_post = spectral_flux(psd_post)
    
    rolloff_pre = spectral_rolloff(freqs_pre, psd_pre, threshold=0.5)
    rolloff_post = spectral_rolloff(freqs_post, psd_post, threshold=0.5)
    
    bw_pre = spectral_bandwidth(freqs_pre, psd_pre)
    bw_post = spectral_bandwidth(freqs_post, psd_post)
    
    hf_pre = high_freq_power(freqs_pre, psd_pre, cutoff=0.15)
    hf_post = high_freq_power(freqs_post, psd_post, cutoff=0.15)
    
    # Return exactly 6 delta features
    return {
        'delta_peak_ratio': peak_post - peak_pre,
        'delta_flatness': flat_post - flat_pre,
        'delta_flux': flux_post - flux_pre,
        'delta_rolloff50': rolloff_post - rolloff_pre,
        'delta_bandwidth': bw_post - bw_pre,
        'delta_hf_power': hf_post - hf_pre,
    }


def spectral_features_all(
    x0: np.ndarray,
    x1: np.ndarray,
    include_v2: bool = True
) -> Dict[str, float]:
    """
    Compute all spectral features: base (6) + v2 (6).
    
    Args:
        x0: Pre-break values (raw)
        x1: Post-break values (raw)
        include_v2: Whether to include v2 features (default: True)
        
    Returns:
        Dictionary of 6 or 12 spectral features
    """
    features = spectral_features(x0, x1)
    
    if include_v2:
        features_v2 = spectral_features_v2(x0, x1)
        features.update(features_v2)
    
    return features


def spectral_features_deltas_only(x0: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """
    Compute only delta spectral features (no _pre/_post).
    
    Used for multiscale windowed features to avoid feature explosion.
    Returns 8 delta features: 2 from base + 6 from v2.
    
    Args:
        x0: Pre-break values (raw)
        x1: Post-break values (raw)
        
    Returns:
        Dictionary of 8 delta-only spectral features
    """
    # Get base deltas
    base_feats = spectral_features(x0, x1)
    deltas = {
        'delta_log_low_high': base_feats['delta_log_low_high'],
        'delta_spec_entropy': base_feats['delta_spec_entropy'],
    }
    
    # Get v2 deltas (all v2 features are deltas)
    v2_feats = spectral_features_v2(x0, x1)
    deltas.update(v2_feats)
    
    return deltas


if __name__ == "__main__":
    # Self-check: determinism and sanity
    print("=" * 60)
    print("SPECTRAL FEATURES SELF-CHECK")
    print("=" * 60)
    
    # Create deterministic random data
    np.random.seed(42)
    x0 = np.random.randn(100) + np.sin(np.linspace(0, 4*np.pi, 100))
    x1 = np.random.randn(100) * 1.5 + np.cos(np.linspace(0, 6*np.pi, 100))
    
    # Compute base features
    print("\n--- Base Features (6) ---")
    features = spectral_features(x0, x1)
    for name, value in features.items():
        print(f"  {name:25s} = {value:12.6f}")
    
    # Compute v2 features
    print("\n--- V2 Features (6) ---")
    features_v2 = spectral_features_v2(x0, x1)
    for name, value in features_v2.items():
        print(f"  {name:25s} = {value:12.6f}")
    
    # Compute all features
    print("\n--- All Features (12) ---")
    features_all = spectral_features_all(x0, x1, include_v2=True)
    for name, value in features_all.items():
        print(f"  {name:25s} = {value:12.6f}")
    
    # Compute deltas only
    print("\n--- Deltas Only (8) ---")
    features_deltas = spectral_features_deltas_only(x0, x1)
    for name, value in features_deltas.items():
        print(f"  {name:25s} = {value:12.6f}")
    
    # Check for NaNs and infinities
    all_values = list(features_all.values())
    has_nan = any(np.isnan(v) for v in all_values)
    has_inf = any(np.isinf(v) for v in all_values)
    
    print(f"\nSanity checks:")
    print(f"  All finite (no NaN):     {'✓' if not has_nan else '✗ FAIL'}")
    print(f"  All finite (no Inf):     {'✓' if not has_inf else '✗ FAIL'}")
    print(f"  Base features = 6:       {'✓' if len(features) == 6 else '✗ FAIL'}")
    print(f"  V2 features = 6:         {'✓' if len(features_v2) == 6 else '✗ FAIL'}")
    print(f"  All features = 12:       {'✓' if len(features_all) == 12 else '✗ FAIL'}")
    print(f"  Deltas only = 8:         {'✓' if len(features_deltas) == 8 else '✗ FAIL'}")
    
    # Test determinism
    features_all2 = spectral_features_all(x0, x1, include_v2=True)
    max_diff = max(abs(features_all[k] - features_all2[k]) for k in features_all.keys())
    
    print(f"  Deterministic (repeat):  {'✓' if max_diff == 0.0 else '✗ FAIL'}")
    print(f"    Max difference: {max_diff:.2e}")
    
    # Test short series handling
    x0_short = np.random.randn(10)
    x1_short = np.random.randn(10)
    features_short = spectral_features_all(x0_short, x1_short, include_v2=True)
    all_zero = all(v == 0.0 for v in features_short.values())
    
    print(f"  Short series (n=10):     {'✓' if all_zero else '✗ FAIL'}")
    print(f"    Returns all zeros: {all_zero}")
    
    # Test entropy bounds (should be 0 to 1)
    entropy_in_bounds = (
        0.0 <= features['spec_entropy_pre'] <= 1.0 and
        0.0 <= features['spec_entropy_post'] <= 1.0
    )
    print(f"  Entropy in [0,1]:        {'✓' if entropy_in_bounds else '✗ FAIL'}")
    
    # Test flatness bounds (should be 0 to 1)
    flatness_in_bounds = (
        -1.0 <= features_v2['delta_flatness'] <= 1.0
    )
    print(f"  Flatness delta in [-1,1]:{'✓' if flatness_in_bounds else '✗ FAIL'}")
    
    print("\n" + "=" * 60)
    print("SELF-CHECK COMPLETE")
    print("=" * 60)
