"""
Wavelet (time-frequency) features for structural break detection.

Uses discrete wavelet transform to capture multi-resolution signal changes.
Wavelet features are sensitive to both frequency content AND temporal location,
making them ideal for detecting structural breaks.

All features are leakage-safe: preprocessing uses PRE segment statistics only.
"""

import numpy as np
from typing import Dict

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


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


def haar_dwt(x: np.ndarray, level: int = 1) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Simple Haar wavelet transform (manual implementation).
    
    Used as fallback if pywt is not available.
    
    Args:
        x: Input signal (must have length power of 2, or will be padded)
        level: Decomposition level
        
    Returns:
        approx: Approximation coefficients at final level
        details: List of detail coefficients [D1, D2, ..., D_level]
    """
    # Pad to next power of 2 if needed
    n = len(x)
    n_pad = 2 ** int(np.ceil(np.log2(n)))
    if n_pad > n:
        x = np.pad(x, (0, n_pad - n), mode='constant')
    
    details = []
    approx = x.copy()
    
    for _ in range(level):
        n_curr = len(approx)
        if n_curr < 2:
            break
        
        # Haar decomposition: average and difference
        approx_new = (approx[::2] + approx[1::2]) / np.sqrt(2)
        detail = (approx[::2] - approx[1::2]) / np.sqrt(2)
        
        details.append(detail)
        approx = approx_new
    
    return approx, details


def compute_dwt(x: np.ndarray, wavelet: str = 'db4', level: int = 3) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Compute discrete wavelet transform.
    
    Uses pywt if available, otherwise falls back to Haar.
    
    Args:
        x: Input signal
        wavelet: Wavelet name (e.g., 'db4', 'haar')
        level: Decomposition level
        
    Returns:
        approx: Approximation coefficients
        details: List of detail coefficients [D1, D2, ..., D_level]
    """
    if HAS_PYWT:
        # Use pywt for better wavelets
        coeffs = pywt.wavedec(x, wavelet, level=level)
        approx = coeffs[0]
        details = coeffs[1:]  # [cD1, cD2, ..., cD_level]
        return approx, details
    else:
        # Fallback to simple Haar
        return haar_dwt(x, level=level)


def wavelet_entropy(details: list[np.ndarray]) -> float:
    """
    Compute normalized wavelet entropy from detail coefficients.
    
    High entropy = energy distributed across scales (complex signal)
    Low entropy = energy concentrated in few scales (simple signal)
    
    Args:
        details: List of detail coefficient arrays [D1, D2, ..., D_L]
        
    Returns:
        Normalized wavelet entropy (0 to 1)
    """
    if len(details) == 0:
        return 0.0
    
    # Compute energy per level
    energies = [np.sum(d ** 2) for d in details]
    total_energy = sum(energies) + 1e-12
    
    # Probability distribution
    probs = [e / total_energy for e in energies]
    
    # Shannon entropy
    entropy = -sum(p * np.log(p + 1e-12) for p in probs if p > 0)
    
    # Normalize by max possible entropy
    max_entropy = np.log(len(details))
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def energy_shares(details: list[np.ndarray]) -> tuple[float, float]:
    """
    Compute energy share of lowest and highest frequency bands.
    
    Args:
        details: List of detail coefficients [D1, D2, ..., D_L]
                 D1 = finest scale (highest freq)
                 D_L = coarsest scale (lowest freq)
        
    Returns:
        low_energy_share: Fraction of energy in coarsest scale
        high_energy_share: Fraction of energy in finest scale
    """
    if len(details) == 0:
        return 0.0, 0.0
    
    # Compute energy per level
    energies = [np.sum(d ** 2) for d in details]
    total_energy = sum(energies) + 1e-12
    
    # Shares
    high_share = energies[0] / total_energy  # D1 (finest)
    low_share = energies[-1] / total_energy  # D_L (coarsest)
    
    return low_share, high_share


def wavelet_features(x0: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """
    Compute wavelet (time-frequency) feature family.
    
    ⚠️ ANTI-LEAKAGE: Preprocessing uses PRE segment statistics only!
    
    Features computed (12 total):
    - wav_entropy_pre, wav_entropy_post, delta_wav_entropy
    - wav_low_energy_share_pre, wav_low_energy_share_post, delta_wav_low_energy_share
    - wav_high_energy_share_pre, wav_high_energy_share_post, delta_wav_high_energy_share
    - delta_wav_energy_l1, delta_wav_energy_l2, delta_wav_energy_l3
    
    Args:
        x0: Pre-break values (raw)
        x1: Post-break values (raw)
        
    Returns:
        Dictionary of 12 wavelet features
    """
    # Handle short sequences gracefully
    if len(x0) < 16 or len(x1) < 16:
        return {
            'wav_entropy_pre': 0.0,
            'wav_entropy_post': 0.0,
            'delta_wav_entropy': 0.0,
            'wav_low_energy_share_pre': 0.0,
            'wav_low_energy_share_post': 0.0,
            'delta_wav_low_energy_share': 0.0,
            'wav_high_energy_share_pre': 0.0,
            'wav_high_energy_share_post': 0.0,
            'delta_wav_high_energy_share': 0.0,
            'delta_wav_energy_l1': 0.0,
            'delta_wav_energy_l2': 0.0,
            'delta_wav_energy_l3': 0.0,
        }
    
    # Standardize using PRE statistics only (leakage-safe)
    x0_scaled, x1_scaled = safe_standardize_pre_only(x0, x1)
    
    # Compute wavelet decomposition (3 levels)
    _, details_pre = compute_dwt(x0_scaled, wavelet='db4', level=3)
    _, details_post = compute_dwt(x1_scaled, wavelet='db4', level=3)
    
    # Ensure we have 3 levels (pad with zeros if signal too short)
    while len(details_pre) < 3:
        details_pre.append(np.array([0.0]))
    while len(details_post) < 3:
        details_post.append(np.array([0.0]))
    
    # Wavelet entropy
    entropy_pre = wavelet_entropy(details_pre)
    entropy_post = wavelet_entropy(details_post)
    
    # Energy shares (low and high frequency bands)
    low_share_pre, high_share_pre = energy_shares(details_pre)
    low_share_post, high_share_post = energy_shares(details_post)
    
    # Per-level energy changes
    energies_pre = [np.sum(d ** 2) + 1e-12 for d in details_pre[:3]]
    energies_post = [np.sum(d ** 2) + 1e-12 for d in details_post[:3]]
    
    delta_energy_l1 = np.log(energies_post[0]) - np.log(energies_pre[0])
    delta_energy_l2 = np.log(energies_post[1]) - np.log(energies_pre[1])
    delta_energy_l3 = np.log(energies_post[2]) - np.log(energies_pre[2])
    
    # Return exactly 12 features
    return {
        'wav_entropy_pre': entropy_pre,
        'wav_entropy_post': entropy_post,
        'delta_wav_entropy': entropy_post - entropy_pre,
        'wav_low_energy_share_pre': low_share_pre,
        'wav_low_energy_share_post': low_share_post,
        'delta_wav_low_energy_share': low_share_post - low_share_pre,
        'wav_high_energy_share_pre': high_share_pre,
        'wav_high_energy_share_post': high_share_post,
        'delta_wav_high_energy_share': high_share_post - high_share_pre,
        'delta_wav_energy_l1': delta_energy_l1,
        'delta_wav_energy_l2': delta_energy_l2,
        'delta_wav_energy_l3': delta_energy_l3,
    }


if __name__ == "__main__":
    # Self-check: determinism and sanity
    print("=" * 70)
    print("WAVELET FEATURES SELF-CHECK")
    print("=" * 70)
    
    print(f"\nPyWavelets available: {HAS_PYWT}")
    
    # Create deterministic synthetic data
    np.random.seed(42)
    
    # Pre-break: smooth low-frequency signal
    t = np.linspace(0, 10, 200)
    x0 = np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.random.randn(200)
    
    # Post-break: higher frequency + noise
    x1 = np.sin(2 * np.pi * 2.0 * t) + 0.5 * np.random.randn(200)
    
    # Compute features
    print("\n--- Wavelet Features (12) ---")
    features = wavelet_features(x0, x1)
    for name, value in features.items():
        print(f"  {name:30s} = {value:12.6f}")
    
    # Check for NaNs and infinities
    all_values = list(features.values())
    has_nan = any(np.isnan(v) for v in all_values)
    has_inf = any(np.isinf(v) for v in all_values)
    
    print(f"\nSanity checks:")
    print(f"  All finite (no NaN):       {'✓' if not has_nan else '✗ FAIL'}")
    print(f"  All finite (no Inf):       {'✓' if not has_inf else '✗ FAIL'}")
    print(f"  Exactly 12 features:       {'✓' if len(features) == 12 else '✗ FAIL'}")
    
    # Test determinism
    features2 = wavelet_features(x0, x1)
    max_diff = max(abs(features[k] - features2[k]) for k in features.keys())
    
    print(f"  Deterministic (repeat):    {'✓' if max_diff < 1e-10 else '✗ FAIL'}")
    print(f"    Max difference: {max_diff:.2e}")
    
    # Test short series handling
    x0_short = np.random.randn(10)
    x1_short = np.random.randn(10)
    features_short = wavelet_features(x0_short, x1_short)
    all_zero = all(v == 0.0 for v in features_short.values())
    
    print(f"  Short series (n=10):       {'✓' if all_zero else '✗ FAIL'}")
    print(f"    Returns all zeros: {all_zero}")
    
    # Test entropy bounds (should be 0 to 1)
    entropy_in_bounds = (
        0.0 <= features['wav_entropy_pre'] <= 1.0 and
        0.0 <= features['wav_entropy_post'] <= 1.0
    )
    print(f"  Entropy in [0,1]:          {'✓' if entropy_in_bounds else '✗ FAIL'}")
    
    # Test energy shares sum approximately to ≤ 1
    share_sum_pre = features['wav_low_energy_share_pre'] + features['wav_high_energy_share_pre']
    share_sum_post = features['wav_low_energy_share_post'] + features['wav_high_energy_share_post']
    shares_valid = (0.0 <= share_sum_pre <= 1.0 and 0.0 <= share_sum_post <= 1.0)
    print(f"  Energy shares in [0,1]:    {'✓' if shares_valid else '✗ FAIL'}")
    
    print("\n" + "=" * 70)
    print("SELF-CHECK COMPLETE")
    print("=" * 70)
    print("\nExpected feature counts with wavelet:")
    print("  - Single-scale + wavelet: 23 + 12 = 35 features")
    print("  - Single-scale + spectral + wavelet: 23 + 12 + 12 = 47 features")
    print("  - Multiscale + spectral + wavelet (1 window):")
    print("    * Full: 23 base + 12 spectral + 12 wavelet = 47")
    print("    * 3 windows: 3 × (23 base + 8 spectral deltas) = 93")
    print("    * 1 boundary window: 1 × 12 wavelet = 12")
    print("    * Total: 47 + 93 + 12 = 152 features")
