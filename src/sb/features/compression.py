"""
Compression-based features for structural break detection.

Novel approach from 6th place solution (KaggleJM): Use compression algorithms
to measure the "predictability" or "complexity" of time series segments.

Key insight: If a segment is more structured/predictable, it compresses better.
Structural breaks often change the compressibility of the series.

Features:
- Z-lib compression ratios
- Lempel-Ziv complexity
- Normalized compression distance (NCD)
"""

import numpy as np
import zlib
from typing import Dict


def lempel_ziv_complexity(sequence: np.ndarray) -> int:
    """
    Compute Lempel-Ziv complexity of a sequence.
    
    LZ complexity measures the number of distinct patterns in a sequence.
    Higher complexity = less repetitive/predictable.
    
    Args:
        sequence: Input array (will be binarized)
        
    Returns:
        LZ complexity (integer)
    """
    # Convert to binary string (above/below median)
    if len(sequence) == 0:
        return 0
    
    median = np.median(sequence)
    binary = ''.join(['1' if x >= median else '0' for x in sequence])
    
    if len(binary) == 0:
        return 0
    
    # Compute LZ complexity
    n = len(binary)
    complexity = 1
    i = 0
    prefix_len = 1
    
    while prefix_len + i < n:
        # Check if substring binary[i:i+prefix_len] exists in binary[0:i+prefix_len]
        substring = binary[i:i+prefix_len]
        prefix = binary[0:i+prefix_len]
        
        if substring in prefix:
            prefix_len += 1
        else:
            complexity += 1
            i += prefix_len
            prefix_len = 1
    
    return complexity


def zlib_compression_ratio(sequence: np.ndarray) -> float:
    """
    Compute Z-lib compression ratio.
    
    Ratio = compressed_size / original_size
    Lower ratio = more compressible = more structured
    
    Args:
        sequence: Input array
        
    Returns:
        Compression ratio (0-1, lower is more compressible)
    """
    if len(sequence) == 0:
        return 1.0
    
    # Convert to bytes
    bytes_data = sequence.astype(np.float32).tobytes()
    
    # Compress
    compressed = zlib.compress(bytes_data, level=9)
    
    # Compute ratio
    ratio = len(compressed) / len(bytes_data)
    
    return ratio


def normalized_compression_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Compute Normalized Compression Distance (NCD) between two sequences.
    
    NCD is a similarity metric: 0 = identical, 1 = completely different
    
    NCD(x,y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))
    where C(x) is compressed size of x
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        NCD value (0-1)
    """
    if len(seq1) == 0 or len(seq2) == 0:
        return 1.0
    
    # Convert to bytes
    bytes1 = seq1.astype(np.float32).tobytes()
    bytes2 = seq2.astype(np.float32).tobytes()
    bytes_combined = np.concatenate([seq1, seq2]).astype(np.float32).tobytes()
    
    # Compress
    c1 = len(zlib.compress(bytes1, level=9))
    c2 = len(zlib.compress(bytes2, level=9))
    c12 = len(zlib.compress(bytes_combined, level=9))
    
    # Compute NCD
    ncd = (c12 - min(c1, c2)) / max(c1, c2)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, ncd))


def compute_compression_features(x0: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """
    Compute compression-based features.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        
    Returns:
        Dictionary of compression features
    """
    features = {}
    
    # Z-lib compression ratios
    zlib_pre = zlib_compression_ratio(x0)
    zlib_post = zlib_compression_ratio(x1)
    
    features['zlib_pre'] = zlib_pre
    features['zlib_post'] = zlib_post
    features['zlib_diff'] = zlib_post - zlib_pre
    features['zlib_ratio'] = zlib_post / (zlib_pre + 1e-8)
    
    # Compressibility change (positive = became more compressible = more structured)
    features['compressibility_increase'] = zlib_pre - zlib_post
    
    # Lempel-Ziv complexity
    lz_pre = lempel_ziv_complexity(x0)
    lz_post = lempel_ziv_complexity(x1)
    
    features['lz_complexity_pre'] = lz_pre
    features['lz_complexity_post'] = lz_post
    features['lz_complexity_diff'] = lz_post - lz_pre
    features['lz_complexity_ratio'] = lz_post / (lz_pre + 1e-8)
    
    # Normalized LZ (by length)
    features['lz_density_pre'] = lz_pre / (len(x0) + 1)
    features['lz_density_post'] = lz_post / (len(x1) + 1)
    features['lz_density_diff'] = features['lz_density_post'] - features['lz_density_pre']
    
    # Normalized Compression Distance between pre/post
    features['ncd_pre_post'] = normalized_compression_distance(x0, x1)
    
    # Compare last part of pre vs first part of post (boundary region)
    boundary_size = min(100, len(x0) // 4, len(x1) // 4)
    if boundary_size > 10:
        x0_boundary = x0[-boundary_size:]
        x1_boundary = x1[:boundary_size]
        features['ncd_boundary'] = normalized_compression_distance(x0_boundary, x1_boundary)
    else:
        features['ncd_boundary'] = features['ncd_pre_post']
    
    # Compression consistency (is compression ratio stable within segments?)
    # Split pre/post into halves and check consistency
    if len(x0) >= 100:
        mid0 = len(x0) // 2
        zlib_pre_first = zlib_compression_ratio(x0[:mid0])
        zlib_pre_second = zlib_compression_ratio(x0[mid0:])
        features['zlib_pre_consistency'] = np.abs(zlib_pre_first - zlib_pre_second)
    else:
        features['zlib_pre_consistency'] = 0.0
    
    if len(x1) >= 100:
        mid1 = len(x1) // 2
        zlib_post_first = zlib_compression_ratio(x1[:mid1])
        zlib_post_second = zlib_compression_ratio(x1[mid1:])
        features['zlib_post_consistency'] = np.abs(zlib_post_first - zlib_post_second)
    else:
        features['zlib_post_consistency'] = 0.0
    
    # Consistency change (structural break often reduces consistency)
    features['zlib_consistency_change'] = (features['zlib_post_consistency'] - 
                                           features['zlib_pre_consistency'])
    
    return features


def compute_compression_features_multiscale(x0: np.ndarray, x1: np.ndarray,
                                            windows=[50, 100, 250]) -> Dict[str, float]:
    """
    Compute compression features at multiple scales.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        windows: List of window sizes
        
    Returns:
        Dictionary of compression features at multiple scales
    """
    features = {}
    
    # Full-scale features
    full_feats = compute_compression_features(x0, x1)
    for k, v in full_feats.items():
        features[f'{k}_full'] = v
    
    # Multi-scale features (boundary-focused)
    for w in windows:
        # Last w points of x0, first w points of x1
        x0_w = x0[-w:] if len(x0) >= w else x0
        x1_w = x1[:w] if len(x1) >= w else x1
        
        if len(x0_w) > 10 and len(x1_w) > 10:  # Need minimum data for compression
            w_feats = compute_compression_features(x0_w, x1_w)
            for k, v in w_feats.items():
                features[f'{k}_w{w}'] = v
    
    return features


if __name__ == '__main__':
    # Test compression features
    import time
    
    print("Testing compression features...")
    print("="*70)
    
    # Test case 1: Structured vs random
    np.random.seed(42)
    x_structured = np.sin(np.linspace(0, 4*np.pi, 500))  # Predictable sine wave
    x_random = np.random.normal(0, 1, 500)  # Random noise
    
    zlib_structured = zlib_compression_ratio(x_structured)
    zlib_random = zlib_compression_ratio(x_random)
    
    print(f"Z-lib compression ratio:")
    print(f"  Structured (sine): {zlib_structured:.4f}")
    print(f"  Random (noise):    {zlib_random:.4f}")
    print(f"  Difference:        {zlib_random - zlib_structured:.4f}")
    print(f"  ✓ Random data is less compressible (higher ratio)\n")
    
    # Test case 2: Structural break
    x0 = np.random.normal(0, 1, 500)  # Noisy
    x1 = np.sin(np.linspace(0, 4*np.pi, 500))  # Structured
    
    features = compute_compression_features(x0, x1)
    
    print(f"Compression features for break (noise → sine):")
    print(f"  zlib_pre:                 {features['zlib_pre']:.4f}")
    print(f"  zlib_post:                {features['zlib_post']:.4f}")
    print(f"  compressibility_increase: {features['compressibility_increase']:.4f}")
    print(f"  lz_complexity_pre:        {features['lz_complexity_pre']}")
    print(f"  lz_complexity_post:       {features['lz_complexity_post']}")
    print(f"  ncd_pre_post:             {features['ncd_pre_post']:.4f}")
    print(f"  ✓ Post-break is more structured (lower zlib, higher NCD)\n")
    
    # Test speed
    start = time.time()
    for _ in range(100):
        compute_compression_features(x0, x1)
    elapsed = time.time() - start
    
    print(f"Speed test (100 iterations): {elapsed:.2f}s ({elapsed*10:.1f}ms per sample)")
    print("="*70)
