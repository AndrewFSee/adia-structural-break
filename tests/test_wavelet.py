"""Quick test of wavelet implementation."""

import sys
sys.path.insert(0, 'c:/Users/Andrew/projects/adia_structural_break')

import numpy as np
from src.sb.features import wavelet

# Test wavelet module
print("=" * 70)
print("Testing wavelet module...")
print("=" * 70)

# Create test data
np.random.seed(42)
t = np.linspace(0, 10, 200)
x0 = np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.random.randn(200)
x1 = np.sin(2 * np.pi * 2.0 * t) + 0.5 * np.random.randn(200)

# Compute features
features = wavelet.wavelet_features(x0, x1)

print(f"\nComputed {len(features)} features:")
for name, val in features.items():
    print(f"  {name:30s} = {val:12.6f}")

# Verify
assert len(features) == 12, f"Expected 12 features, got {len(features)}"
assert all(np.isfinite(v) for v in features.values()), "Some features are NaN or Inf!"

print("\n✓ All checks passed!")
print("\nExpected feature counts:")
print("  Single-scale + wavelet: 23 + 12 = 35")
print("  Multiscale + spectral + wavelet:")
print("    Full: 23 + 12 spectral + 12 wavelet = 47")
print("    3 windows: 3 × (23 + 8 spectral) = 93")
print("    1 boundary wavelet: 12")
print("    Total: 47 + 93 + 12 = 152 features")
