"""Quick test of spectral v2 implementation."""

import numpy as np
from src.sb.features import spectral

# Create test data
np.random.seed(42)
x0 = np.random.randn(100) + np.sin(np.linspace(0, 4*np.pi, 100))
x1 = np.random.randn(100) * 1.5 + np.cos(np.linspace(0, 6*np.pi, 100))

print("=" * 70)
print("SPECTRAL V2 TEST")
print("=" * 70)

# Test base features (v1)
print("\n1. Base features (v1) - 6 features:")
feats_v1 = spectral.spectral_features(x0, x1)
for name, val in feats_v1.items():
    print(f"   {name:25s} = {val:12.6f}")

# Test v2 features
print("\n2. V2 features - 6 features:")
feats_v2 = spectral.spectral_features_v2(x0, x1)
for name, val in feats_v2.items():
    print(f"   {name:25s} = {val:12.6f}")

# Test combined
print("\n3. All features (v1 + v2) - 12 features:")
feats_all = spectral.spectral_features_all(x0, x1, include_v2=True)
print(f"   Total count: {len(feats_all)}")
assert len(feats_all) == 12, f"Expected 12, got {len(feats_all)}"

# Test deltas only
print("\n4. Deltas only - 8 features:")
feats_deltas = spectral.spectral_features_deltas_only(x0, x1)
for name, val in feats_deltas.items():
    print(f"   {name:25s} = {val:12.6f}")
print(f"   Total count: {len(feats_deltas)}")
assert len(feats_deltas) == 8, f"Expected 8, got {len(feats_deltas)}"

# Verify all finite
all_vals = list(feats_all.values())
assert all(np.isfinite(v) for v in all_vals), "Some values are NaN or Inf!"

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print("\nExpected feature counts:")
print("  - Single-scale + spectral: 23 + 12 = 35 features")
print("  - Multiscale + spectral:")
print("    * Full scale: 23 base + 12 spectral = 35")
print("    * 3 windows: 3 × (23 base + 8 spectral deltas) = 93")
print("    * Total: 35 + 93 = 128 features")
