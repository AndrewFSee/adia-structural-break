"""Quick test of boundary_tail_shape features."""
import sys
sys.path.insert(0, 'src')

import numpy as np
from sb.features.boundary_tail_shape import compute_boundary_tail_shape_features

print("=" * 70)
print("TESTING BOUNDARY TAIL-SHAPE FEATURES")
print("=" * 70)

# Test 1: Basic functionality
print("\n Test 1: Basic functionality")
np.random.seed(42)
x0 = np.random.randn(200)
x1 = np.random.randn(200) + 1.0

features = compute_boundary_tail_shape_features(x0, x1, windows=(25, 50, 100))

# Count features
all_keys = list(features.keys())
bl_ts_keys = [k for k in all_keys if k.startswith('bl_ts_')]
loc_keys = [k for k in bl_ts_keys if '_loc_' in k]
dod_keys = [k for k in bl_ts_keys if '_dod_' in k]

print(f"  Total features: {len(features)}")
print(f"  bl_ts_ features: {len(bl_ts_keys)}")
print(f"  Localization features: {len(loc_keys)}")
print(f"  DoD features: {len(dod_keys)}")

# Check finite rate
n_finite = sum(1 for v in features.values() if np.isfinite(v))
finite_pct = 100 * n_finite / len(features)
print(f"  Finite features: {n_finite}/{len(features)} ({finite_pct:.1f}%)")

# Show some key features
print("\n  Sample features:")
sample_keys = [
    'bl_ts_hill_alpha_delta_q95_w50',
    'bl_ts_qspace_hi_delta_w50',
    'bl_ts_asym_delta_w50',
    'bl_ts_loc_diff_hill_alpha_delta_q95_w50',
    'bl_ts_dod_hill_alpha_delta_q95_slope_logw'
]
for key in sample_keys:
    if key in features:
        val = features[key]
        if np.isfinite(val):
            print(f"    {key}: {val:.4f}")
        else:
            print(f"    {key}: NaN")

# Test 2: Window ordering invariance
print("\n Test 2: Window ordering invariance")
features_sorted = compute_boundary_tail_shape_features(x0, x1, windows=(25, 50, 100))
features_shuffled = compute_boundary_tail_shape_features(x0, x1, windows=(100, 25, 50))

# Check a DoD feature
dod_key = 'bl_ts_dod_hill_alpha_delta_q95_delta_wmin_wmax'
if dod_key in features_sorted and dod_key in features_shuffled:
    val1 = features_sorted[dod_key]
    val2 = features_shuffled[dod_key]
    if np.isfinite(val1) and np.isfinite(val2):
        diff = abs(val1 - val2)
        print(f"  {dod_key}:")
        print(f"    Sorted: {val1:.6f}")
        print(f"    Shuffled: {val2:.6f}")
        print(f"    Difference: {diff:.2e}")
        if diff < 1e-8:
            print("    ✓ Window ordering is invariant")
        else:
            print("    ✗ WARNING: Window ordering affects results")
    else:
        print(f"  {dod_key}: NaN (skipping check)")

# Test 3: Heavy-tail detection
print("\n Test 3: Heavy-tail detection")
np.random.seed(123)
x0_norm = np.random.randn(200)
x1_heavy = np.random.randn(200)
# Add extreme spikes
spike_idx = np.random.choice(len(x1_heavy), size=15, replace=False)
x1_heavy[spike_idx] += np.random.uniform(4, 7, size=15)

features_heavy = compute_boundary_tail_shape_features(x0_norm, x1_heavy, windows=(50,))

key_metrics = [
    'bl_ts_p_hi_delta_t25_w50',
    'bl_ts_mean_excess_hi_delta_t25_w50',
    'bl_ts_hill_alpha_delta_q95_w50'
]

print("  Metrics (should be positive for heavy tail):")
for key in key_metrics:
    if key in features_heavy:
        val = features_heavy[key]
        if np.isfinite(val):
            status = "✓" if val > 0 else "✗"
            print(f"    {status} {key}: {val:.4f}")
        else:
            print(f"    ? {key}: NaN")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
