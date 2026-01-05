# Distance-of-Distance (DoD) Features

## Overview

DoD features are meta-features that analyze how distribution distance metrics change across different window sizes. These features capture scale-dependent patterns and provide higher-order signal about the nature of structural breaks.

## Feature Count

- **Base boundary_dist features**: 210 (30 base + 180 tail-restricted)
- **New DoD features**: 82
- **Total boundary_dist features**: 292

## DoD Feature Categories

### A) DoD Statistics (32 features)

Analyze how distances scale with window size for 4 metric vectors across 4 windows + full scope:

**Metrics analyzed:**
- `bl_wasserstein_z` (standardized Wasserstein distance)
- `bl_energy_z` (standardized Energy distance)  
- `bl_tail_wins_wasserstein_z` (tail-winsorized Wasserstein)
- `bl_tail_wins_energy_z` (tail-winsorized Energy)

**Statistics per metric (8 features × 4 metrics = 32 features):**
1. `{prefix}_slope_logw` - Linear regression slope of distance ~ log(window)
2. `{prefix}_intercept_logw` - Intercept of linear fit
3. `{prefix}_range` - max(distance) - min(distance) across windows
4. `{prefix}_std` - Standard deviation across windows
5. `{prefix}_cv` - Coefficient of variation (std / mean)
6. `{prefix}_delta_wmin_wmax` - distance[largest_w] - distance[smallest_w]
7. `{prefix}_ratio_wmin_wmax` - distance[largest_w] / distance[smallest_w]
8. `{prefix}_curv_wA_wB_wC` - Second-order curvature using first 3 windows

**Example features:**
```
bl_dod_wasserstein_z_slope_logw: 0.1100
bl_dod_wasserstein_z_range: 0.2831
bl_dod_wasserstein_z_cv: 0.1024
```

### B) Cross-Metric Agreement (20 features)

Measures agreement between Wasserstein and Energy distances:

**Per window (4 windows × 4 feature pairs = 16 features):**
- `bl_dod_absdiff_wassz_enez_w{w}` - Absolute difference
- `bl_dod_reldiff_wassz_enez_w{w}` - Relative difference
- `bl_dod_absdiff_wins_wassz_wins_enez_w{w}` - Winsorized absolute difference
- `bl_dod_reldiff_wins_wassz_wins_enez_w{w}` - Winsorized relative difference

**Full scope (4 features):**
- `bl_dod_absdiff_wassz_enez_full`
- `bl_dod_reldiff_wassz_enez_full`
- `bl_dod_absdiff_wins_wassz_wins_enez_full`
- `bl_dod_reldiff_wins_wassz_wins_enez_full`

**Example features:**
```
bl_dod_absdiff_wassz_enez_w25: 0.1956
bl_dod_reldiff_wassz_enez_w25: 0.1333
```

**Interpretation:** Low values indicate Wasserstein and Energy agree on the regime change magnitude.

### C) Outlier Sensitivity (10 features)

Quantifies how much outliers inflate distances (raw z - winsorized z):

**Per window (4 windows × 2 metrics = 8 features):**
- `bl_dod_outlier_sens_wassz_w{w}` - Wasserstein outlier impact
- `bl_dod_outlier_sens_enez_w{w}` - Energy outlier impact

**Full scope (2 features):**
- `bl_dod_outlier_sens_wassz_full`
- `bl_dod_outlier_sens_enez_full`

**Example feature:**
```
bl_dod_outlier_sens_wassz_w25: 0.0064
```

**Interpretation:** Positive values mean raw distances are larger (outliers present), negative means winsorized is larger (unlikely).

### D) Tail-vs-Bulk Consistency (20 features)

Compares tail-only distances with winsorized full-distribution distances:

**Per window (4 windows × 2 quantiles × 2 metrics = 16 features):**
- `bl_dod_tailbulk_wassz_{q90|q95}_both_w{w}` - Wasserstein tail consistency
- `bl_dod_tailbulk_enez_{q90|q95}_both_w{w}` - Energy tail consistency

**Full scope (4 features):**
- `bl_dod_tailbulk_wassz_{q90|q95}_both_full`
- `bl_dod_tailbulk_enez_{q90|q95}_both_full`

**Example feature:**
```
bl_dod_tailbulk_wassz_q90_both_w25: -0.2577
```

**Interpretation:** 
- Positive: Tail-only distance > winsorized (tail dominates)
- Negative: Winsorized > tail-only (bulk also changing)
- Near zero: Consistent across tail and bulk

## Use Cases

1. **Scale-dependent detection**: `slope_logw` captures if regime change grows with window size
2. **Metric robustness**: Cross-metric agreement shows if signal is consistent across distance types
3. **Outlier diagnosis**: Outlier sensitivity quantifies extreme value impact
4. **Regime characterization**: Tail-vs-bulk consistency identifies whether breaks are:
   - Tail-driven (positive consistency)
   - Bulk-driven (negative consistency)
   - Uniform (near-zero consistency)

## Implementation Details

### Helper Functions

```python
def _collect_window_series(features, key_pattern, windows):
    """Collect valid (finite) feature values across windows."""
    # Returns (logw, d) arrays for DoD analysis
    
def _dod_stats(logw, d, prefix, eps=1e-12):
    """Compute 8 DoD statistics from distance series."""
    # Returns dict with slope, intercept, range, std, cv, delta, ratio, curvature
```

### Robustness

- All features initialize to NaN for missing data
- Finite-value filtering at collection stage
- Safe division with epsilon for relative metrics
- Minimum sample requirements (n >= 2 for most stats, n >= 3 for curvature)

## Validation

Self-check output:
```
DoD Statistics:
  bl_dod_wasserstein_z_slope_logw: 0.1100
  bl_dod_wasserstein_z_range: 0.2831
  bl_dod_wasserstein_z_cv: 0.1024

Cross-metric agreement:
  bl_dod_absdiff_wassz_enez_w25: 0.1956
  bl_dod_reldiff_wassz_enez_w25: 0.1333

Outlier sensitivity:
  bl_dod_outlier_sens_wassz_w25: 0.0064

Tail-vs-bulk consistency:
  bl_dod_tailbulk_wassz_q90_both_w25: -0.2577

Total DoD features: 82
Total features: 292
```

All DoD features compute successfully and are finite for normal test cases.
