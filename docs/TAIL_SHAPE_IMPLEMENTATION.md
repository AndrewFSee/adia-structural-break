# Boundary Tail-Shape Features Implementation Summary

## Overview
Implemented comprehensive boundary-localized tail-shape change features for structural break detection in the ADIA project. This module complements the existing `boundary_dist.py` (distribution distances) by characterizing how tail shape characteristics change near the structural break boundary.

## Implementation Details

### 1. Module: `src/sb/features/boundary_tail_shape.py`

**Feature Families** (prefix: `bl_ts_`):

#### A. Hill Tail Index (Tail Heaviness)
- **Upper tail**: q90, q95 fractions
- **Lower tail**: q90, q95 fractions  
- **Metrics per scope**: 
  - `alpha0`, `alpha1` (pre/post estimates)
  - `delta` (signed difference)
  - `absdelta` (absolute difference)
  - `ndiff` (normalized difference)
- **Total**: 12 features per scope (6 upper + 6 lower)

#### B. Exceedance Probabilities & Mean Excess
- **Thresholds**: t=2.0, 2.5, 3.0 standard deviations
- **Metrics per threshold**:
  - `p_hi_delta`: Upper tail probability shift
  - `p_lo_delta`: Lower tail probability shift
  - `mean_excess_hi_delta`: Mean excess above threshold
  - `mean_excess_lo_delta`: Mean excess below threshold
- **Total**: 12 features per scope

#### C. Quantile Spacing Ratios
- **Upper tail**: `(q99-q95)/(q95-q50)`
- **Lower tail**: `(q50-q05)/(q05-q01)`
- **Metrics**: delta, absdelta, ndiff
- **Total**: 6 features per scope

#### D. Tail Asymmetry
- **Definition**: `(q99-q50) - (q50-q01)`
- **Metrics**: delta, absdelta, ndiff
- **Total**: 3 features per scope

#### E. Localization Features (Window vs Full)
- **Compared metrics**: hill_alpha_delta_q95, qspace_hi_delta, qspace_lo_delta, asym_delta
- **Comparisons per metric**: diff, absdiff, ndiff, ratio
- **Total**: 16 features per window (4 metrics × 4 comparisons)

#### F. DoD (Distance-of-Distance) Across Windows
- **Analyzed metrics**: hill_alpha_delta_q95, qspace_hi_delta, qspace_lo_delta, asym_delta
- **Statistics per metric**: 
  - `slope_logw`, `intercept_logw` (trend vs log(window))
  - `range`, `std`, `cv` (variation)
  - `delta_wmin_wmax` (largest - smallest window)
  - `ratio_wmin_wmax` (largest / smallest window)
  - `curv_wA_wB_wC` (curvature using 3 smallest windows)
- **Total**: 32 features (4 metrics × 8 statistics)

### 2. Feature Counts

**Default windows**: (25, 50, 100, 250)

- **Per window**: ~33 features (12 Hill + 12 exceedance + 6 qspace + 3 asym)
- **Per window (with localization)**: ~49 features (33 base + 16 localization)
- **Windows × 4**: ~196 features
- **Full segment**: ~33 features
- **DoD features**: ~32 features
- **Total**: ~261 features

### 3. Key Design Decisions

#### Robustness
- **NaN handling**: Clean inputs, safe guards on all computations
- **Minimum samples**: Require ≥10 for quantiles, ≥16 for Hill estimators
- **Numerical stability**: eps=1e-12 guards on divisions
- **Robust standardization**: Use MAD-based z-scores per segment

#### Window Ordering Invariance
- **Problem**: DoD features assumed sorted windows but didn't enforce sorting
- **Solution**: Added `_collect_window_series_ts()` with `np.argsort(logw)`
- **Result**: DoD features now consistent regardless of window tuple order

#### Leakage Safety
- **All features computed per-series from x0/x1 only**
- **No global statistics or cross-series information**
- **Standardization uses per-segment robust scale**

### 4. Integration

#### A. Feature Extraction (`src/sb/features/base.py`)
- Already integrated via `use_boundary_tail_shape` parameter
- Called in `compute_single_series_features()`:
  ```python
  if use_boundary_tail_shape:
      windows = config.MULTI_SCALE_WINDOWS if hasattr(config, 'MULTI_SCALE_WINDOWS') else (25, 50, 100, 250)
      boundary_tail_shape_feats = boundary_tail_shape.compute_boundary_tail_shape_features(x0, x1, windows=windows)
      features.update(boundary_tail_shape_feats)
  ```

#### B. CLI Flags (Already Present)
- `scripts/diagnostic_baseline.py`: `--boundary-tail-shape`
- `scripts/train_local.py`: `--boundary-tail-shape`
- `scripts/infer_local.py`: `--boundary-tail-shape`

#### C. Diagnostic Output (`scripts/diagnostic_baseline.py`)
Enhanced diagnostics to show:
```
Boundary tail-shape features: YES
  → 261 tail-shape features (bl_ts_*), X.X% NaN avg
    • 64 localization features (window vs full)
    • 32 DoD features (statistics across windows)
```

### 5. Testing & Validation

#### Self-Check Tests (in `__main__`)
1. **Heavy-tail detection**: Spikes in x1 should increase exceedance probabilities
2. **Location shift stability**: Pure shifts shouldn't affect standardized tail-shape metrics
3. **Window ordering invariance**: Shuffled windows produce identical DoD features

#### Test Script (`test_tail_shape.py`)
- Feature count validation
- Finite rate check
- Window ordering verification
- Heavy-tail detection confirmation

### 6. Usage Examples

#### Basic Usage
```python
from sb.features.boundary_tail_shape import compute_boundary_tail_shape_features
import numpy as np

x0 = np.random.randn(200)  # Pre-break
x1 = np.random.randn(200) + 1  # Post-break

features = compute_boundary_tail_shape_features(x0, x1, windows=(25, 50, 100, 250))
```

#### Via Pipeline
```bash
# Training
python scripts/train_local.py --boundary-tail-shape --mode xgb

# Diagnostics
python scripts/diagnostic_baseline.py --boundary-tail-shape

# Inference
python scripts/infer_local.py --boundary-tail-shape --model-path output/model.pkl
```

### 7. Feature Naming Convention

**Prefix**: `bl_ts_` (boundary-localized tail-shape)

**Structure**:
- Base: `bl_ts_{metric}_{scope}`
- Delta: `bl_ts_{metric}_delta_{scope}`
- Variants: `_absdelta`, `_ndiff` suffixes
- Localization: `bl_ts_loc_{comparison}_{metric}_{scope}`
- DoD: `bl_ts_dod_{metric}_{statistic}`

**Examples**:
- `bl_ts_hill_alpha_delta_q95_w50`
- `bl_ts_qspace_hi_absdelta_full`
- `bl_ts_loc_ratio_hill_alpha_delta_q95_w100`
- `bl_ts_dod_asym_delta_slope_logw`

### 8. Comparison with boundary_dist.py

| Aspect | boundary_dist.py | boundary_tail_shape.py |
|--------|------------------|------------------------|
| Focus | Distribution distances | Tail shape characteristics |
| Main metrics | Wasserstein, Energy | Hill alpha, quantile spacing |
| Standardization | Pooled robust scale | Per-segment robust scale |
| Features | 396 | 261 |
| Prefix | `bl_`, `bl_tail_` | `bl_ts_` |
| Tail treatment | Tail-restricted distances | Tail heaviness & curvature |

Both modules are:
- ✅ Leakage-free
- ✅ NaN-robust
- ✅ Numerically stable
- ✅ Window-order invariant (for DoD features)

### 9. Future Enhancements (Optional)

- **Generalized Pareto Distribution (GPD)** fitting for tail modeling
- **Tail dependence measures** (copula-based) for joint tail behavior
- **Conditional tail expectations** for risk assessment
- **Tail correlation** between windows
- **Non-parametric tail estimators** (Pickands, moment-based)

### 10. Files Modified

1. ✅ `src/sb/features/boundary_tail_shape.py` - Complete rewrite with new features
2. ✅ `scripts/diagnostic_baseline.py` - Added diagnostic output for bl_ts_* features
3. ✅ `test_tail_shape.py` - Created test script
4. ✅ `src/sb/features/base.py` - Already integrated (no changes needed)
5. ✅ CLI scripts - Already have `--boundary-tail-shape` flag (no changes needed)

### 11. Verification Checklist

- [x] Module imports successfully
- [x] Features compute without errors
- [x] NaN rates are reasonable
- [x] Window ordering is invariant
- [x] Heavy-tail detection works
- [x] Location shift stability works
- [x] Localization features computed
- [x] DoD features computed
- [x] CLI flag exists and propagates
- [x] Diagnostic output shows feature counts
- [x] Integration with base.py complete

## Summary

Implemented a comprehensive suite of 261 boundary-localized tail-shape features that complement the existing distribution distance features. The implementation is:

1. **Robust**: Handles NaNs, minimum samples, numerical stability
2. **Correct**: Window-order invariant DoD features
3. **Leakage-free**: All per-series computations
4. **Integrated**: Wired into CLI, diagnostics, and feature extraction pipeline
5. **Tested**: Self-checks and test script validate behavior
6. **Documented**: Clear naming conventions and comprehensive docstrings

The features capture tail heaviness (Hill estimator), exceedance behavior, quantile curvature, asymmetry, localization (window vs full), and multi-scale trends (DoD), providing rich signal for structural break detection in heavy-tailed or regime-changing time series.
