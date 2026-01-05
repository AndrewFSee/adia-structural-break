# Feature Naming Conventions

**Last Updated**: December 30, 2025

This document provides a comprehensive reference for all feature naming conventions used in the ADIA structural break detection project.

---

## Table of Contents

1. [General Principles](#general-principles)
2. [Boundary-Localized Features](#boundary-localized-features)
3. [Distribution & Dynamics Features](#distribution--dynamics-features)
4. [Spectral & Wavelet Features](#spectral--wavelet-features)
5. [Model-Based Features](#model-based-features)
6. [Suffix Conventions](#suffix-conventions)
7. [Quick Reference Table](#quick-reference-table)

---

## General Principles

### Naming Structure
```
{prefix}_{metric}_{variant}_{scope}_{suffix}
```

**Components:**
- **prefix**: Feature family identifier (e.g., `bl_`, `bl_ts_`, `spec_`)
- **metric**: Core measurement (e.g., `wasserstein`, `energy`, `hill_alpha`)
- **variant**: Modification or computation type (e.g., `z` for standardized, `delta` for difference)
- **scope**: Window size or segment (e.g., `w25`, `w50`, `full`)
- **suffix**: Additional qualifiers (e.g., `upper`, `lower`, `q95`)

### Design Principles
1. **Leakage-free**: All features computed per-series only
2. **NaN-robust**: Safe handling of missing/invalid data
3. **Numerically stable**: eps guards on divisions, robust scales
4. **Consistent naming**: Predictable patterns across feature families

---

## Boundary-Localized Features

### Module: `boundary_dist.py`

#### 1. Base Distance Features
**Prefix**: `bl_`

Compares distributions at the boundary between pre-break (x0) and post-break (x1) segments.

**Patterns:**
```
bl_{metric}_w{window}          # Raw distance at boundary window
bl_{metric}_z_w{window}        # Standardized (by robust scale)
bl_{metric}_full               # Full segment comparison
bl_mean_delta_w{window}        # Location shift
bl_median_delta_w{window}      # Median shift
```

**Examples:**
- `bl_wasserstein_w25` - Wasserstein distance, 25-sample boundary window
- `bl_energy_z_w50` - Standardized Energy distance, 50-sample window
- `bl_wasserstein_full` - Wasserstein over full segments
- `bl_mean_delta_w100` - Mean shift at 100-sample boundary

**Metrics:**
- `wasserstein` - Wasserstein (earth mover's) distance
- `energy` - Energy distance
- `mean_delta` - Change in mean
- `median_delta` - Change in median

#### 2. Tail-Restricted Features
**Prefix**: `bl_tail_`

Focuses on extreme values and tail behavior.

**Patterns:**
```
bl_tail_{metric}_q{quantile}_{side}_{scope}           # Tail-only distances
bl_tail_{metric}_z_q{quantile}_{side}_{scope}         # Standardized tail distances
bl_tail_wins_{metric}_{scope}                          # Winsorized distances
bl_tail_wins_{metric}_z_{scope}                        # Standardized winsorized
bl_tail_p_{hi|lo}_delta_q{quantile}_{scope}           # Exceedance probability
bl_tail_mean_excess_{hi|lo}_delta_q{quantile}_{scope} # Mean excess beyond threshold
```

**Quantiles:**
- `q90` - Top/bottom 10% (frac=0.10)
- `q95` - Top/bottom 5% (frac=0.05)

**Sides:**
- `upper` - Upper tail only
- `lower` - Lower tail only
- `both` - Both tails combined

**Examples:**
- `bl_tail_wasserstein_q95_upper_w25` - Wasserstein on top 5%, 25-sample window
- `bl_tail_energy_z_q90_both_w50` - Standardized Energy on both tails, 50-sample window
- `bl_tail_wins_wasserstein_w100` - Wasserstein on winsorized data (clipped at q05/q95)
- `bl_tail_p_hi_delta_q95_w25` - Change in probability of exceeding 95th percentile
- `bl_tail_mean_excess_hi_delta_q90_w50` - Change in mean value above 90th percentile

#### 3. Distance-of-Distance (DoD) Features
**Prefix**: `bl_dod_`

Analyzes how distances change across window scales (multi-scale analysis).

**Patterns:**
```
bl_dod_{metric}_slope_logw           # Trend vs log(window)
bl_dod_{metric}_intercept_logw       # Intercept of trend
bl_dod_{metric}_range                # max - min across windows
bl_dod_{metric}_std                  # Standard deviation
bl_dod_{metric}_cv                   # Coefficient of variation
bl_dod_{metric}_delta_wmin_wmax      # Largest - smallest window
bl_dod_{metric}_ratio_wmin_wmax      # Largest / smallest window
bl_dod_{metric}_curv_wA_wB_wC        # Curvature (2nd derivative)
```

**Cross-Metric Comparisons:**
```
bl_dod_absdiff_{metric1}_{metric2}_{scope}    # |m1 - m2|
bl_dod_reldiff_{metric1}_{metric2}_{scope}    # (m1-m2)/(|m1|+|m2|)
bl_dod_diff_{metric1}_{metric2}_{scope}       # Signed difference
bl_dod_ratio_{metric1}_{metric2}_{scope}      # m1 / m2
bl_dod_ndiff_{metric1}_{metric2}_{scope}      # Normalized difference
bl_dod_logratio_{metric1}_{metric2}_{scope}   # log(m1/m2)
```

**Outlier Sensitivity:**
```
bl_dod_outlier_sens_{metric}_{scope}          # raw_z - winsorized_z
```

**Tail-vs-Bulk Consistency:**
```
bl_dod_tailbulk_{metric}_q{quantile}_{side}_{scope}
```

**Examples:**
- `bl_dod_wasserstein_z_slope_logw` - How standardized Wasserstein changes with window size
- `bl_dod_energy_z_delta_wmin_wmax` - Difference between largest and smallest window
- `bl_dod_absdiff_wassz_enez_w50` - Agreement between Wasserstein and Energy at w50
- `bl_dod_outlier_sens_wassz_w25` - Impact of outliers on Wasserstein at w25

#### 4. Localization Features
**Prefix**: `bl_loc_`

Compares boundary window vs full segment to detect localized changes.

**Patterns:**
```
bl_loc_diff_{metric}_{scope}         # window - full
bl_loc_absdiff_{metric}_{scope}      # |window - full|
bl_loc_ndiff_{metric}_{scope}        # Normalized difference
bl_loc_ratio_{metric}_{scope}        # window / full
```

**Examples:**
- `bl_loc_diff_wassz_w50` - How much more (or less) signal at boundary vs full
- `bl_loc_ratio_enez_w100` - Boundary intensity relative to full segment

---

### Module: `boundary_tail_shape.py`

#### 1. Tail-Shape Base Features
**Prefix**: `bl_ts_`

Characterizes tail shape (heaviness, curvature, asymmetry).

**Hill Tail Index (Heaviness):**
```
bl_ts_hill_alpha0_q{quantile}_{scope}          # Pre-break tail index
bl_ts_hill_alpha1_q{quantile}_{scope}          # Post-break tail index
bl_ts_hill_alpha_delta_q{quantile}_{scope}     # Signed difference
bl_ts_hill_alpha_absdelta_q{quantile}_{scope}  # Absolute difference
bl_ts_hill_alpha_ndiff_q{quantile}_{scope}     # Normalized difference
bl_ts_hill_alpha_delta_q{quantile}_lower_{scope}  # Lower tail version
```

**Exceedance Metrics:**
```
bl_ts_p_{hi|lo}_delta_t{threshold}_{scope}              # Probability shift
bl_ts_mean_excess_{hi|lo}_delta_t{threshold}_{scope}    # Mean excess shift
```

**Quantile Spacing Ratios:**
```
bl_ts_qspace_{hi|lo}_delta_{scope}       # (q99-q95)/(q95-q50) for hi
bl_ts_qspace_{hi|lo}_absdelta_{scope}    # Absolute version
bl_ts_qspace_{hi|lo}_ndiff_{scope}       # Normalized version
```

**Tail Asymmetry:**
```
bl_ts_asym_delta_{scope}                 # (q99-q50) - (q50-q01)
bl_ts_asym_absdelta_{scope}              # Absolute version
bl_ts_asym_ndiff_{scope}                 # Normalized version
```

**Thresholds:**
- `t20` - 2.0 standard deviations
- `t25` - 2.5 standard deviations
- `t30` - 3.0 standard deviations

**Examples:**
- `bl_ts_hill_alpha_delta_q95_w50` - Change in tail heaviness (q95) at 50-sample window
- `bl_ts_p_hi_delta_t25_w100` - Change in probability of exceeding 2.5σ
- `bl_ts_qspace_hi_delta_full` - Change in upper tail spacing ratio over full segment
- `bl_ts_asym_delta_w25` - Change in tail asymmetry at 25-sample window

#### 2. Tail-Shape Localization
**Prefix**: `bl_ts_loc_`

Compares tail-shape features at boundary vs full segment.

**Patterns:**
```
bl_ts_loc_diff_{metric}_{scope}          # window - full
bl_ts_loc_absdiff_{metric}_{scope}       # |window - full|
bl_ts_loc_ndiff_{metric}_{scope}         # Normalized difference
bl_ts_loc_ratio_{metric}_{scope}         # window / full
```

**Examples:**
- `bl_ts_loc_diff_hill_alpha_delta_q95_w50` - How tail heaviness change differs at boundary
- `bl_ts_loc_ratio_qspace_hi_delta_w100` - Boundary vs full tail spacing intensity

#### 3. Tail-Shape DoD Features
**Prefix**: `bl_ts_dod_`

Multi-scale analysis of tail-shape metrics across windows.

**Patterns:**
```
bl_ts_dod_{metric}_slope_logw
bl_ts_dod_{metric}_intercept_logw
bl_ts_dod_{metric}_range
bl_ts_dod_{metric}_std
bl_ts_dod_{metric}_cv
bl_ts_dod_{metric}_delta_wmin_wmax
bl_ts_dod_{metric}_ratio_wmin_wmax
bl_ts_dod_{metric}_curv_wA_wB_wC
```

**Examples:**
- `bl_ts_dod_hill_alpha_delta_q95_slope_logw` - Trend in tail heaviness across scales
- `bl_ts_dod_asym_delta_delta_wmin_wmax` - Change in asymmetry from smallest to largest window

---

## Distribution & Dynamics Features

### Module: `dist.py`

**No consistent prefix** - legacy features with varied naming.

**Common patterns:**
```
delta_entropy                    # Entropy change
energy                           # Energy distance (1D)
wasserstein                      # Wasserstein distance (1D)
q{percentile}_delta              # Quantile difference (e.g., q10_delta, q50_delta, q90_delta)
mad_ratio                        # MAD(x1) / MAD(x0)
iqr_ratio_robust                 # IQR ratio
acf1_shift                       # ACF-1 change
```

### Module: `dynamics.py`

**No consistent prefix** - volatility and transition features.

**Patterns:**
```
vol_pre                          # Pre-break volatility
vol_post                         # Post-break volatility
vol_ratio                        # vol_post / vol_pre
transition_vol                   # Volatility near transition
```

### Module: `multiscale.py`

**Window-specific features** - includes scale information in suffix.

**Patterns:**
```
{base_feature}_w{window}         # Feature at specific window size
```

---

## Spectral & Wavelet Features

### Module: `spectral.py`

**Prefix**: `spec_` (v2 features)

**Patterns:**
```
spec_power_shift_band{n}         # Power change in frequency band
spec_peak_freq_shift             # Dominant frequency shift
spec_bandwidth_ratio             # Bandwidth change
spec_spectral_entropy_delta      # Entropy change in frequency domain
```

**Legacy (no prefix):**
```
freq_peak_shift                  # Peak frequency change
power_ratio                      # Power ratio
spectral_centroid_shift          # Centroid shift
```

### Module: `wavelet.py`

**Prefix**: `wav_`

**Patterns:**
```
wav_energy_delta_d{level}        # Wavelet energy change at decomposition level
wav_entropy_delta                # Wavelet entropy change
wav_scale_shift                  # Characteristic scale change
```

---

## Model-Based Features

### Module: `ar_features.py`

**Prefix**: `ar_` or `ar1_`

**Patterns:**
```
ar1_coef_pre                     # AR(1) coefficient pre-break
ar1_coef_post                    # AR(1) coefficient post-break
ar1_coef_delta                   # Coefficient change
ar_predict_mse_shift             # Prediction error shift
```

### Module: `ar_kalman_features.py`

**Prefix**: `ark_`

**Patterns:**
```
ark_transition_vol               # Transition volatility
ark_state_shift                  # State change magnitude
ark_smoothness                   # Smoothness metric
```

---

## Suffix Conventions

### Window/Scope Suffixes
```
_w{size}                         # Boundary window (e.g., _w25, _w50, _w100, _w250)
_full                            # Full segment
_pre                             # Pre-break segment only
_post                            # Post-break segment only
```

### Quantile Suffixes
```
_q{percentile}                   # Quantile level (e.g., _q90, _q95, _q99)
```

### Side Suffixes (for tail features)
```
_upper                           # Upper tail
_lower                           # Lower tail
_both                            # Both tails
_hi                              # High side
_lo                              # Low side
```

### Threshold Suffixes (for exceedances)
```
_t{value}                        # Standard deviation threshold (e.g., _t20, _t25, _t30)
_k{value}                        # Robust SD threshold (e.g., _k25, _k35)
```

### Variant Suffixes
```
_z                               # Standardized (z-score or robust scale)
_delta                           # Signed difference (post - pre)
_absdelta                        # Absolute difference
_ndiff                           # Normalized difference: (post-pre)/(|post|+|pre|)
_ratio                           # Ratio (post / pre)
_logratio                        # Log ratio: log(post/pre)
```

### Statistical Suffixes (for DoD)
```
_slope_logw                      # Slope vs log(window)
_intercept_logw                  # Intercept
_range                           # max - min
_std                             # Standard deviation
_cv                              # Coefficient of variation
_delta_wmin_wmax                 # Largest - smallest
_ratio_wmin_wmax                 # Largest / smallest
_curv_wA_wB_wC                   # Curvature (second derivative)
```

---

## Quick Reference Table

### Feature Family Overview

| Prefix | Module | Count | Description |
|--------|--------|-------|-------------|
| `bl_` | boundary_dist.py | 30 | Base boundary distances |
| `bl_tail_` | boundary_dist.py | 180 | Tail-restricted distances |
| `bl_dod_` | boundary_dist.py | 122 | Distance-of-distance (multi-scale) |
| `bl_loc_` | boundary_dist.py | 64 | Localization (window vs full) |
| `bl_ts_` | boundary_tail_shape.py | 165 | Tail-shape characteristics |
| `bl_ts_loc_` | boundary_tail_shape.py | 64 | Tail-shape localization |
| `bl_ts_dod_` | boundary_tail_shape.py | 32 | Tail-shape DoD |
| `spec_` | spectral.py | ~20 | Spectral (v2) |
| `wav_` | wavelet.py | ~15 | Wavelet |
| `ar_`, `ar1_` | ar_features.py | ~10 | AR model-based |
| `ark_` | ar_kalman_features.py | ~8 | AR-Kalman |
| (none) | dist.py, dynamics.py | ~30 | Legacy base features |

### Total Feature Count by Module

| Module | Features | With Flag |
|--------|----------|-----------|
| Base (dist + dynamics + AR) | ~40 | Always included |
| boundary_dist.py | 396 | `--boundary-dist` |
| boundary_tail_shape.py | 281 | `--boundary-tail-shape` |
| multiscale.py | Variable | `--multiscale` |
| spectral.py | ~35 | `--spectral` |
| wavelet.py | ~15 | `--wavelet` |
| boundary.py | ~50 | `--boundary` |

---

## Best Practices

### 1. Adding New Features
When adding new features, follow these guidelines:

**Choose an appropriate prefix:**
- Boundary-localized? → `bl_*`
- Tail-specific? → `bl_tail_*` or `bl_ts_*`
- Multi-scale analysis? → `*_dod_*`
- Spectral? → `spec_*`
- Model-based? → `ar_*`, `ark_*`, etc.

**Use consistent suffixes:**
- Always include window/scope (`_w25`, `_full`)
- Use `_delta` for differences, `_ratio` for ratios
- Use `_z` for standardized versions

**Maintain semantic clarity:**
- `delta` = post - pre (signed)
- `absdelta` = |post - pre| (absolute)
- `ratio` = post / pre (can be < 1)
- `ndiff` = (post - pre) / (|post| + |pre|) (bounded -1 to 1)

### 2. Standardization Conventions

**When to use `_z` suffix:**
- Feature divided by robust scale (MAD * 1.4826)
- Makes features scale-invariant
- Example: `bl_wasserstein_z_w50` = `bl_wasserstein_w50` / robust_scale(pooled)

**When to use `_delta` vs `_shift`:**
- Use `_delta` for subtraction: post - pre
- Use `_shift` for changes in derived quantities (peaks, centroids)

### 3. Window Ordering Invariance

**Important**: All DoD features should be window-order invariant. Features computing `delta_wmin_wmax`, `ratio_wmin_wmax`, and `curv_wA_wB_wC` must:
1. Sort windows by size (not input order)
2. Use smallest window as "min", largest as "max"
3. Use first 3 smallest windows for curvature

Implemented via:
```python
idx = np.argsort(logw)
logw = logw[idx]
d = d[idx]
```

---

## Examples by Use Case

### Detecting Heavy-Tail Regime Changes
```
bl_tail_wasserstein_q95_upper_w25      # Is upper tail more different?
bl_tail_p_hi_delta_q95_w25             # More extreme values?
bl_ts_hill_alpha_delta_q95_w25         # Tail heavier?
bl_dod_outlier_sens_wassz_w25          # Outlier impact
```

### Detecting Localized Changes
```
bl_loc_ratio_wassz_w50                 # Boundary stronger than full?
bl_ts_loc_ratio_asym_delta_w50         # Asymmetry localized?
bl_dod_wasserstein_z_slope_logw        # Trend across scales
```

### Multi-Scale Analysis
```
bl_dod_wasserstein_z_delta_wmin_wmax   # How much distance grows with scale
bl_dod_energy_z_cv                     # Consistency across scales
bl_ts_dod_hill_alpha_delta_q95_range   # Variability in tail heaviness
```

### Tail Shape Characterization
```
bl_ts_hill_alpha_delta_q95_full        # Overall tail heaviness change
bl_ts_qspace_hi_delta_w100             # Upper tail curvature change
bl_ts_asym_delta_full                  # Asymmetry shift
bl_ts_p_hi_delta_t25_w50               # Exceedance at 2.5σ
```

---

## Related Documentation

- **TAIL_FEATURES_SUMMARY.md**: Detailed tail feature implementation
- **TAIL_SHAPE_IMPLEMENTATION.md**: Tail-shape feature specification
- **DoD_Features_Summary.md**: Distance-of-distance methodology
- **INTEGRATION_SUMMARY.md**: Feature integration into pipeline

---

## Changelog

### 2025-12-30
- Initial comprehensive documentation
- Consolidated naming conventions from all modules
- Added quick reference table and use case examples
