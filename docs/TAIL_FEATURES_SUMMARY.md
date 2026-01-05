# Tail-Restricted Distribution Distance Features

## Summary

Extended `boundary_dist.py` with 180 new tail-restricted features that capture extreme value behavior and heavy-tail regime changes near the structural break boundary.

## Feature Categories

### A) Tail-Only Distances (24 per scope = 120 total)
Compares only the tail subsets of distributions to detect extreme value shifts.

**Parameters:**
- `q_tail ∈ {0.9, 0.95}`: Tail quantile threshold
- `side ∈ {"upper", "lower", "both"}`: Which tail to compare

**Features:**
- `bl_tail_wasserstein_q{90|95}_{upper|lower|both}_w{w}`: Wasserstein distance on tail subset
- `bl_tail_energy_q{90|95}_{upper|lower|both}_w{w}`: Energy distance on tail subset
- `bl_tail_wasserstein_z_q{90|95}_{upper|lower|both}_w{w}`: Standardized Wasserstein (normalized by robust scale)
- `bl_tail_energy_z_q{90|95}_{upper|lower|both}_w{w}`: Standardized Energy

**Example:** `bl_tail_wasserstein_q95_upper_w25` compares the top 5% of x0_tail vs top 5% of x1_head within a 25-sample boundary window.

### B) Winsorized Distances (4 per scope = 20 total)
Robust distances with outliers clipped at 5th/95th percentiles.

**Features:**
- `bl_tail_wins_wasserstein_w{w}`: Wasserstein on winsorized data
- `bl_tail_wins_energy_w{w}`: Energy on winsorized data
- `bl_tail_wins_wasserstein_z_w{w}`: Standardized winsorized Wasserstein
- `bl_tail_wins_energy_z_w{w}`: Standardized winsorized Energy

**Purpose:** Less sensitive to extreme outliers than full-distribution distances.

### C) Tail Mass Diagnostics (8 per scope = 40 total)
Measures shifts in tail probability and mean excess.

**Features:**
- `bl_tail_p_hi_delta_q{90|95}_w{w}`: Change in upper tail exceedance probability
- `bl_tail_p_lo_delta_q{90|95}_w{w}`: Change in lower tail exceedance probability
- `bl_tail_mean_excess_hi_delta_q{90|95}_w{w}`: Change in mean value above upper threshold
- `bl_tail_mean_excess_lo_delta_q{90|95}_w{w}`: Change in mean value below lower threshold

**Example:** If `bl_tail_p_hi_delta_q95_w25 > 0`, x1 has more samples exceeding the pooled 95th percentile than x0.

## Total Feature Count

| Category | Features per scope | Scopes | Total |
|----------|-------------------|--------|-------|
| Tail-only distances | 24 | 5 | 120 |
| Winsorized distances | 4 | 5 | 20 |
| Tail mass diagnostics | 8 | 5 | 40 |
| **TOTAL** | **36** | **5** | **180** |

**Scopes:** w25, w50, w100, w250, full

**Combined with existing features:**
- Original boundary_dist features: 30
- New tail features: 180
- **Total boundary_dist features: 210**

## Implementation Details

### Helper Functions

1. **`quantile_bounds(arr, q_low, q_high)`**
   - Returns (lower_quantile, upper_quantile) or (nan, nan) if empty

2. **`select_tail(arr, q=0.9, side="upper"|"lower"|"both")`**
   - Extracts tail subset above/below/both quantile thresholds
   - Returns empty array if <2 samples remain

3. **`winsorize(arr, q_low=0.05, q_high=0.95)`**
   - Clips values to quantile bounds
   - Returns empty array if <2 samples

4. **`safe_distances(a, b)`**
   - Wrapper for Wasserstein and Energy distances with error handling
   - Returns (wass, energy) or (nan, nan)

5. **`compute_tail_features(x0_scope, x1_scope, scope_name)`**
   - Main function computing all 36 tail features for a given scope
   - Returns dict with NaN for all features if insufficient data

### Integration

Modified `compute_boundary_dist_features()`:
- Added tail feature computation for each window w ∈ {25, 50, 100, 250}
- Added tail feature computation for full segment
- Uses `compute_tail_features()` and merges results via `features.update()`

### Self-Check (Test 5)

Added heavy-tail regime change test:
- x0: Normal distribution
- x1: Normal + occasional large positive spikes in top 10%
- Assertions:
  - `bl_tail_wasserstein_q95_upper_w25 > 0`
  - `bl_tail_p_hi_delta_q95_w25 > 0` (more extreme values in x1)
  - Winsorized distance < standard distance (outlier robustness)

## Use Cases

1. **Volatility Spikes:** Detect sudden increases in extreme values (e.g., financial crises)
2. **Heavy-Tail Transitions:** Identify shifts from normal to fat-tailed distributions
3. **Outlier Regime Changes:** Capture changes in outlier frequency/magnitude
4. **Asymmetric Tail Behavior:** Separate upper vs lower tail dynamics

## NaN Handling

- All features return `np.nan` if:
  - x0 or x1 has < 2 clean samples
  - Tail subset has < 2 samples after selection
  - Distance computation fails (e.g., numerical errors)
- Standardized features return `np.nan` if scale ≈ 0

## Naming Convention

All features use consistent `bl_tail_*` prefix:
- `bl_tail_wasserstein_*`: Raw Wasserstein distance
- `bl_tail_energy_*`: Raw Energy distance
- `bl_tail_*_z_*`: Standardized versions (divided by robust scale)
- `bl_tail_p_*_delta_*`: Probability shift
- `bl_tail_mean_excess_*_delta_*`: Mean excess shift
- `bl_tail_wins_*`: Winsorized distance

## Fold-Safety

✅ All features computed per-series only (no cross-series aggregation)
✅ No global statistics used
✅ Safe for use in CV loops without leakage
