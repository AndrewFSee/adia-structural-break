# Tail-Restricted Features Integration Summary

## Files Edited

### 1. `src/sb/features/boundary_dist.py`
**Changes:**
- Fixed duplicate test output in `__main__` self-check block
- Removed stale assertions that appeared after "ALL TESTS PASSED ✓"
- Self-check now prints each test result once and ends cleanly

**What stayed the same:**
- All 5 tests remain (fast, local, no heavy frameworks)
- All tail feature logic unchanged (compute_tail_features, helper functions)
- Feature computation in compute_boundary_dist_features unchanged

### 2. `scripts/diagnostic_baseline.py`
**Changes:**
- Updated `--boundary-dist` help text to mention "180 tail-restricted features"
- Added diagnostic prints after feature extraction when `--boundary-dist` is enabled:
  - Counts `bl_tail_*` features and shows NaN percentage
  - Counts base `bl_*` features and shows NaN percentage
- Updated "Next steps" to include `--boundary` and `--boundary-dist` flags in suggested train command

**Example output:**
```
Feature shape: (10000, 250)
  → 180 tail-restricted features (bl_tail_*), 12.3% NaN avg
  → 30 base distribution distance features (bl_*), 8.1% NaN avg
```

### 3. `scripts/train_local.py`
**Changes:**
- Updated `--boundary-dist` help text to mention "tail-restricted features"
- Added print statement for boundary distribution distances in config summary
- Fixed syntax error (missing newlines between print statements)

### 4. `scripts/infer_local.py`
**Changes:**
- Updated `--boundary-dist` help text to mention "tail features; must match training"

## Integration Points (Already Complete)

### ✅ Feature Pipeline (`src/sb/features/base.py`)
- `compute_single_series_features()` calls `boundary_dist.compute_boundary_dist_features()` when `use_boundary_dist=True`
- Returns all features (base + tail) automatically via `features.update(boundary_dist_feats)`
- No separate flag needed for tail features - they come "for free" with `--boundary-dist`

### ✅ CLI Flags (All Scripts)
- `--boundary-dist` flag present in diagnostic_baseline.py, train_local.py, infer_local.py
- Passed through as `use_boundary_dist=args.boundary_dist` to compute_features()
- No duplicate or conflicting flags

### ✅ Fold-Safety
- All features computed per-series only (no cross-series stats)
- NaN handling via fold-safe MedianImputer in CV loop (no global fillna)
- Break likelihood computed fold-safely via BreakLikelihoodScorer

### ✅ Feature Count
- Original boundary_dist: 30 features
- New tail features: 180 features
- **Total with --boundary-dist: 210 features**

## CLI Usage Examples

### Diagnostic Baseline with Tail Features
```bash
python scripts/diagnostic_baseline.py --boundary-dist
python scripts/diagnostic_baseline.py --boundary-dist --topk 50
python scripts/diagnostic_baseline.py --boundary --boundary-dist --topk 100
```

### Training with Tail Features
```bash
python scripts/train_local.py --mode gbm --boundary-dist
python scripts/train_local.py --mode gbm --boundary --boundary-dist
python scripts/train_local.py --mode gbm --multiscale --boundary-dist
```

### Inference with Tail Features
```bash
python scripts/infer_local.py --mode gbm --boundary-dist --model models/trained_model.joblib
```

## Feature Names

All tail features follow consistent naming:

**Tail-only distances (24 per scope):**
- `bl_tail_wasserstein_q{90|95}_{upper|lower|both}_w{25|50|100|250|full}`
- `bl_tail_energy_q{90|95}_{upper|lower|both}_w{25|50|100|250|full}`
- `bl_tail_wasserstein_z_q{90|95}_{upper|lower|both}_w{25|50|100|250|full}`
- `bl_tail_energy_z_q{90|95}_{upper|lower|both}_w{25|50|100|250|full}`

**Winsorized distances (4 per scope):**
- `bl_tail_wins_wasserstein_w{25|50|100|250|full}`
- `bl_tail_wins_energy_w{25|50|100|250|full}`
- `bl_tail_wins_wasserstein_z_w{25|50|100|250|full}`
- `bl_tail_wins_energy_z_w{25|50|100|250|full}`

**Tail mass diagnostics (8 per scope):**
- `bl_tail_p_hi_delta_q{90|95}_w{25|50|100|250|full}`
- `bl_tail_p_lo_delta_q{90|95}_w{25|50|100|250|full}`
- `bl_tail_mean_excess_hi_delta_q{90|95}_w{25|50|100|250|full}`
- `bl_tail_mean_excess_lo_delta_q{90|95}_w{25|50|100|250|full}`

## Correctness Guarantees

✅ No changes to existing feature names or meanings
✅ No global fillna/median (fold-safe imputation only)
✅ Graceful handling of missing/short series (returns NaN)
✅ All tail features computed per-series from x0/x1 only
✅ No cross-series aggregation or global statistics
✅ Compatible with existing CV infrastructure

## Testing

Run self-check to verify implementation:
```bash
python -m src.sb.features.boundary_dist
```

Expected output:
- 5 tests (normal, short segments, tiny segments, NaNs, heavy-tail regime)
- All tests pass
- No duplicate output
- Clean termination
