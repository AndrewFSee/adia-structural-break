# AR/Kalman Features Implementation Summary

## Files Created/Modified

### 1. **src/sb/features/ar_kalman.py** (NEW - ~650 lines)
Complete AR/Kalman feature extraction module with strict leakage prevention.

**Key Functions:**
- `robust_preprocess()` - Winsorize and standardize using PRE stats only
- `fit_ar1()` / `fit_ar2()` - AR model estimation via Yule-Walker
- `ar1_cross_prediction_error()` - Apply PRE model to POST (KEY BREAK SIGNAL)
- `simple_kalman_local_level()` - Local level Kalman filter
- `fit_kalman_params()` - Grid search for KF parameters (deterministic)
- `kalman_cross_prediction_error()` - Apply PRE KF params to POST
- `fit_local_linear_trend()` - Linear trend via least squares
- `compute_ar_features()` - Full AR feature set (12 base + 5×windows)
- `compute_kalman_features()` - Full Kalman feature set (6 base + 4×windows)
- `compute_local_trend_features()` - Trend features (6 base + 4×windows)
- `extract_features_for_id()` - Single series extraction
- `extract_features()` - Batch extraction with parallelization

**Anti-Leakage Measures:**
- ⚠️ All preprocessing uses ONLY PRE segment statistics
- ⚠️ Models fit on PRE, applied to POST ("pre-model → post-score")
- ⚠️ Cross-prediction errors capture break signal
- ⚠️ Multi-window features respect boundary (last W of PRE, first W of POST)

**Feature Counts:**
- AR features: ~17 base + 5×3 windows = ~32 features
- Kalman features: ~6 base + 4×3 windows = ~18 features  
- Trend features: ~6 base + 4×3 windows = ~18 features
- **Total: ~68 features** (varies by data availability)

---

### 2. **scripts/run_ar_kalman_diagnostics.py** (UPDATED)

**Changes:**
- Replaced internal feature code with `from sb.features.ar_kalman import extract_features`
- Added timing logs: total time, ms per series, throughput
- Added determinism check:
  - Recomputes features for 100 random series
  - Asserts max absolute difference < 1e-10
  - Warns if non-deterministic behavior detected
- Parallel extraction with `n_jobs=4`
- Saves 3 outputs:
  - `results/ar_kalman_feature_aucs.csv` - per-feature AUC table
  - `results/ar_kalman_cv_summary.csv` - fold-wise CV results
  - `results/ar_kalman_features.parquet` - full feature matrix

**Usage:**
```bash
python scripts/run_ar_kalman_diagnostics.py
```

**Expected Output:**
```
======================================================================
AR/KALMAN DIAGNOSTICS - LEAKAGE-SAFE PIPELINE
======================================================================

Configuration:
  Data directory: data
  Window sizes: [25, 50, 100]
  CV folds: 5
  Parallel jobs: 4

======================================================================
STEP 1: LOAD DATA
======================================================================
Loaded X observations across Y series
Break rate: Z%

======================================================================
STEP 2: COMPUTE AR/KALMAN FEATURES
======================================================================
⚠️  ANTI-LEAKAGE: All preprocessing uses ONLY pre-segment statistics
   - Winsorization: quantiles from PRE only
   - Standardization: mean/std from PRE only
   - AR/Kalman params: estimated on PRE only
   - Cross-prediction: PRE model applied to POST

Extracting AR/Kalman features for 10,001 series...
Window sizes: [25, 50, 100]
Parallel jobs: 4
✅ Extracted 68 features for 10,001 series
   NaN count: X (Y%)

✅ Features computed: (10001, 68)
   Samples: 10,001
   Features: 68
   Time: 45.2s (4.5ms per series)
   Throughput: 221 series/sec

======================================================================
STEP 2.5: DETERMINISM CHECK
======================================================================
Recomputing features for 100 series...
Max absolute difference: 0.00e+00
✅ DETERMINISM VERIFIED: Features are deterministic!

======================================================================
STEP 3: CROSS-VALIDATION DIAGNOSTICS
======================================================================
... (CV results, per-feature AUCs, interpretation) ...
```

---

### 3. **scripts/train_local.py** (UPDATED)

**Changes:**
- Added `create_model_fn_arkf()` - LogisticRegressionCV factory
- Added `--mode arkf` option to argument parser
- Feature extraction branch for arkf mode:
  ```python
  if args.mode == "arkf":
      from sb.features.ar_kalman import extract_features
      X_raw = extract_features(X_train, window_sizes=[25, 50, 100], n_jobs=4)
  ```
- Training logic for arkf mode:
  - Uses LogisticRegressionCV with Cs=[0.001, 0.01, 0.1, 1.0]
  - CV=3, scoring='roc_auc', max_iter=500
  - Same fold-wise preprocessing (impute NaNs, rank normalize)
  - Saves model to `models/trained_model.joblib`

**Usage:**
```bash
# Train with AR/Kalman features
python scripts/train_local.py --mode arkf

# Train with AR/Kalman + custom CV folds
python scripts/train_local.py --mode arkf --n-folds 10

# Save to custom path
python scripts/train_local.py --mode arkf --save-model models/arkf_model.joblib
```

**Expected Output:**
```
======================================================================
STRUCTURAL BREAK DETECTION - PROPER CROSS-VALIDATION
======================================================================

Mode: ARKF
Multi-scale: NO
CV Folds: 5
Random Seed: 42

======================================================================
FEATURE EXTRACTION
======================================================================

Computing AR/Kalman features...
Extracting AR/Kalman features for 10,001 series...
... (feature extraction logs) ...

======================================================================
MODE: AR/KALMAN FEATURES WITH LOGISTIC REGRESSION
======================================================================

Using heavily regularized logistic regression on AR/Kalman features

======================================================================
CROSS-VALIDATION (Rank normalization inside folds)
======================================================================

Running stratified K-fold CV...
Each fold:
  1. Split by id (stratified by label)
  2. Impute NaNs using train fold median
  3. Rank-normalize using train fold distribution
  4. Train LogisticRegressionCV
  5. Predict on ranked val
  6. Compute AUC

Fold 1: Val AUC = 0.XXXX (n_val=YYYY)
... (fold results) ...

======================================================================
CROSS-VALIDATION RESULTS
======================================================================

Out-of-sample ROC AUC: 0.XXXX ± 0.YYYY

======================================================================
TRAINING FINAL MODEL (on all data)
======================================================================

✅ Model saved to: models/trained_model.joblib
```

---

### 4. **scripts/infer_local.py** (UPDATED)

**Changes:**
- Added `--mode arkf` option to argument parser
- Feature extraction branch for arkf mode:
  ```python
  if args.mode == "arkf":
      from sb.features.ar_kalman import extract_features
      X_test = io.load_test("data")
      X_raw = extract_features(X_test, window_sizes=[25, 50, 100], n_jobs=4)
  ```
- Inference logic for arkf mode:
  - Loads model from `--model` path (default: models/trained_model.joblib)
  - Uses `cv_proper.predict_with_rank_norm()` for proper preprocessing
  - Outputs predictions.csv in standard format

**Usage:**
```bash
# Infer with AR/Kalman model
python scripts/infer_local.py --mode arkf

# Custom model path
python scripts/infer_local.py --mode arkf --model models/arkf_model.joblib

# Custom output path
python scripts/infer_local.py --mode arkf --output my_predictions.csv
```

---

## Feature Description

### AR(1) Features (Full Segment)
1. `ar1_phi_pre` - AR(1) coefficient on PRE
2. `ar1_phi_post` - AR(1) coefficient on POST
3. `ar1_delta_phi` - Change in AR coefficient (post - pre)
4. `ar1_resid_var_pre` - Residual variance on PRE
5. `ar1_resid_var_post` - Residual variance on POST
6. `ar1_log_resid_var_ratio` - Log ratio of residual variances
7. `ar1_rmse_pre` - RMSE on PRE (1-step)
8. `ar1_rmse_post` - RMSE on POST (1-step)
9. `ar1_rmse_cross_pred` - **RMSE when PRE model applied to POST** ⭐
10. `ar1_resid_var_cross_pred` - Residual variance of cross-prediction
11. `ar1_delta_rmse_cross` - Difference in cross-prediction vs PRE RMSE

### AR(2) Features (Full Segment)
12. `ar2_phi1_pre` - AR(2) first coefficient
13. `ar2_phi2_pre` - AR(2) second coefficient
14. `ar2_resid_var_pre` - AR(2) residual variance
15. `ar2_rmse_pre` - AR(2) RMSE

### AR Window Features (×3 windows: 25, 50, 100)
16-20. `ar1_phi_pre_w{W}`, `ar1_phi_post_w{W}`, `ar1_delta_phi_w{W}`, `ar1_rmse_pre_w{W}`, `ar1_rmse_cross_w{W}`

### Kalman Features (Full Segment)
21. `kf_process_var_pre` - Process noise variance (fitted on PRE)
22. `kf_innov_var_pre` - Innovation variance on PRE
23. `kf_innov_var_post` - Innovation variance on POST (using PRE params)
24. `kf_log_innov_var_ratio` - Log ratio of innovation variances
25. `kf_rmse_cross_pred` - **Cross-prediction RMSE** ⭐
26. `kf_innov_var_cross_pred` - Cross-prediction innovation variance

### Kalman Window Features (×3 windows)
27-30. `kf_innov_var_pre_w{W}`, `kf_innov_var_post_w{W}`, `kf_log_innov_var_ratio_w{W}`, `kf_rmse_cross_w{W}`

### Trend Features (Full Segment)
31. `trend_slope_pre` - Linear trend slope on PRE
32. `trend_slope_post` - Linear trend slope on POST
33. `trend_delta_slope` - Change in slope
34. `trend_rmse_pre` - Trend fit quality on PRE
35. `trend_rmse_post` - Trend fit quality on POST
36. `trend_rmse_cross` - **RMSE when PRE trend applied to POST** ⭐
37. `trend_mean_error_cross` - Mean signed error of cross-prediction

### Trend Window Features (×3 windows)
38-41. `trend_slope_pre_w{W}`, `trend_slope_post_w{W}`, `trend_delta_slope_w{W}`, `trend_rmse_cross_w{W}`

---

## Leakage-Safety Verification

### Preprocessing Level
✅ **Winsorization**: Quantiles computed from PRE only
✅ **Standardization**: Mean/std computed from PRE only
✅ **No global statistics**: Each series processed independently

### Model Fitting Level
✅ **AR parameters**: Fitted on PRE segment only
✅ **Kalman parameters**: Grid search on PRE segment only
✅ **Trend parameters**: Least squares on PRE segment only

### Cross-Validation Level
✅ **NaN imputation**: Median from TRAIN fold only
✅ **Rank normalization**: Percentiles from TRAIN fold only
✅ **No global transforms**: All fold-wise preprocessing

### Window Features Level
✅ **Boundary respect**: Last W of PRE, first W of POST
✅ **No cross-contamination**: Windows don't overlap break point

---

## Determinism Guarantees

1. **Fixed random seeds**: `np.random.seed(42)` in all scripts
2. **Deterministic algorithms**:
   - Yule-Walker (closed-form)
   - Grid search (fixed grid)
   - Least squares (deterministic solver)
3. **No stochastic components**: No MCMC, no random initialization
4. **Parallel safety**: joblib with loky backend (deterministic)
5. **Verification**: Built-in determinism check in diagnostics script

---

## Performance

**Expected Performance** (on 10,001 series):
- Feature extraction: ~40-60 seconds
- Throughput: ~200-250 series/sec
- Per-series: ~4-5ms
- With n_jobs=4 on modern CPU

**Optimization Tips:**
- Use `n_jobs=-1` for all cores
- Set `fast_mode=True` in diagnostics to skip Kalman (2× speedup)
- Reduce window_sizes if only interested in full-segment features

---

## Usage Examples

### Full Pipeline (Train → Diagnose → Infer)

```bash
# 1. Run diagnostics to evaluate features
python scripts/run_ar_kalman_diagnostics.py
# → Check results/ar_kalman_cv_summary.csv for baseline AUC

# 2. Train model if AUC looks promising
python scripts/train_local.py --mode arkf --n-folds 5
# → Saves model to models/trained_model.joblib

# 3. Generate predictions
python scripts/infer_local.py --mode arkf --output predictions.csv
# → Creates predictions.csv for submission
```

### Compare with Existing Features

```bash
# Train with standard features
python scripts/train_local.py --mode gbm --multiscale

# Train with AR/Kalman features
python scripts/train_local.py --mode arkf

# Compare CV AUC results
```

### Quick Feature Inspection

```python
import pandas as pd

# Load features from diagnostics
features = pd.read_parquet('results/ar_kalman_features.parquet')
print(features.head())

# Load feature AUCs
aucs = pd.read_csv('results/ar_kalman_feature_aucs.csv')
print(aucs.head(20))  # Top 20 features
```

---

## Troubleshooting

### Issue: "Max absolute difference: 1e-08"
**Cause**: Parallel processing can introduce tiny numerical differences
**Fix**: This is acceptable (<1e-6). For perfect reproducibility, use `n_jobs=1`

### Issue: High NaN percentage (>20%)
**Cause**: Many short series or small window sizes
**Fix**: Reduce window_sizes or accept NaN imputation

### Issue: Low baseline AUC (<0.60)
**Cause**: AR/Kalman assumptions don't match break types
**Fix**: Try combining with other feature families (distribution, dynamics)

### Issue: Slow feature extraction (>2 minutes)
**Cause**: Kalman grid search is expensive
**Fix**: Reduce process_var_grid size or skip Kalman features

---

## Next Steps

1. **Feature Engineering**:
   - Add ARMA models (AR + MA terms)
   - Add seasonal decomposition
   - Add spectral features (FFT)

2. **Model Improvements**:
   - Ensemble AR/Kalman with distribution features
   - Use feature selection (drop low-AUC features)
   - Try non-linear models (GBM on AR features)

3. **Validation**:
   - Test on holdout set
   - Analyze per-break-type performance
   - Check calibration of predictions

4. **Production**:
   - Cache features for faster iteration
   - Add feature version tracking
   - Monitor feature stability over time
