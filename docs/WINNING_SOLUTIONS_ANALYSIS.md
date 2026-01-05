# Winning Solutions Analysis - ADIA Lab Structural Break Challenge

## Key Findings

**Your current performance: 0.70 AUC**
**Last year's winning solutions: 0.88-0.90 AUC**

The gap is NOT due to model choice - it's due to **feature engineering**. All winners used similar models (LightGBM/XGBoost/CatBoost) but vastly different features.

---

## 🏆 The "Magic" Feature - Coefficient of Variation (CV)

**Multiple winning teams independently discovered this:**

```python
cv_global = std(entire_series) / mean(entire_series)
```

**Why it works:**
- Acts as a **regime detector** for the data generation process
- Provides unit-less standard deviation allowing cross-series comparison
- Creates natural thresholds (0.03, 0.04, 0.20, 0.27) that segment samples into distinct regimes
- One regime (CV ∈ [0.198, 0.20]) is "easy negatives" with ~1% positive rate
- Single feature AUC: ~53%, but enables 5-10% model AUC boost when combined

**Critical insight:** The data has different generation regimes that need to be identified first.

---

## 🎯 Top Winning Approaches Summary

### 1. 2nd Place Solution (aParsecFromFuture)
**AUC: ~88%** | **Features: 2,408 → 200-500 selected**

**Transformations:**
- Z-score normalization
- Cumulative sum (CUSUM)
- Dense ranking
- Absolute value
- Moving average
- Moving standard deviation

**Feature Categories:**
- **Statistical**: mean, median, std, min, max, skewness, autocorr, CV, quantiles
- **Hypothesis tests**: F-test, Levene's test, Kolmogorov-Smirnov test
- **TabPFN features**: Neural prior-fitted network predictions as meta-features

**Feature Selection:**
- SHAP values (top 200-500)
- LightGBM gain importance (top 500)
- Combined selected features from multiple methods

### 2. 6th Place Solution (KaggleJM)
**AUC: 89.47% CV, 88.38% Public LB** | **Features: ~1000 → 231 selected**

**Key Feature Categories:**

1. **Variance & Transform Features** ⭐ (Most important)
   - Multiple transforms: Raw, EWMA volatility, Rolling std, Standardized residuals, MOSUM variance
   - Metrics: CV, smoothness, pre/post log-ratios, KS/Wasserstein distances, ACF differences

2. **Compression Features** (Novel approach)
   - Z-lib compression ratios (pre vs post)
   - Lempel-Ziv complexity
   - Normalized compression distance
   - Captures "how predictable" each segment is

3. **CUSUM Transformation**
   - `cusum_elbow_category`: Discrete shape label (up-flat, down-flat, trend-up, etc.)
   - `cusum_elbow_sharpness`: Normalized curvature at boundary
   - `cusum_error_wasserstein`: Wasserstein distance of CUSUM residuals

4. **GARCH/Volatility Features**
   - GARCH parameters pre/post (beta, alpha)
   - Volatility persistence, decay rates
   - Standardized residual mismatch (KS test)
   - Volatility forecast divergence

5. **ROCKET Features** (Random Convolutional Kernels)
   - 1D convolutions with various kernels (Laplace, edge detectors, triangular)
   - PPV and std pooling
   - Captures microstructural patterns (edges, ramps, spikes)

6. **SSA (Singular Spectrum Analysis)**
   - Spectral moments: centroid, skew, kurtosis
   - Band energy proportions
   - Eigen-spectra differences pre/post
   - Top singular value ratios

7. **Boundary Transition Features**
   - Local statistics ±100 window around change point
   - Volatility explosion at transition
   - Shock magnitude and decay
   - State persistence before/after

8. **Trend Dynamics**
   - Rolling beta trends
   - Sub-segment analysis (split into 5 slices)
   - Reversion strength
   - Deviation from global trend

**Ensemble:**
- 10-fold bagged ensemble of LightGBM, XGBoost, CatBoost
- Optuna hyperparameter tuning per model

### 3. Top Solution (Chinese Team - BoNing-Gu)
**AUC: 90.29% LGB, 90.25% XGB, 88.62% Public LB** | **Features: 3,690 → 107 selected**

**Pipeline:**
1. **TS Transformations**: RAW, CUMSUM, DIFF
2. **Feature Extraction** (10 function families):
   - `distribution_stats_features`
   - `test_stats_features`
   - `trend_features`
   - `oscillation_features`
   - `cyclic_features`
   - `amplitude_features`
   - `entropy_features`
   - `tsfresh_features`
   - `ar_model_features`
   - `rupture_cost_features`

3. **Cross-period operations** for each feature:
   ```python
   diff = right - left
   ratio = right / (left + 1e-6)
   contribution_left = left / (left + right + 1e-6)
   contribution_right = right / (left + right + 1e-6)
   ratio_to_whole_left = left / (whole + 1e-6)
   ratio_to_whole_right = right / (whole + 1e-6)
   ```

4. **Feature Selection:**
   - Correlation filter (3690 → 2817 at 0.95 threshold)
   - Feature importance (2817 → 500)
   - Permutation importance (500 → 162)

5. **Feature Interactions** ⭐ **Major boost!**
   - Operations: mul, sqmul, add, sub, div
   - 162 features → 162 + 91,449 interactions
   - **AUC jumped from 0.81 → 0.88!**
   - After selection: 800 features → 107 final

**"Magic" Feature Created:**
```python
magic = mul_RAW_1_stats_cv_whole_RAW_1_stats_std_whole
      = (std ** 2) / mean
```

### 4. Other Top Solutions (secabird)

**Additional Feature Types:**

1. **Hurst Exponent** - Long-term memory / mean reversion
2. **Energy Statistic**: `2 * cross_dist - before_dist - after_dist`
3. **Distribution Tests**:
   - Anderson-Darling
   - Cramér-von Mises
   - Mood's median test
   - Fligner-Killeen variance test
4. **Sample Entropy**
5. **AR(5) model coefficients**
6. **Selected tsfresh features**

---

## 🔑 Critical Insights

### What You're Missing:

1. **Coefficient of Variation (CV)** - The regime detector
   - Global CV, pre CV, post CV, and their interactions
   - **This alone can add 5-10% AUC**

2. **Time Series Transformations**
   - You only use RAW series
   - Winners use: CUMSUM, DIFF, RANK, EWMA, MOSUM, residuals
   - **Each transformation captures different aspects**

3. **Feature Interactions**
   - You use 640 base features
   - Winners create 91,000+ interactions, then select best
   - **Interaction features gave 0.81 → 0.88 jump**

4. **Compression-Based Features**
   - Z-lib compression ratios
   - Lempel-Ziv complexity
   - Novel approach, captures predictability

5. **CUSUM Features**
   - Elbow detection at boundary
   - Shape categories
   - Wasserstein distance of residuals

6. **GARCH/Volatility Modeling**
   - Fitted GARCH parameters pre/post
   - Persistence and decay parameters
   - Your variance features were too simple

7. **ROCKET Convolutions**
   - Random 1D convolutions
   - Captures microstructural patterns
   - Alternative to deep learning

8. **SSA (Singular Spectrum Analysis)**
   - Spectral decomposition
   - Eigen-spectrum changes
   - Frequency domain analysis (more advanced than your FFT)

---

## 📊 Performance Progression Typical Path

```
Baseline (rank aggregation):           0.58
+ Basic features:                       0.70  ← YOU ARE HERE
+ CV feature:                           0.75-0.78
+ Transformations (CUMSUM, DIFF):       0.80
+ Advanced features (ROCKET, SSA, GARCH): 0.83
+ Feature interactions:                 0.88
+ Ensemble + tuning:                    0.90
```

---

## 🎯 Recommended Implementation Priority

### Phase 1: Quick Wins (Expected: 0.70 → 0.78)
1. **Add Coefficient of Variation features** ⭐⭐⭐
   - Global CV, pre CV, post CV
   - CV ratios and differences
   - **Highest ROI feature**

2. **Time series transformations**
   - CUMSUM
   - DIFF (first differences)
   - RANK (dense ranking)

3. **Compute all features on transformed series**
   - Your 640 features × 3 transforms = 1,920 features

### Phase 2: Advanced Features (Expected: 0.78 → 0.85)
4. **Compression features**
   - Z-lib compression ratio pre/post
   - Lempel-Ziv complexity

5. **CUSUM-specific features**
   - Elbow detection
   - Shape categories
   - Wasserstein on CUSUM residuals

6. **GARCH features**
   - Fit GARCH(1,1) pre and post
   - Extract alpha, beta parameters
   - Compare persistence

7. **Hypothesis tests** (you're missing)
   - F-test
   - Levene's test
   - Anderson-Darling
   - Cramér-von Mises

### Phase 3: Feature Interactions (Expected: 0.85 → 0.88)
8. **Create feature interactions**
   - Multiply, divide, add, subtract top features
   - Square and square-multiply
   - Select best 100-200 after interaction

9. **Aggressive feature selection**
   - Start with 5,000+ features
   - Use correlation filter
   - Use SHAP or permutation importance
   - Select final 100-200

### Phase 4: Ensemble (Expected: 0.88 → 0.90)
10. **Multi-model ensemble**
    - LightGBM + XGBoost + CatBoost
    - 10-fold bagging
    - Optuna tuning per model

---

## 💡 Why This Works

Your current features (Wasserstein, Energy distance, tail features) are good but:
1. **Only computed on raw series** (winners use 3-5 transformations)
2. **No CV-based regime detection** (critical for this data)
3. **No feature interactions** (where the magic happens)
4. **Missing compression/complexity features** (novel signal)
5. **Missing fitted model parameters** (GARCH, AR coefficients)

The data has **multiple generation regimes** that need different feature spaces to separate. CV identifies the regime, then other features discriminate within regimes.

---

## 🚀 Implementation Strategy

**Option A: Fast Path to 0.78 AUC** (2-3 hours)
- Add CV features
- Add CUMSUM/DIFF transformations
- Recompute existing features on transforms
- Should reach 0.78+ AUC

**Option B: Full Competitive Solution** (1-2 days)
- Implement all Phase 1-3 features
- Feature interactions
- Aggressive selection
- Target: 0.85-0.88 AUC

**Option C: Maximum Performance** (3-5 days)
- Full feature engineering
- Ensemble modeling
- Extensive hyperparameter tuning
- Target: 0.88-0.90 AUC

---

## Code to Start With

I'll create implementations for:
1. CV features (magic feature)
2. Time series transformations
3. Compression features
4. CUSUM features
5. Feature interaction pipeline
6. Enhanced feature selection

Would you like me to start with Phase 1 (quick wins to get to 0.78)?
