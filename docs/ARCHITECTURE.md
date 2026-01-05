# Architecture Overview

## System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                               │
│  CSV: [id, period, value] where period ∈ {0, 1}                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    io.split_series()                             │
│  Segment each series: x0 (period=0), x1 (period=1)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              preprocessing.robust_scale()                        │
│  Apply separately to x0 and x1:                                 │
│  scaled = (x - median) / MAD                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION                             │
│  ┌─────────────────────────────────────────────────┐           │
│  │  Day 1: Distribution Shape (dist.py)            │           │
│  │  • quantile_features() → 3 features             │           │
│  │  • entropy_change() → 1 feature                 │           │
│  └─────────────────────────────────────────────────┘           │
│  ┌─────────────────────────────────────────────────┐           │
│  │  Day 2: Transition Dynamics (dynamics.py)       │           │
│  │  • rolling_var_slope() → 1 feature              │           │
│  │  • kalman_variance_change() → 1 feature         │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                  │
│  Result: 6 features per series                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            features.base.rank_normalize()                        │
│  Convert each feature to percentile ranks ∈ [0, 1]             │
│  (Originality-safe aggregation)                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           features.base.aggregate_features()                     │
│  Average ranks across all features                              │
│  final_score = mean(ranked_features)                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT PREDICTIONS                            │
│  Series: [id → score] where score ∈ [0, 1]                     │
│  Higher score = higher probability of structural break          │
└─────────────────────────────────────────────────────────────────┘
```

## Feature Details

### Day 1: Distribution Shape Features

```
Input: x0 (pre-break), x1 (post-break)
       ↓
   Compute quantiles at [0.1, 0.25, 0.5, 0.75, 0.9]
       ↓
   ┌────────────────────────────────────────┐
   │ delta_q_slope                          │
   │ = |slope(q1) - slope(q0)|              │
   │ where slope = q[90%] - q[10%]          │
   │                                        │
   │ Captures: Distribution deformation     │
   └────────────────────────────────────────┘
   ┌────────────────────────────────────────┐
   │ median_shift                           │
   │ = |median(x1) - median(x0)|            │
   │                                        │
   │ Captures: Robust location change       │
   └────────────────────────────────────────┘
   ┌────────────────────────────────────────┐
   │ iqr_ratio                              │
   │ = IQR(x1) / IQR(x0)                    │
   │ where IQR = q[75%] - q[25%]            │
   │                                        │
   │ Captures: Scale change                 │
   └────────────────────────────────────────┘
   ┌────────────────────────────────────────┐
   │ delta_entropy                          │
   │ = |entropy(x1) - entropy(x0)|          │
   │ where entropy = -Σ p·log(p)            │
   │                                        │
   │ Captures: Information change           │
   └────────────────────────────────────────┘
```

### Day 2: Transition Dynamics Features

```
Input: x1 (post-break segment only)
       ↓
   Compute rolling variance with window=50
       ↓
   ┌────────────────────────────────────────┐
   │ vol_slope_post                         │
   │ = |slope of rolling_var(x1)|           │
   │                                        │
   │ Captures: Delayed regime effects       │
   │ (How volatility evolves after break)   │
   └────────────────────────────────────────┘

Input: x0 and x1
       ↓
   Compute variance of first differences
       ↓
   ┌────────────────────────────────────────┐
   │ delta_kalman_var                       │
   │ = |var(diff(x1)) - var(diff(x0))|      │
   │                                        │
   │ Captures: Process noise shift          │
   │ (Lightweight Kalman approximation)     │
   └────────────────────────────────────────┘
```

## Code Organization

### Core Modules

```
sb/
├── config.py          → All constants (seeds, windows, params)
├── io.py              → Data I/O (split_series, load_data)
├── preprocessing.py   → Scaling (robust_scale, clip_outliers)
├── cv.py              → Evaluation (ROC AUC, cross-validation)
└── pipeline.py        → Orchestration (baseline & GBM pipelines)
```

### Feature Modules

```
sb/features/
├── base.py     → compute_features()      [main entry point]
│               → rank_normalize()        [originality-safe]
│               → aggregate_features()    [final scoring]
│
├── dist.py     → quantile_features()     [Day 1]
│               → entropy()
│               → entropy_change()
│
└── dynamics.py → rolling_var_slope()     [Day 2]
                → kalman_level_variance()
                → kalman_variance_change()
```

### Model Modules (Optional, Day 3+)

```
sb/models/
└── gbm.py      → StructuralBreakGBM      [LightGBM wrapper]
                → train_gbm()
                → predict_gbm()
```

## Execution Paths

### Path 1: Baseline (No ML)

```
solution.infer(X_test)
    ↓
features.base.compute_and_aggregate(X_test)
    ↓
features.base.compute_features(X_test)
    ↓
[For each series:]
    io.split_series(df_id)
    ↓
    preprocessing.robust_scale(x0), robust_scale(x1)
    ↓
    dist.quantile_features(x0, x1)
    dist.entropy_change(x0, x1)
    dynamics.volatility_features(x0, x1)
    ↓
[Aggregate all features]
    ↓
features.base.aggregate_features(feature_df)
    ↓
Return: pd.Series[id → score]
```

### Path 2: With Meta-Model (Optional)

```
solution.train(X_train, y_train)
    ↓
features.base.compute_features(X_train)
    ↓
models.gbm.train_gbm(features, y_train)
    ↓
[Store model in global _model]

solution.infer(X_test)
    ↓
features.base.compute_features(X_test)
    ↓
_model.predict(features)
    ↓
Return: pd.Series[id → score]
```

## Key Design Decisions

### 1. Why Rank Normalization?
```
Problem: Raw feature values may correlate with known tests
Solution: Rank-normalize → only ordering matters
Benefit: Originality safety + ROC AUC compatibility
```

### 2. Why Separate Scaling?
```
Problem: Global scaling mixes pre/post information
Solution: Scale x0 and x1 independently
Benefit: Avoids test-train leakage patterns
```

### 3. Why These Features?
```
Standard tests:        Our features:
- Mean difference   →  Quantile slope change
- Variance ratio    →  Volatility evolution
- KS statistic      →  Entropy change
- t-test            →  Median shift + IQR ratio

Benefit: Non-standard signals = better originality score
```

### 4. Why No ML (Initially)?
```
Advantages of baseline:
✓ Deterministic (required)
✓ Fast (<1s per 1000 series)
✓ Transparent (interpretable)
✓ Originality-safe (rank aggregation)

Add ML later only if:
✓ Baseline is solid (0.80+ AUC)
✓ Need to capture feature interactions
✓ Can maintain determinism
```

## Performance Considerations

### Bottlenecks (by operation)
```
1. Rolling variance computation  [~40% time]
   → Vectorized with pandas.rolling()
   
2. Quantile computation         [~30% time]
   → np.quantile() optimized
   
3. Entropy histogram            [~20% time]
   → Fixed bin count
   
4. Feature iteration            [~10% time]
   → Minimal overhead
```

### Optimization Strategies
```
✓ Vectorize where possible (pandas/numpy)
✓ Avoid Python loops over series
✓ Use fixed bin counts (no optimization)
✓ Cache nothing (determinism first)
✓ Profile before optimizing
```

## Testing Strategy

### 1. Unit Tests (tests/)
- Individual feature functions
- Scaling/preprocessing
- Determinism checks

### 2. Sanity Checks (scripts/sanity_check.py)
- End-to-end determinism
- Speed extrapolation
- Score range validation
- Monotonicity check

### 3. Integration Tests (scripts/train_local.py)
- Full pipeline on real data
- ROC AUC evaluation
- Feature distribution analysis

---

This architecture ensures:
✅ Modularity (easy to add features)
✅ Testability (each component isolated)
✅ Maintainability (clear separation of concerns)
✅ Performance (vectorized operations)
✅ Determinism (no hidden randomness)
