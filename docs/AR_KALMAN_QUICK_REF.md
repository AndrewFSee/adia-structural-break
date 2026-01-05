# AR/Kalman Features - Quick Reference

## 🚀 Quick Start

```bash
# Run full diagnostic pipeline
python scripts/run_ar_kalman_diagnostics.py

# Train AR/Kalman model
python scripts/train_local.py --mode arkf

# Generate predictions
python scripts/infer_local.py --mode arkf
```

## 📊 Key Files

| File | Purpose | Output |
|------|---------|--------|
| `src/sb/features/ar_kalman.py` | Feature extraction | ~68 AR/Kalman features |
| `scripts/run_ar_kalman_diagnostics.py` | Evaluate features | 3 CSV files + parquet |
| `scripts/train_local.py --mode arkf` | Train model | Saved model file |
| `scripts/infer_local.py --mode arkf` | Inference | predictions.csv |

## 🔑 Core Features

### Most Important (⭐ = Break Signal)
- ⭐ `ar1_rmse_cross_pred` - PRE model applied to POST
- ⭐ `kf_rmse_cross_pred` - PRE Kalman on POST  
- ⭐ `trend_rmse_cross` - PRE trend on POST
- `ar1_delta_phi` - Change in AR coefficient
- `kf_log_innov_var_ratio` - Innovation variance ratio
- `trend_delta_slope` - Change in trend slope

### Window Variants
All features available at boundaries: `_w25`, `_w50`, `_w100`

## 🛡️ Anti-Leakage Guarantees

✅ **Preprocessing**: PRE quantiles/mean/std only  
✅ **Model Fitting**: PRE segment only  
✅ **Cross-Prediction**: PRE → POST (no reverse!)  
✅ **CV Transforms**: Train fold only  
✅ **Windows**: Last W of PRE, first W of POST  

## 📈 Expected Results

```
Baseline AUC (rank aggregation): 0.65-0.75
Best single feature AUC: 0.60-0.65
Top 10 features contain signal: >0.58 AUC each
```

If baseline AUC < 0.60 → Features need work  
If baseline AUC > 0.75 → Ready for submission!

## ⚡ Performance

- **Speed**: ~200 series/sec (4 cores)
- **Features**: 68 total (~17 AR + ~18 Kalman + ~18 Trend + windows)
- **NaN rate**: <10% typical

## 🔧 Configuration

```python
# In scripts/run_ar_kalman_diagnostics.py
window_sizes = [25, 50, 100]  # Boundary windows
n_jobs = 4  # Parallel cores
n_splits = 5  # CV folds

# In ar_kalman.py
winsorize_quantiles = (0.01, 0.99)  # Outlier clipping
process_var_grid = np.logspace(-3, 0, 5)  # Kalman params
```

## 🐛 Common Issues

| Issue | Fix |
|-------|-----|
| High NaN% | Reduce window_sizes |
| Slow extraction | Use n_jobs=-1 or reduce grid |
| Low AUC | Combine with other features |
| Non-deterministic | Use n_jobs=1 for perfect reproducibility |

## 📚 Full Documentation

See [docs/AR_KALMAN_IMPLEMENTATION.md](AR_KALMAN_IMPLEMENTATION.md) for:
- Complete feature list with descriptions
- Detailed leakage prevention strategy
- Performance optimization tips
- Troubleshooting guide
