# Project Setup Complete! 🎉

Your structural break detection project is ready. Here's what was created:

## 📁 Project Structure

```
adia_structural_break/
├── README.md              ← Project overview
├── QUICKSTART.md          ← Detailed usage guide
├── pyproject.toml         ← Package configuration
├── .gitignore             ← Git ignore rules
├── setup_check.py         ← Verify installation
├── solution.py            ← ⭐ SUBMISSION FILE (train/infer functions)
│
├── src/sb/                ← Core library
│   ├── __init__.py
│   ├── config.py          ← All constants & seeds
│   ├── io.py              ← Data loading (split_series, iter_series)
│   ├── preprocessing.py   ← Robust scaling functions
│   ├── cv.py              ← Cross-validation & evaluation
│   ├── pipeline.py        ← End-to-end orchestration
│   │
│   ├── features/          ← Feature extraction modules
│   │   ├── __init__.py
│   │   ├── base.py        ← Feature orchestration & rank aggregation
│   │   ├── dist.py        ← Day 1: quantile_features, entropy
│   │   └── dynamics.py    ← Day 2: rolling_var_slope, kalman_variance
│   │
│   └── models/            ← Model wrappers
│       ├── __init__.py
│       └── gbm.py         ← LightGBM wrapper (optional, Day 3+)
│
├── scripts/               ← Development utilities
│   ├── train_local.py     ← Local training & evaluation
│   ├── infer_local.py     ← Generate predictions locally
│   ├── sanity_check.py    ← Verify determinism, speed, correctness
│   └── make_features.py   ← Extract & cache features
│
└── tests/                 ← Unit tests
    ├── test_features.py   ← Test individual feature functions
    └── test_determinism.py ← Test full pipeline determinism
```

## 🚀 Quick Start (3 commands)

```bash
# 1. Install
pip install -e .

# 2. Verify setup
python setup_check.py

# 3. Run sanity checks
python scripts\sanity_check.py
```

## 📊 Implemented Features (Day 1-2 Baseline)

### Day 1: Distribution Shape
✅ **delta_q_slope** - Distribution deformation measure
✅ **median_shift** - Robust location change  
✅ **iqr_ratio** - Scale change via IQR
✅ **delta_entropy** - Information content change

### Day 2: Transition Dynamics  
✅ **vol_slope_post** - Post-break volatility evolution
✅ **delta_kalman_var** - Process noise shift proxy

### Aggregation Strategy
✅ Robust scaling (median/MAD)
✅ Rank normalization per feature
✅ Simple average for final score
✅ **No ML** (deterministic by design)

## 🎯 Key Design Principles

1. **Originality-safe**: Non-standard signals to avoid correlation with known tests
2. **Deterministic**: Fixed seeds, no randomness (required by platform)
3. **Fast**: Target <1s per 1000 series
4. **Competitive**: Target 0.80+ ROC AUC for baseline

## 📝 Next Steps

### Step 1: Get Data
Download training data from CrunchDAO and place it in your project folder.

Expected format:
```csv
id,period,time,value,label
series_001,0,0,1.234,0
series_001,0,1,1.567,0
series_001,1,0,1.890,0
```

### Step 2: Local Training
```bash
python scripts\train_local.py --data train.csv
```

This will:
- ✅ Compute all 6 features
- ✅ Rank-normalize and aggregate
- ✅ Report ROC AUC score
- ✅ Show score distributions

### Step 3: Local Inference
```bash
python scripts\infer_local.py --data test.csv --output predictions.csv
```

### Step 4: Run All Sanity Checks
```bash
python scripts\sanity_check.py
```

Verifies:
- ✅ Determinism (same input → same output)
- ✅ Speed (extrapolates to full dataset)
- ✅ Monotonicity (scores correlate with breaks)
- ✅ Score range (all in [0, 1])
- ✅ No missing values

### Step 5: Submit to CrunchDAO
Once sanity checks pass:
1. Upload `solution.py` to the platform
2. Platform will call `train(X_train, y_train)` and `infer(X_test)`
3. Check your score on the leaderboard!

## 🔧 Optional Enhancements (Day 3+)

After baseline is working:

### 1. Multi-scale Features
Compute same features on different window sizes around the break point.

### 2. Add Meta-Model
Uncomment LightGBM code in `solution.py`:
```python
# In train():
global _model
features_df = features.base.compute_features(X_train)
_model = models.gbm.train_gbm(features_df, y_train)

# In infer():
if _model is not None:
    features_df = features.base.compute_features(X_test)
    scores = pd.Series(_model.predict(features_df), index=features_df.index)
```

### 3. More Features
Add to `src/sb/features/`:
- Frequency domain features (FFT)
- Autocorrelation changes
- Permutation entropy
- Hurst exponent
- Wavelet coefficients

## 📚 Key Files to Know

### For Development
- [src/sb/features/base.py](src/sb/features/base.py) - Add new features here
- [src/sb/config.py](src/sb/config.py) - Tune hyperparameters here
- [scripts/train_local.py](scripts/train_local.py) - Main dev script

### For Submission
- [solution.py](solution.py) - The only file uploaded to CrunchDAO
- Must contain `train()` and `infer()` functions
- Currently uses baseline (no ML)

### For Testing
- [scripts/sanity_check.py](scripts/sanity_check.py) - Before every submission
- [tests/test_determinism.py](tests/test_determinism.py) - Unit tests

## 💡 Tips

1. **Start simple**: Get the baseline working first (0.80+ AUC)
2. **Check determinism**: Run `sanity_check.py` often
3. **Profile speed**: 16k series must finish in reasonable time
4. **Monitor originality**: Avoid correlation with standard tests (KS, t-test, etc.)
5. **Use rank aggregation**: Safer than raw feature values for submission

## 🆘 Troubleshooting

**Import errors?**
```bash
pip install -e .
```

**Slow performance?**
- Profile with `python -m cProfile scripts/infer_local.py`
- Optimize bottleneck features first

**Low AUC?**
- Verify data format
- Check feature distributions
- Visualize score histograms by label
- Try multi-scale variants

**Determinism fails?**
- Check all random seeds are set
- Avoid operations with undefined ordering
- Test with `tests/test_determinism.py`

## 📖 Resources

- **CrunchDAO Hub**: https://hub.crunchdao.com
- **Documentation**: https://docs.crunchdao.com  
- **Baseline Notebook**: Check CrunchDAO docs for official starter
- **This Project**: Read [QUICKSTART.md](QUICKSTART.md) for detailed usage

---

**You're all set!** 🚀

Start with `python setup_check.py` and follow the steps above.

Good luck with the competition! 🏆
