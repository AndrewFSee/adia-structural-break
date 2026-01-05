# Quick Start Guide

## Installation

1. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Linux/Mac
   ```

2. **Install the package**:
   ```bash
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   python -c "import sb; print(sb.__version__)"
   ```

## Usage

### 1. Run Sanity Checks

Test the implementation with synthetic data:

```bash
python scripts/sanity_check.py
```

This checks:
- ✅ Determinism (same input → same output)
- ✅ Speed (fast enough for submission)
- ✅ Monotonicity (scores correlate with breaks)
- ✅ Score range (all predictions in [0, 1])
- ✅ No missing values

### 2. Local Training

Train and evaluate on your data:

```bash
# If labels are in the data file (as 'label' column)
python scripts/train_local.py --data train.csv

# If labels are in a separate file
python scripts/train_local.py --data train.csv --labels labels.csv

# Use LightGBM meta-model (Day 3+)
python scripts/train_local.py --data train.csv --mode gbm
```

### 3. Local Inference

Generate predictions:

```bash
# Basic inference
python scripts/infer_local.py --data test.csv --output predictions.csv

# With evaluation (if you have labels)
python scripts/infer_local.py --data test.csv --output predictions.csv --labels test_labels.csv
```

### 4. Extract Features

Compute and cache features for analysis:

```bash
python scripts/make_features.py --data train.csv --output features.csv
```

## Expected Data Format

Your CSV files should have these columns:

- `id`: Series identifier (string or int)
- `period`: 0 for pre-break, 1 for post-break
- `value`: The time series value
- `label` (training only): 0 for no break, 1 for break

Example:
```csv
id,period,time,value,label
series_001,0,0,1.234,0
series_001,0,1,1.567,0
series_001,1,0,1.890,0
...
```

## Project Structure

```
adia_structural_break/
├── src/sb/              # Core library
│   ├── config.py        # Configuration & constants
│   ├── io.py            # Data loading utilities
│   ├── preprocessing.py # Normalization helpers
│   ├── features/        # Feature extraction
│   │   ├── dist.py      # Distribution shape features
│   │   ├── dynamics.py  # Transition dynamics features
│   │   └── base.py      # Feature orchestration
│   ├── models/          # Model training/prediction
│   │   └── gbm.py       # LightGBM wrapper
│   ├── cv.py            # Cross-validation utilities
│   └── pipeline.py      # End-to-end pipeline
├── solution.py          # ⭐ SUBMISSION FILE (train/infer)
├── scripts/             # Development utilities
│   ├── train_local.py   # Local training script
│   ├── infer_local.py   # Local inference script
│   ├── sanity_check.py  # Sanity checks
│   └── make_features.py # Feature extraction
└── tests/               # Unit tests
```

## Features (Day 1-2 Baseline)

### Day 1: Distribution Shape
1. **delta_q_slope**: Distribution deformation (90th - 10th percentile change)
2. **median_shift**: Robust location change
3. **iqr_ratio**: Scale change via interquartile range
4. **delta_entropy**: Information content change

### Day 2: Transition Dynamics
5. **vol_slope_post**: Post-break volatility evolution
6. **delta_kalman_var**: Process noise shift proxy

All features use:
- **Robust scaling** (median/MAD) to avoid correlation with classic tests
- **Rank-based aggregation** for originality safety
- **Deterministic computation** (no randomness)

## Baseline Strategy

The Day 1-2 baseline uses **NO machine learning**:

1. Extract 6 orthogonal features per series
2. Apply robust scaling (median/MAD) separately to pre/post segments
3. Rank-normalize each feature across all series
4. Average the ranks → final score ∈ [0, 1]

This approach is:
- ✅ Fast (< 1s per 1000 series)
- ✅ Deterministic (required by platform)
- ✅ Originality-safe (non-standard signals)
- ✅ Competitive (target: 0.80+ ROC AUC)

## Next Steps (Day 3+)

Once the baseline works well:

1. **Multi-scale variants**: Compute same features on different window sizes
2. **Light meta-model**: Add logistic regression or GAM on top of features
3. **Tail sensitivity**: Add extreme value features for edge cases
4. **Stress-test originality**: Ensure low Spearman correlation with standard tests

## Submission

The [solution.py](solution.py) file contains the `train()` and `infer()` functions required by CrunchDAO:

```python
from solution import train, infer

# Platform calls these functions
train(X_train, y_train)  # Optional: can be empty for baseline
predictions = infer(X_test)  # Must return Series with predictions
```

## Troubleshooting

**Q: Getting import errors?**
- Make sure you installed with `pip install -e .`
- Check that you're in the virtual environment

**Q: Sanity checks fail?**
- Check determinism: Are you using fixed seeds?
- Check speed: Try with fewer series first

**Q: Low ROC AUC?**
- Verify your data format matches expected structure
- Check label distribution (balanced?)
- Try visualizing feature distributions

**Q: Ready to submit?**
1. Run `python scripts/sanity_check.py` - all tests pass?
2. Test on sample data - reasonable predictions?
3. Upload [solution.py](solution.py) to CrunchDAO platform

## Resources

- [CrunchDAO Structural Break Challenge](https://hub.crunchdao.com)
- [Official Baseline Notebook](https://colab.research.google.com/...)
- [Documentation](https://docs.crunchdao.com)
- [Submission Video Tutorial](https://youtube.com/...)
