# GBM Mode Usage Guide

The GBM (Gradient Boosting Machine) mode uses LightGBM on top of the engineered features for better performance.

## Quick Start

### 1. Train with GBM
```bash
# Train on CrunchDAO data
python scripts/train_local.py --mode gbm

# Custom model save location
python scripts/train_local.py --mode gbm --save-model my_model.joblib
```

This will:
- Compute all 6 features
- Train LightGBM classifier
- Show feature importances
- Save model to `models/trained_model.joblib`

### 2. Infer with GBM
```bash
# Use the trained model
python scripts/infer_local.py --mode gbm

# Specify model path
python scripts/infer_local.py --mode gbm --model models/trained_model.joblib

# On CrunchDAO test set
python scripts/infer_local.py --mode gbm --crunchdao
```

## Baseline vs GBM Comparison

```bash
# Train both and compare
python scripts/train_local.py --mode baseline
python scripts/train_local.py --mode gbm

# Test both on test set
python scripts/infer_local.py --mode baseline --output baseline_preds.csv
python scripts/infer_local.py --mode gbm --output gbm_preds.csv
```

## When to Use Each Mode

### Use Baseline When:
- ✅ You want maximum determinism
- ✅ You need fast inference
- ✅ You want to understand the model
- ✅ You're checking originality safety
- ✅ Getting started (Day 1-2)

### Use GBM When:
- ✅ Baseline is working well (0.80+ AUC)
- ✅ You want to squeeze out extra performance
- ✅ You're fine with a black-box model
- ✅ You have good training data (10k series)
- ✅ Moving to Day 3+

## Typical Performance

With 10,000 training series:
- **Baseline**: 0.78-0.82 ROC AUC (fast, interpretable)
- **GBM**: 0.82-0.86 ROC AUC (slower, better accuracy)

## Feature Importance

After training with GBM, you'll see which features matter most:

```
Feature importance:
  delta_kalman_var     0.285
  vol_slope_post       0.241
  delta_entropy        0.189
  median_shift         0.147
  delta_q_slope        0.092
  iqr_ratio           0.046
```

This helps you understand:
1. Which signals are most predictive
2. Whether to engineer more variants
3. Which features to focus on for Day 3+

## Model Files

Models are saved to `models/` directory (in .gitignore):
```
models/
└── trained_model.joblib    (default location)
```

The model file contains:
- Trained LightGBM classifier
- Feature names
- Best iteration number

## Troubleshooting

**Q: "Model file not found" error**
- A: Train the model first with `python scripts/train_local.py --mode gbm`

**Q: Different results on same data?**
- A: GBM has deterministic seed set, should be consistent. Check if you're using the same model file.

**Q: GBM slower than baseline?**
- A: Yes, GBM is slower (needs to compute tree predictions). For 100 series: baseline ~1s, GBM ~3s.

**Q: Can I use GBM for submission?**
- A: The baseline is already competitive and safer. GBM is for local testing/improvement.

## Advanced: Cross-Validation

To properly evaluate GBM vs baseline:

```python
from sb import data_loader, features, models, cv
from sklearn.model_selection import cross_val_score

# Load data
df, y = data_loader.load_for_training("data")
X = features.base.compute_features(df)

# Baseline scores (rank aggregation)
baseline_scores = features.base.aggregate_features(X)
baseline_auc = cv.evaluate_predictions(y.values, baseline_scores.values)
print(f"Baseline CV AUC: {baseline_auc:.4f}")

# GBM scores (5-fold CV)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gbm_aucs = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = models.gbm.train_gbm(X_train, y_train)
    preds = model.predict(X_val)
    auc = roc_auc_score(y_val, preds)
    gbm_aucs.append(auc)

print(f"GBM CV AUC: {np.mean(gbm_aucs):.4f} ± {np.std(gbm_aucs):.4f}")
```

## Summary Commands

```bash
# Full workflow with GBM
python scripts/inspect_data.py              # Understand data
python scripts/train_local.py --mode gbm    # Train GBM
python scripts/infer_local.py --mode gbm    # Test GBM
```

For submission, you still use `solution.py` which defaults to baseline (safer for originality checks).
