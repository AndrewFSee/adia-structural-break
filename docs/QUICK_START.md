# Quick Start Guide - Refactored Pipeline

## Training

### Basic Training (Standard Features)
```bash
python scripts/train_local.py --mode gbm
```

### Training with Multi-Scale Features (Recommended)
```bash
python scripts/train_local.py --mode gbm --multiscale
```

### Custom CV Folds
```bash
python scripts/train_local.py --mode gbm --multiscale --n-folds 10
```

### Expected Output
```
Out-of-sample CV AUC: 0.7842 ± 0.0234
Fold AUCs: ['0.7654', '0.7891', '0.8012', '0.7723', '0.7931']
✅ Model saved to: models/trained_model.joblib
```

## Inference

### Basic Inference
```bash
python scripts/infer_local.py --mode gbm
```

### Inference with Multi-Scale (Must Match Training)
```bash
python scripts/infer_local.py --mode gbm --multiscale
```

### Custom Output Path
```bash
python scripts/infer_local.py --mode gbm --output my_predictions.csv
```

## Baseline Comparison

### Train Baseline (No ML)
```bash
python scripts/train_local.py --mode baseline
```

### Infer with Baseline
```bash
python scripts/infer_local.py --mode baseline
```

## Key Commands Reference

| Task | Command | Notes |
|------|---------|-------|
| Train GBM | `python scripts/train_local.py --mode gbm` | Standard features |
| Train GBM + Multi-scale | `python scripts/train_local.py --mode gbm --multiscale` | Recommended |
| Infer GBM | `python scripts/infer_local.py --mode gbm` | Uses saved model |
| Infer GBM + Multi-scale | `python scripts/infer_local.py --mode gbm --multiscale` | Must match training |
| Baseline | `python scripts/train_local.py --mode baseline` | No ML, just rank aggregation |
| Sanity Check | `python scripts/sanity_check.py` | Verify determinism |

## Understanding the Output

### Training Output
```
Out-of-sample CV AUC: 0.7842 ± 0.0234
```
- **Mean AUC**: Expected performance on unseen data
- **Std AUC**: Stability across folds (lower is better)
- **Target**: CV AUC > 0.80, Std < 0.03

### Inference Output
```
Test set AUC: 0.7756
```
- Should be **close to CV AUC** (±2-3%)
- If test AUC << CV AUC: Still overfitting
- If test AUC >> CV AUC: Lucky test set

## Troubleshooting

### Low CV AUC (< 0.75)
- Try `--multiscale` for more features
- Check feature distributions (NaN values?)
- Verify data quality (enough samples per class?)

### High CV Std (> 0.05)
- Increase `--n-folds` for more stable estimates
- Check for class imbalance
- Consider more regularization in config.py

### Test AUC Much Lower than CV AUC
- Possible data leakage still present
- Check that inference uses same features as training
- Verify `--multiscale` flag matches between train/infer

### NaN Values in Features
- Normal for small windows in multi-scale
- Filled with median automatically
- If excessive (>10%), check data preprocessing

## Configuration

### LightGBM Parameters (config.py)
```python
LIGHTGBM_PARAMS = {
    "learning_rate": 0.03,      # Slow learning (reduce overfitting)
    "max_depth": 4,             # Shallow trees
    "num_leaves": 15,           # Simple trees
    "min_data_in_leaf": 200,    # Require more samples per leaf
    "lambda_l2": 1.0,           # L2 regularization
}
```

### Multi-Scale Windows (config.py)
```python
MULTI_SCALE_WINDOWS = [50, 100, 250]  # Boundary-focused windows
```

## Platform Submission

### Method 1: Use solution.py Directly
```python
from solution import train, infer

# Training
train(X_train, y_train, use_multiscale=True)

# Inference
predictions = infer(X_test)
```

### Method 2: Export Trained Model
```bash
# Train locally
python scripts/train_local.py --mode gbm --multiscale

# Model saved to: models/trained_model.joblib
# Upload this model to platform
```

## Performance Expectations

| Scenario | Expected CV AUC | Expected Test AUC |
|----------|-----------------|-------------------|
| Baseline (no ML) | 0.70-0.75 | 0.68-0.73 |
| GBM (standard) | 0.75-0.80 | 0.73-0.78 |
| GBM + Multi-scale | 0.78-0.82 | 0.76-0.80 |

**Goal**: Test AUC within **2-3%** of CV AUC (indicates good generalization)

## File Locations

| File | Purpose |
|------|---------|
| `models/trained_model.joblib` | Saved GBM model |
| `predictions.csv` | Inference output |
| `src/sb/config.py` | Hyperparameters |
| `src/sb/cv_proper.py` | Cross-validation logic |
| `src/sb/features/multiscale.py` | Multi-scale features |

## Best Practices

1. **Always use `--multiscale`** for better performance
2. **Match training and inference** (same `--multiscale` flag)
3. **Trust CV AUC** over in-sample AUC
4. **Check CV std** (should be < 0.03)
5. **Verify determinism** with multiple runs
6. **Compare with baseline** to validate improvement

## Common Mistakes

❌ **Don't**: Use different features for training and inference  
✅ **Do**: Use same `--multiscale` flag for both

❌ **Don't**: Trust in-sample AUC (optimistic)  
✅ **Do**: Trust CV AUC (realistic)

❌ **Don't**: Rank features before CV split  
✅ **Do**: Rank inside each fold (automatic in cv_proper.py)

❌ **Don't**: Tune parameters based on test set  
✅ **Do**: Tune parameters based on CV AUC only
