# Testing Checklist - Refactored Pipeline

## Pre-Testing Setup

- [ ] Python environment configured (Python 3.8+)
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] CrunchDAO data in `data/` directory
  - [ ] `data/X_train.parquet`
  - [ ] `data/y_train.parquet`
  - [ ] `data/X_test.parquet` (optional)

## Phase 1: Basic Functionality

### Test 1: Import Verification
```bash
python -c "from src.sb import cv_proper, features; print('✅ Imports OK')"
```
- [ ] No import errors
- [ ] cv_proper module loads
- [ ] multiscale module loads

### Test 2: Feature Extraction
```bash
python -c "
from src.sb import data_loader, features
df, y = data_loader.load_for_training('data')
X = features.base.compute_features(df, use_multiscale=False)
print(f'✅ Features: {X.shape}')
"
```
- [ ] Features extracted successfully
- [ ] Shape: (n_samples, 6) for standard
- [ ] No errors

### Test 3: Multi-Scale Features
```bash
python -c "
from src.sb import data_loader, features
df, y = data_loader.load_for_training('data')
X = features.base.compute_features(df, use_multiscale=True)
print(f'✅ Multi-scale features: {X.shape}')
"
```
- [ ] Features extracted successfully
- [ ] Shape: (n_samples, 24) for multi-scale
- [ ] NaN values handled (filled with median)

## Phase 2: Training Pipeline

### Test 4: Baseline Training
```bash
python scripts/train_local.py --mode baseline
```
Expected output:
```
Baseline ROC AUC: 0.70-0.75
```
- [ ] Training completes without errors
- [ ] AUC is reasonable (0.65-0.80)
- [ ] No crashes or warnings

### Test 5: GBM Training (Standard Features)
```bash
python scripts/train_local.py --mode gbm
```
Expected output:
```
Out-of-sample CV AUC: 0.75-0.80 ± 0.02-0.04
✅ Model saved to: models/trained_model.joblib
```
- [ ] Cross-validation runs (5 folds)
- [ ] CV AUC reported (not in-sample AUC!)
- [ ] Model saved successfully
- [ ] CV std < 0.05 (stable)

### Test 6: GBM Training (Multi-Scale Features)
```bash
python scripts/train_local.py --mode gbm --multiscale
```
Expected output:
```
Out-of-sample CV AUC: 0.78-0.82 ± 0.02-0.03
Feature shape: (n_samples, 24)
✅ Model saved to: models/trained_model.joblib
```
- [ ] 24 features extracted
- [ ] CV AUC > standard features (hopefully!)
- [ ] Model saved successfully
- [ ] CV std < 0.03 (very stable)

### Test 7: Custom CV Folds
```bash
python scripts/train_local.py --mode gbm --n-folds 10
```
- [ ] Runs 10 folds instead of 5
- [ ] Takes longer (expected)
- [ ] CV std should be lower (more folds)

## Phase 3: Inference Pipeline

### Test 8: Baseline Inference
```bash
python scripts/infer_local.py --mode baseline --output test_baseline.csv
```
- [ ] Inference completes
- [ ] Predictions saved to test_baseline.csv
- [ ] Score range: [0, 1]
- [ ] Mean: 0.2-0.8 (not extreme)

### Test 9: GBM Inference (Standard)
```bash
# First train
python scripts/train_local.py --mode gbm
# Then infer
python scripts/infer_local.py --mode gbm --output test_gbm.csv
```
- [ ] Model loads successfully
- [ ] Inference completes
- [ ] Predictions saved
- [ ] Score range: [0, 1]

### Test 10: GBM Inference (Multi-Scale)
```bash
# First train with multi-scale
python scripts/train_local.py --mode gbm --multiscale
# Then infer with multi-scale
python scripts/infer_local.py --mode gbm --multiscale --output test_gbm_multi.csv
```
- [ ] 24 features extracted
- [ ] Model loads successfully
- [ ] Predictions saved
- [ ] Scores differ from standard (expected)

## Phase 4: Generalization Testing

### Test 11: CV vs Test Performance
```bash
# Train and note CV AUC
python scripts/train_local.py --mode gbm --multiscale
# Infer on test set with labels
python scripts/infer_local.py --mode gbm --multiscale
```
Compare:
- [ ] Test AUC within 2-3% of CV AUC
- [ ] If Test AUC << CV AUC: Still overfitting
- [ ] If Test AUC >> CV AUC: Lucky test set
- [ ] Ideally: Test AUC ≈ CV AUC ± 0.02

### Test 12: Baseline vs GBM Comparison
```bash
# Baseline
python scripts/train_local.py --mode baseline
# GBM
python scripts/train_local.py --mode gbm --multiscale
```
Compare:
- [ ] GBM CV AUC > Baseline AUC (by 5-10%)
- [ ] If not: Check regularization, features, or data quality

### Test 13: Standard vs Multi-Scale
```bash
# Standard
python scripts/train_local.py --mode gbm
# Multi-scale
python scripts/train_local.py --mode gbm --multiscale
```
Compare:
- [ ] Multi-scale CV AUC >= Standard CV AUC
- [ ] Multi-scale CV std <= Standard CV std (more stable)
- [ ] If not: Multi-scale may not help this dataset

## Phase 5: Determinism & Reproducibility

### Test 14: Determinism Check
```bash
# Run twice
python scripts/train_local.py --mode gbm --multiscale > run1.log
python scripts/train_local.py --mode gbm --multiscale > run2.log
diff run1.log run2.log
```
- [ ] CV AUCs are IDENTICAL across runs
- [ ] Fold AUCs are IDENTICAL across runs
- [ ] Model predictions are IDENTICAL
- [ ] If different: Check random seeds, determinism flags

### Test 15: Sanity Check Script
```bash
python scripts/sanity_check.py
```
- [ ] All checks pass
- [ ] Determinism verified
- [ ] No warnings or errors

## Phase 6: Edge Cases

### Test 16: Small Dataset
```bash
# Test with only first 100 samples
python -c "
from src.sb import data_loader, features, cv_proper
df, y = data_loader.load_for_training('data')
# Take first 100 ids
ids = df['id'].unique()[:100]
df_small = df[df['id'].isin(ids)]
y_small = y[y.index.isin(ids)]
print(f'Small dataset: {len(ids)} samples')
"
```
- [ ] Handles small datasets gracefully
- [ ] CV still runs (may have warnings)
- [ ] No crashes

### Test 17: Imbalanced Labels
Check label distribution:
```bash
python -c "
from src.sb import data_loader
df, y = data_loader.load_for_training('data')
print(f'Label distribution: {dict(y.value_counts())}')
print(f'Break rate: {y.mean():.2%}')
"
```
- [ ] Works with imbalanced data (e.g., 10% breaks)
- [ ] Stratified CV handles imbalance
- [ ] AUC still computed correctly

### Test 18: Missing Values
```bash
python -c "
from src.sb import data_loader, features
df, y = data_loader.load_for_training('data')
X = features.base.compute_features(df, use_multiscale=True)
print(f'NaN count: {X.isna().sum().sum()}')
"
```
- [ ] NaN values from small windows handled
- [ ] Filled with median (0.5 after ranking)
- [ ] No propagation to predictions

## Phase 7: Performance Benchmarks

### Test 19: Speed Test
```bash
time python scripts/train_local.py --mode gbm --multiscale
```
- [ ] Training completes in reasonable time (<5 min for 1000 samples)
- [ ] CV overhead is acceptable
- [ ] Multi-scale adds <2x time vs standard

### Test 20: Memory Usage
```bash
# Monitor memory during training
python scripts/train_local.py --mode gbm --multiscale
```
- [ ] Memory usage is reasonable (<2 GB)
- [ ] No memory leaks
- [ ] Scales with dataset size

## Phase 8: Final Validation

### Test 21: Full Pipeline Test
```bash
# Complete workflow
python scripts/train_local.py --mode gbm --multiscale
python scripts/infer_local.py --mode gbm --multiscale
python scripts/sanity_check.py
```
- [ ] All steps complete successfully
- [ ] CV AUC > 0.75
- [ ] Test AUC ≈ CV AUC
- [ ] Determinism verified

### Test 22: solution.py Integration
```python
from solution import train, infer
from src.sb import data_loader

# Load data
df_train, y_train = data_loader.load_for_training('data')
df_test = data_loader.load_crunchdao_data('data', split='test')

# Train
train(df_train, y_train, use_multiscale=True)

# Infer
predictions = infer(df_test)

print(f"✅ solution.py works! Predictions: {len(predictions)}")
```
- [ ] train() completes with CV output
- [ ] infer() returns scores ∈ [0, 1]
- [ ] Ready for platform submission

## Final Checklist

### Code Quality
- [x] No data leakage (rank norm inside folds)
- [x] Heavy regularization (learning_rate=0.03, etc.)
- [x] Multi-scale features implemented
- [x] Full determinism (fixed seeds, force_col_wise)
- [x] Proper CV evaluation (no in-sample)
- [x] Clean error handling

### Documentation
- [x] REFACTORING.md - Comprehensive guide
- [x] QUICK_START.md - User guide
- [x] REFACTORING_SUMMARY.md - Overview
- [x] TESTING_CHECKLIST.md - This file

### Performance
- [ ] CV AUC > 0.75 (minimum acceptable)
- [ ] CV AUC > 0.80 (good)
- [ ] CV std < 0.03 (stable)
- [ ] Test AUC ≈ CV AUC ± 0.02 (generalizes)

### Ready for Submission
- [ ] All tests pass
- [ ] CV AUC meets target
- [ ] Generalization verified
- [ ] Determinism confirmed
- [ ] Documentation complete

## Troubleshooting

### If Tests Fail

#### Import Errors
```bash
pip install -r requirements.txt
```

#### CV AUC Too Low (<0.70)
- Check data quality
- Try multi-scale features
- Verify labels are correct

#### CV std Too High (>0.05)
- Increase n_folds
- Check for data issues
- Reduce model complexity

#### Test AUC << CV AUC
- Still data leakage somewhere
- Check feature extraction
- Verify inference uses same process

#### Non-Deterministic Results
- Check random seeds
- Verify force_col_wise=True
- Check for threading issues

## Success Criteria

✅ **Pipeline is ready if**:
1. All Phase 1-8 tests pass
2. CV AUC > 0.80
3. CV std < 0.03
4. Test AUC within 2% of CV AUC
5. Results are deterministic

🚀 **Ready for platform submission!**
