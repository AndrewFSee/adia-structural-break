# Development Checklist

## Day 1-2: Baseline Setup ✅

### Installation & Verification
- [ ] Create virtual environment
- [ ] Install dependencies (`pip install -e .`)
- [ ] Run `python setup_check.py` - all tests pass
- [ ] Run `python scripts/sanity_check.py` - all tests pass

### Data Preparation
- [ ] Download training data from CrunchDAO
- [ ] Verify data format (id, period, value columns)
- [ ] Check for missing values or anomalies
- [ ] Understand label distribution (balanced?)

### Baseline Training
- [ ] Run local training: `python scripts/train_local.py --data train.csv`
- [ ] Check ROC AUC (target: 0.80+)
- [ ] Inspect feature distributions
- [ ] Verify determinism (run twice, same result)

### Feature Analysis
- [ ] Extract features: `python scripts/make_features.py --data train.csv`
- [ ] Visualize feature distributions by label
- [ ] Check for correlations between features
- [ ] Identify which features are most discriminative

### Validation
- [ ] Split data into train/validation
- [ ] Test on validation set
- [ ] Check score distribution (break vs no-break separation)
- [ ] Verify no data leakage

## Day 3-4: Enhancement (Optional)

### Multi-scale Features
- [ ] Implement features at different window sizes
- [ ] Test on subsets of data near the break point
- [ ] Compare performance vs baseline

### Meta-Model
- [ ] Uncomment LightGBM code in solution.py
- [ ] Train with cross-validation
- [ ] Check feature importances
- [ ] Avoid overfitting (validate on hold-out)

### Additional Features
- [ ] Frequency domain (FFT ratios)
- [ ] Autocorrelation changes (ACF deltas)
- [ ] Permutation entropy
- [ ] Slope changes (robust regression)
- [ ] Extreme value features

### Originality Check
- [ ] Compute correlation with KS test
- [ ] Compute correlation with t-test
- [ ] Compute correlation with variance ratio test
- [ ] Ensure Spearman correlation < 0.7 with each

## Day 5: Pre-Submission

### Final Validation
- [ ] Run all sanity checks: `python scripts/sanity_check.py`
- [ ] Test determinism: `python tests/test_determinism.py`
- [ ] Test speed on large sample (extrapolate to 16k)
- [ ] Verify score range [0, 1]

### Code Quality
- [ ] Remove debug prints
- [ ] Clean up unused imports
- [ ] Check for hardcoded paths
- [ ] Ensure all seeds are set

### Documentation
- [ ] Update README if needed
- [ ] Document any custom features
- [ ] Note any assumptions made

### Submission Prep
- [ ] Test solution.py in isolation
- [ ] Verify train() and infer() signatures
- [ ] Check that solution.py imports work
- [ ] Test on fresh Python environment

## Submission Day

### Pre-Upload
- [ ] One final sanity check run
- [ ] Verify solution.py is latest version
- [ ] Check file size (not too large)
- [ ] Review submission guidelines

### Upload
- [ ] Upload solution.py to CrunchDAO
- [ ] Wait for initial validation
- [ ] Check for any error messages
- [ ] Verify submission is accepted

### Post-Submission
- [ ] Check leaderboard position
- [ ] Note public score
- [ ] Save this version (git tag)
- [ ] Document what worked/didn't work

## Iteration Checklist

For each improvement attempt:

- [ ] Create a branch (if using git)
- [ ] Make ONE change at a time
- [ ] Test locally (`train_local.py`)
- [ ] Run sanity checks
- [ ] Compare to previous best
- [ ] If better: merge and submit
- [ ] If worse: analyze why, revert

## Quick Reference

### Must-Have Before Submission
✅ Deterministic (sanity check passes)
✅ Fast enough (< reasonable time for 16k series)
✅ Scores in [0, 1]
✅ No missing predictions
✅ ROC AUC competitive (0.80+)

### Nice-to-Have
⭐ Feature importance analysis
⭐ Cross-validation results
⭐ Originality verification
⭐ Multiple model variants tested

### Avoid
❌ Non-deterministic operations
❌ Hardcoded file paths
❌ External dependencies not in pyproject.toml
❌ Over-optimizing on training data
❌ Copying standard test statistics directly

---

Use this checklist to track your progress. Good luck! 🚀
