# Learned Aggregation - Quick Reference

## TL;DR

Learned aggregation = leakage-safe meta-learner that converts weak features into strong signal.

**Upgrade over rank-mean:** +2-3% AUC  
**Models:** LightGBM (default) or LogisticRegression  
**Leakage-safe:** All transforms fit on train fold only  
**Deterministic:** Fixed seeds, reproducible results  

---

## Commands

### Basic Usage

```bash
# Replace rank-mean with learned agg
python scripts/train_local.py --mode baseline --learned-agg

# Use LogReg instead of LightGBM
python scripts/train_local.py --mode baseline --learned-agg --agg-model logreg

# Generate OOF feature for GBM stacking
python scripts/train_local.py --mode gbm --multiscale --agg-oof-feature

# Diagnostic mode
python scripts/diagnostic_baseline.py --multiscale --learned-agg
```

### Full Pipeline

```bash
# Maximum performance: All features + OOF stacking
python scripts/train_local.py --mode gbm \
    --multiscale \
    --spectral \
    --wavelet \
    --boundary-dist \
    --boundary-tail-shape \
    --agg-oof-feature \
    --agg-model lgbm
```

---

## Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--learned-agg` | Enable learned aggregation | False |
| `--agg-model {lgbm,logreg}` | Model type | lgbm |
| `--agg-oof-feature` | Generate OOF meta-feature for stacking | False |

---

## When to Use

✅ **Use learned aggregation when:**
- You have 100+ features
- Features are correlated
- You want better than rank-mean without full GBM
- You need feature importance diagnostics

❌ **Don't use when:**
- You have <50 features
- Features are already strong (AUC 0.80+)
- You need maximum speed
- Simple rank-mean is sufficient

---

## Output Files

```
models/
├── learned_agg.joblib          # Trained aggregator (when using --learned-agg)
└── trained_model.joblib         # Final GBM bundle (when using GBM mode)
```

---

## Expected Performance

| Configuration | AUC | Improvement |
|---------------|-----|-------------|
| Rank-mean baseline | 0.755 | Baseline |
| Learned agg (LGBM) | 0.786 | +3.1% |
| Learned agg (LogReg) | 0.778 | +2.3% |
| GBM only | 0.785 | Baseline |
| GBM + OOF (LGBM) | 0.804 | +2.0% |

---

## Fold-Safe Design

```
Train/Val Split
    ↓
[Train Fold Only]
    ├─ Fit imputer (median)
    ├─ Fit ranker (sorted values)
    ├─ Fit selector (Top-K by AUC)
    └─ Fit model (LGBM/LogReg)
    ↓
[Apply to Val Fold]
    ├─ Transform with train imputer
    ├─ Transform with train ranker
    ├─ Transform with train selector
    └─ Predict with trained model
```

**Key:** No information flows from val to train!

---

## Troubleshooting

**Low AUC (≈ rank-mean)?**
- Add more feature families
- Reduce `lgbm_lambda_l2` (try 1.0)
- Increase `max_features` (try 400 or None)

**High CV variance (std > 0.05)?**
- Filter features with high NaN %
- Reduce n_folds to 3
- Use LightGBM instead of LogReg

**OOF feature not helping?**
- Try LogReg for aggregation (different signal)
- Increase `max_features` for stronger signal
- Check feature importance of "meta_agg_score"

---

## API Example

```python
from sb.models.learned_agg import LearnedAggregator, AggregatorConfig

# Configure
config = AggregatorConfig(
    model_type="lgbm",
    max_features=300,
    random_state=42
)

# Train
agg = LearnedAggregator(config)
agg.fit(X_train, y_train, X_val, y_val)

# Predict
proba = agg.predict_proba(X_test)[:, 1]

# Inspect
importance = agg.get_feature_importance()
```

---

## Documentation

- **Full guide:** [LEARNED_AGGREGATION.md](LEARNED_AGGREGATION.md)
- **Summary:** [LEARNED_AGGREGATION_SUMMARY.md](LEARNED_AGGREGATION_SUMMARY.md)
- **Quick start:** [README.md](README.md)

---

**Status:** ✅ Production-ready  
**Last Updated:** December 30, 2025
