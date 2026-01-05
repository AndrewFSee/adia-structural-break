# Experiments

This folder contains all training experiments conducted to improve model performance.

## Best Model ⭐

**`train_advanced_tests.py`** - 0.8966 AUC

Uses advanced statistical tests (Cramér-von Mises, energy distance, Wilcoxon, Mood's test, tail features, spectral features, permutation entropy) combined with base features and standard statistical tests.

```bash
python experiments/train_advanced_tests.py
```

## Other Experiments

### Ensemble Approaches

- **`train_final_push.py`** - 0.8866 AUC
  - 5 diverse LightGBM models with different configurations
  - 100 feature selection + isotonic calibration

- **`train_stacking.py`** - 0.8829 AUC
  - Multi-algorithm stacking (LightGBM, XGBoost, CatBoost)
  - Finding: Diverse LightGBM outperforms multi-algorithm

### Feature Selection

- **`train_aggressive_selection.py`** - Tested 33-150 features
  - Finding: 100 features is optimal
  - 33 features: 0.8748 AUC
  - 100 features: 0.8866 AUC
  - 150 features: 0.8866 AUC (same as 100)

- **`train_with_rfe.py`** - Recursive Feature Elimination
  - Computationally expensive, didn't improve

### Deep Learning

- **`train_with_autoencoder_v2.py`** - 0.8682 AUC (-0.0284)
  - PyTorch autoencoder for reconstruction error features
  - Only 1 feature made top 100
  - Finding: Statistical features already capture patterns

### Time Series Methods

- **`train_with_wavelets.py`** - 0.8950 AUC (-0.0016)
  - Wavelet decomposition (3-level db4)
  - Zero wavelet features in top 100
  - Finding: Multi-scale already covered by existing features

- **`train_with_changepoints.py`** - 0.8651 AUC (-0.0315)
  - Bayesian change point detection (ruptures library)
  - Pelt, Binary Segmentation, Window-based methods
  - Only 1 feature made top 100

### Feature Interactions

- **`train_interactions.py`** / **`train_interactions_v2.py`**
  - Polynomial feature interactions
  - Tested but didn't improve significantly

### TabPFN

- **`train_with_tabpfn.py`** - Not completed
  - Requires HuggingFace authentication
  - Transformer-based tabular predictor

## Experiment Summary

| Experiment | AUC | Change | Features | Notes |
|------------|-----|--------|----------|-------|
| **Advanced Tests** ⭐ | **0.8966** | **baseline** | 100 | **Best model** |
| Final Push | 0.8866 | -0.0100 | 100 | Diverse ensemble |
| Stacking | 0.8829 | -0.0137 | 100 | Multi-algorithm |
| Wavelets | 0.8950 | -0.0016 | 100 | 0 wavelets selected |
| Autoencoder | 0.8682 | -0.0284 | 100 | 1 AE feature selected |
| Change Points | 0.8651 | -0.0315 | 100 | 1 CP feature selected |
| Aggressive (150) | 0.8866 | -0.0100 | 150 | Same as 100 |
| Aggressive (33) | 0.8748 | -0.0218 | 33 | Too few features |

## Key Learnings

1. **Statistical tests >> Deep learning**: Simple statistical tests (Cramér-von Mises, energy distance) beat sophisticated neural networks

2. **Feature selection matters**: 100 features is the sweet spot (33 too few, 150 no better)

3. **Diverse LightGBM wins**: 5 diverse LightGBM models > stacking with multiple algorithms

4. **Existing features sufficient**: Wavelets, autoencoders, change points don't add value beyond current statistical features

5. **Advanced tests are key**: 8 advanced statistical features made top 100, including `energy_cumsum` at #7 overall
