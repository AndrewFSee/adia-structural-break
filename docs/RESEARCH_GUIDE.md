# Research Guide: Improving Structural Break Detection

## Current Status
- **Current AUC**: 0.8866 (100 features, diverse LightGBM ensemble + calibration)
- **Target**: 0.88-0.90 AUC (winner range)
- **Gap**: Need ~0.001-0.014 AUC improvement

## Testing in Progress
1. **Autoencoder features** (train_with_autoencoder.py) - Running now
2. **GPT-Researcher** (research_features.py) - Requires API keys

---

## Promising Directions to Explore

### 1. Advanced Statistical Tests ⭐⭐⭐
**Status**: Partially implemented (Anderson-Darling, Cohen's d, Levene, F-test, KS)

**Missing powerful tests:**
- **Cramér-von Mises test**: More powerful than KS for detecting all types of distribution differences
  ```python
  from scipy.stats import cramervonmises_2samp
  stat = cramervonmises_2samp(pre, post).statistic
  ```
  
- **Energy statistic**: Distance-based, powerful for multivariate distributions
  ```python
  from scipy.stats import energy_distance
  energy = energy_distance(pre, post)
  ```
  
- **Wilcoxon rank-sum test**: Robust non-parametric alternative to t-test
  ```python
  from scipy.stats import ranksums
  stat, pval = ranksums(pre, post)
  ```

- **Mood's test**: Tests for scale differences (variance)
  ```python
  from scipy.stats import mood
  stat, pval = mood(pre, post)
  ```

**Implementation priority**: HIGH - Easy to add, likely to improve

---

### 2. Wavelet Transform Features ⭐⭐⭐
**Status**: Not implemented

**Why wavelets?**
- Captures both time and frequency information
- Good for detecting change points at different scales
- Used in winning solutions for similar problems

**Basic implementation:**
```python
import pywt

def wavelet_features(series):
    coeffs = pywt.wavedec(series, 'db4', level=3)
    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_energy_level_{i}'] = np.sum(coeff**2)
        features[f'wavelet_std_level_{i}'] = np.std(coeff)
    return features

# Compare pre vs post
pre_wavelet = wavelet_features(pre)
post_wavelet = wavelet_features(post)
ratio_features = {k: post_wavelet[k]/(pre_wavelet[k]+1e-8) for k in pre_wavelet.keys()}
```

**Implementation priority**: HIGH - Proven effective for change point detection

---

### 3. CUSUM Refinements ⭐⭐
**Status**: Basic CUSUM implemented

**Advanced CUSUM variants:**
- **Weighted CUSUM**: Weight recent observations more
- **Adaptive CUSUM**: Adjust threshold based on variance
- **Page's CUSUM**: Original formulation with optimal detection

**Example - Adaptive CUSUM:**
```python
def adaptive_cusum(series, h_factor=5):
    """h_factor: threshold multiplier (typically 4-5)"""
    mu = np.mean(series)
    sigma = np.std(series)
    h = h_factor * sigma
    
    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))
    
    for i in range(1, len(series)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (series[i] - mu) - sigma/2)
        cusum_neg[i] = max(0, cusum_neg[i-1] - (series[i] - mu) - sigma/2)
    
    return {
        'adaptive_cusum_max_pos': np.max(cusum_pos),
        'adaptive_cusum_max_neg': np.max(cusum_neg),
        'adaptive_cusum_crossings': np.sum((cusum_pos > h) | (cusum_neg > h))
    }
```

**Implementation priority**: MEDIUM - Incremental improvement

---

### 4. Tail Behavior Analysis ⭐⭐⭐
**Status**: Some tail features exist (Hill estimator)

**Additional tail features:**
- **Tail index ratio**: Heavy-tailed vs light-tailed
- **Extreme value theory**: GPD parameters
- **Quantile ratios**: Q95/Q5, Q99/Q1

```python
def tail_features(pre, post):
    # Quantile ratios
    pre_q95_q05 = np.percentile(pre, 95) / (np.percentile(pre, 5) + 1e-8)
    post_q95_q05 = np.percentile(post, 95) / (np.percentile(post, 5) + 1e-8)
    
    # Tail weight (kurtosis)
    from scipy.stats import kurtosis
    pre_kurt = kurtosis(pre)
    post_kurt = kurtosis(post)
    
    # Number of extreme values (> 3 std)
    pre_extremes = np.sum(np.abs(pre - np.mean(pre)) > 3*np.std(pre))
    post_extremes = np.sum(np.abs(post - np.mean(post)) > 3*np.std(post))
    
    return {
        'tail_quantile_ratio_change': post_q95_q05 / (pre_q95_q05 + 1e-8),
        'tail_kurtosis_diff': post_kurt - pre_kurt,
        'tail_extremes_ratio': post_extremes / (pre_extremes + 1)
    }
```

**Implementation priority**: MEDIUM-HIGH - Complements existing features

---

### 5. Rolling Window Statistics ⭐⭐
**Status**: Basic rolling mean/std implemented

**Advanced rolling features:**
- **Rolling correlation**: Autocorrelation structure changes
- **Rolling entropy**: Information content over time
- **Rolling range**: Max-min in windows

```python
def rolling_advanced(series, windows=[10, 20, 50]):
    features = {}
    
    for w in windows:
        # Rolling autocorrelation
        rolling_acf = pd.Series(series).rolling(w).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        )
        features[f'rolling_acf_std_w{w}'] = rolling_acf.std()
        
        # Rolling range
        rolling_range = pd.Series(series).rolling(w).apply(lambda x: x.max() - x.min())
        features[f'rolling_range_mean_w{w}'] = rolling_range.mean()
        
        # Rolling skewness
        from scipy.stats import skew
        rolling_skew = pd.Series(series).rolling(w).apply(lambda x: skew(x))
        features[f'rolling_skew_change_w{w}'] = abs(rolling_skew.iloc[-1] - rolling_skew.iloc[0])
    
    return features
```

**Implementation priority**: MEDIUM - May capture temporal dynamics

---

### 6. Spectral Features (FFT) ⭐⭐
**Status**: Not implemented

**Why spectral analysis?**
- Detects changes in frequency composition
- Reveals periodicity changes
- Complements time-domain features

```python
def spectral_features(pre, post):
    from scipy.fft import fft, fftfreq
    
    # FFT of pre and post
    pre_fft = np.abs(fft(pre))[:len(pre)//2]
    post_fft = np.abs(fft(post))[:len(post)//2]
    
    # Normalize
    pre_fft = pre_fft / (np.sum(pre_fft) + 1e-8)
    post_fft = post_fft / (np.sum(post_fft) + 1e-8)
    
    # Compare distributions
    spectral_diff = np.sum(np.abs(pre_fft - post_fft))
    spectral_kl = np.sum(pre_fft * np.log((pre_fft + 1e-8)/(post_fft + 1e-8)))
    
    # Dominant frequency shift
    pre_dominant = np.argmax(pre_fft)
    post_dominant = np.argmax(post_fft)
    
    return {
        'spectral_l1_distance': spectral_diff,
        'spectral_kl_divergence': spectral_kl,
        'spectral_dominant_freq_shift': abs(post_dominant - pre_dominant)
    }
```

**Implementation priority**: MEDIUM - Different perspective on data

---

### 7. Permutation Entropy ⭐⭐
**Status**: Not implemented

**Why permutation entropy?**
- Complexity measure robust to noise
- Detects changes in time series structure
- Computationally efficient

```python
def permutation_entropy(series, m=3, tau=1):
    """
    m: embedding dimension (3-7 typical)
    tau: time delay
    """
    from itertools import permutations
    
    # Create patterns
    perms = list(permutations(range(m)))
    perm_counts = {p: 0 for p in perms}
    
    for i in range(len(series) - (m-1)*tau):
        idx = [i + j*tau for j in range(m)]
        pattern = tuple(np.argsort(series[idx]))
        perm_counts[pattern] += 1
    
    # Calculate entropy
    total = sum(perm_counts.values())
    probs = [c/total for c in perm_counts.values() if c > 0]
    entropy = -sum(p * np.log(p) for p in probs)
    
    return entropy

# Compare pre vs post
pe_pre = permutation_entropy(pre)
pe_post = permutation_entropy(post)
pe_ratio = pe_post / (pe_pre + 1e-8)
```

**Implementation priority**: MEDIUM - Novel complexity measure

---

### 8. Pseudo-Labeling / Semi-Supervised ⭐
**Status**: Not implemented

**Approach:**
1. Train model on labeled data
2. Get high-confidence predictions on test set
3. Add as "pseudo-labels" to training
4. Retrain

**Benefits:**
- More training data
- Better generalization
- Common in winning Kaggle solutions

**Implementation priority**: LOW - Requires test set access

---

### 9. Bayesian Change Point Detection ⭐⭐
**Status**: Not implemented

**Libraries:**
- `ruptures`: Multiple algorithms (Pelt, Binary Segmentation, Window)
- `bayesian-changepoint-detection`: Bayesian online algorithm

```python
import ruptures as rpt

def bayesian_changepoint(series, n_bkps=1):
    # Pelt algorithm
    algo = rpt.Pelt(model="rbf").fit(series)
    result = algo.predict(pen=10)
    
    # Features
    if len(result) > 0:
        changepoint_location = result[0] / len(series)  # Normalized position
        confidence = 1.0 / (np.std(series) + 1e-8)  # Inverse variance as confidence
    else:
        changepoint_location = 0.5
        confidence = 0.0
    
    return {
        'bayesian_cp_location': changepoint_location,
        'bayesian_cp_confidence': confidence
    }
```

**Implementation priority**: MEDIUM - Different algorithm class

---

### 10. Feature Interactions (Done Right) ⭐
**Status**: Failed attempt (replaced features instead of adding)

**Correct approach:**
- Keep all original features
- Add selected interactions (not all)
- Use domain knowledge to select

**Example:**
```python
# Multiply related features
X['cv_compression_interaction'] = X['cv_global_full'] * X['zlib_pre_full']
X['cusum_wasserstein_interaction'] = X['cusum_pre_path_length_full'] * X['bl_tail_wins_wasserstein_z_w100']

# Ratios of related features
X['cv_compression_ratio'] = X['cv_global_full'] / (X['zlib_pre_full'] + 1e-8)
```

**Implementation priority**: LOW - Previous attempt failed

---

## Quick Wins (Implement First)

### Priority 1: Additional Statistical Tests
- Cramér-von Mises
- Energy statistic
- Wilcoxon rank-sum
- **Estimated time**: 1 hour
- **Expected gain**: +0.001-0.003 AUC

### Priority 2: Wavelet Features
- Multi-scale decomposition
- Energy ratios across scales
- **Estimated time**: 2 hours
- **Expected gain**: +0.002-0.005 AUC

### Priority 3: Tail Behavior
- Quantile ratios
- Extreme value counts
- Kurtosis changes
- **Estimated time**: 1 hour
- **Expected gain**: +0.001-0.002 AUC

---

## Implementation Script Template

```python
# Add to src/sb/features/advanced_tests.py

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import cramervonmises_2samp, energy_distance, ranksums, mood

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract advanced statistical test features."""
    features = []
    
    series_ids = df.index.get_level_values(0).unique()
    
    for series_id in series_ids:
        series_data = df.loc[series_id]
        pre = series_data[series_data['period'] == 0]['value'].values
        post = series_data[series_data['period'] == 1]['value'].values
        
        feat = {}
        
        # Cramér-von Mises
        try:
            cvm = cramervonmises_2samp(pre, post)
            feat['cvm_stat'] = cvm.statistic
            feat['cvm_pval'] = cvm.pvalue
        except:
            feat['cvm_stat'] = 0
            feat['cvm_pval'] = 1
        
        # Energy distance
        try:
            feat['energy_dist'] = energy_distance(pre, post)
        except:
            feat['energy_dist'] = 0
        
        # Wilcoxon rank-sum
        try:
            wstat, wpval = ranksums(pre, post)
            feat['wilcoxon_stat'] = wstat
            feat['wilcoxon_pval'] = wpval
        except:
            feat['wilcoxon_stat'] = 0
            feat['wilcoxon_pval'] = 1
        
        # Mood's test
        try:
            mstat, mpval = mood(pre, post)
            feat['mood_stat'] = mstat
            feat['mood_pval'] = mpval
        except:
            feat['mood_stat'] = 0
            feat['mood_pval'] = 1
        
        features.append(feat)
    
    return pd.DataFrame(features, index=series_ids)
```

---

## Next Steps

1. **Wait for autoencoder results** (running now)
2. **Implement Priority 1**: Advanced statistical tests (quick win)
3. **Implement Priority 2**: Wavelet features (high impact)
4. **Test each addition**: Measure AUC improvement
5. **Combine best features**: Final ensemble

---

## GPT-Researcher Setup (Optional)

To use the research_features.py script:

```powershell
# 1. Get API keys (if you don't have them)
# - OpenAI: https://platform.openai.com/api-keys
# - Tavily: https://tavily.com (free tier available)

# 2. Set environment variables
$env:OPENAI_API_KEY = "sk-..."
$env:TAVILY_API_KEY = "tvly-..."

# 3. Run research
python research_features.py all
```

This will generate detailed research reports on:
- Feature engineering techniques
- Detection methods
- Competition-winning approaches
- Statistical tests
- Compression features
- Transformations

Reports saved to `research_results/` directory.
