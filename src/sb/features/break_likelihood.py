"""
Break likelihood scoring using hand-selected high-signal features.

This module provides a simple, interpretable baseline that aggregates
a small set of top-performing features into a single break likelihood score.

Two scoring modes:
- rank_mean: Rank-based aggregation (robust to outliers)
- zscore_logit: Z-score weighted sum through sigmoid (parametric)

All operations are leakage-safe: statistics computed per-fold across ids only.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List


class BreakLikelihoodScorer:
    """
    Fold-safe break likelihood scorer.
    
    Fits on train data, transforms train/val/test consistently.
    This prevents leakage when using break_likelihood as a feature in CV.
    """
    
    def __init__(self, 
                 feature_list: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 mode: str = "rank_mean"):
        """
        Initialize scorer.
        
        Args:
            feature_list: List of feature names to aggregate (default: DEFAULT_FEATURE_LIST)
            weights: Dict of feature -> weight (default: DEFAULT_WEIGHTS)
            mode: Scoring mode ("rank_mean" or "zscore_logit")
        """
        self.feature_list = feature_list
        self.weights = weights
        self.mode = mode
        self.available_features_ = None
        self.feature_weights_ = None
        self.train_sorted_ = {}  # For rank_mean mode
        self.train_medians_ = {}  # For zscore_logit mode
        self.train_iqrs_ = {}  # For zscore_logit mode
        
    def fit(self, X_train: pd.DataFrame) -> 'BreakLikelihoodScorer':
        """
        Fit scorer on training data.
        
        Args:
            X_train: Training features (raw, not ranked)
            
        Returns:
            self
        """
        # Use defaults if not specified
        feature_list = self.feature_list if self.feature_list is not None else DEFAULT_FEATURE_LIST
        weights = self.weights if self.weights is not None else DEFAULT_WEIGHTS
        
        # Select available features
        self.available_features_ = [f for f in feature_list if f in X_train.columns]
        
        if len(self.available_features_) == 0:
            # No features available - will return 0.5 in transform
            return self
        
        # Get weights for available features (normalize to sum to 1)
        self.feature_weights_ = np.array([weights.get(f, 1.0) for f in self.available_features_])
        self.feature_weights_ = self.feature_weights_ / self.feature_weights_.sum()
        
        if self.mode == "rank_mean":
            # Store sorted values per feature for ranking
            for feat in self.available_features_:
                values = X_train[feat].values
                valid = values[~np.isnan(values)]
                self.train_sorted_[feat] = np.sort(valid) if len(valid) > 0 else np.array([])
                
        elif self.mode == "zscore_logit":
            # Store robust statistics per feature
            for feat in self.available_features_:
                values = X_train[feat].values
                valid = values[~np.isnan(values)]
                if len(valid) > 0:
                    self.train_medians_[feat] = np.median(valid)
                    q75, q25 = np.percentile(valid, [75, 25])
                    iqr = q75 - q25
                    self.train_iqrs_[feat] = iqr if iqr > 1e-8 else 1.0
                else:
                    self.train_medians_[feat] = 0.0
                    self.train_iqrs_[feat] = 1.0
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.Series:
        """
        Transform features to break likelihood score.
        
        Args:
            X: Features to transform (raw, not ranked)
            
        Returns:
            Series of break likelihood scores in [0, 1]
        """
        if self.available_features_ is None:
            raise ValueError("Scorer not fitted. Call fit() first.")
        
        if len(self.available_features_) == 0:
            # No features available - return neutral score
            return pd.Series(0.5, index=X.index, name='break_likelihood')
        
        # Extract selected features
        X_selected = X[self.available_features_].copy()
        
        if self.mode == "rank_mean":
            # Rank each feature using train distribution
            X_ranked = pd.DataFrame(index=X.index, columns=self.available_features_)
            
            for feat in self.available_features_:
                sorted_train = self.train_sorted_[feat]
                
                if len(sorted_train) == 0:
                    X_ranked[feat] = 0.5
                else:
                    values = X_selected[feat].values
                    ranks = np.searchsorted(sorted_train, values, side='right')
                    normalized = ranks / len(sorted_train)
                    normalized = np.clip(normalized, 0.0, 1.0)
                    X_ranked[feat] = normalized
            
            # Fill NaN with 0.5
            X_ranked = X_ranked.fillna(0.5)
            
            # Weighted mean
            scores = (X_ranked.astype(float) * self.feature_weights_).sum(axis=1)
            
        elif self.mode == "zscore_logit":
            # Robust z-score using train statistics
            X_z = pd.DataFrame(index=X.index, columns=self.available_features_)
            
            for feat in self.available_features_:
                values = X_selected[feat].values
                median = self.train_medians_[feat]
                iqr = self.train_iqrs_[feat]
                z = (values - median) / iqr
                X_z[feat] = z
            
            # Fill NaN with 0 (neutral)
            X_z = X_z.fillna(0.0)
            
            # Weighted sum
            weighted_sum = (X_z.astype(float) * self.feature_weights_).sum(axis=1)
            
            # Sigmoid to [0, 1]
            scores = 1.0 / (1.0 + np.exp(-weighted_sum))
            scores = pd.Series(scores, index=X.index)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Ensure [0, 1] range and proper naming
        scores = scores.clip(0.0, 1.0)
        scores.name = 'break_likelihood'
        
        return scores
    
    def fit_transform(self, X_train: pd.DataFrame) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(X_train).transform(X_train)


# Default feature list: hand-selected top performers
# Prioritize AR(1) residual changes and energy distance
DEFAULT_FEATURE_LIST = [
    # AR(1) residual features (strongest predictors)
    'delta_rmse',
    'delta_rmse_w50',
    'delta_rmse_w100',
    'delta_rmse_w250',
    'delta_resid_var',
    'delta_resid_var_w50',
    'delta_resid_var_w100',
    'delta_resid_var_w250',
    
    # Energy distance (strong two-sample statistic)
    'energy',
    'energy_w50',
    'energy_w100',
    'energy_w250',
    
    # Wasserstein distance
    'wasserstein',
    'wasserstein_w250',
    
    # AR(1) coefficient and autocorrelation changes
    'delta_ar1_phi',
    'delta_ar1_phi_w50',
    'delta_ar1_phi_w100',
    'delta_ar1_phi_w250',
    'acf1_shift',
    'acf1_shift_w50',
    'acf1_shift_w100',
    'acf1_shift_w250',
    
    # MAD ratio (scale changes)
    'mad_ratio',
    'mad_ratio_w50',
    'mad_ratio_w100',
    'mad_ratio_w250',
    
    # Spectral features (if available)
    'delta_log_low_high',
    'delta_log_low_high_w50',
    'delta_log_low_high_w100',
    'delta_log_low_high_w250',
    'delta_spec_entropy',
    'delta_spec_entropy_w50',
    'delta_spec_entropy_w100',
    'delta_spec_entropy_w250',
    'delta_peak_ratio',
    'delta_flatness',
    'delta_flux',
    'delta_rolloff50',
    'delta_bandwidth',
    'delta_hf_power',
    
    # Wavelet energy changes (if available)
    'delta_wav_energy_l1',
    'delta_wav_energy_l1_w250',
    'delta_wav_energy_l2',
    'delta_wav_energy_l2_w250',
    'delta_wav_energy_l3',
    'delta_wav_energy_l3_w250',
    'delta_wav_entropy',
    'delta_wav_low_energy_share',
    'delta_wav_high_energy_share',
]

# Default weights: equal for simplicity
# Can be tuned based on per-feature AUC
DEFAULT_WEIGHTS = {
    # AR(1) residual features - highest priority
    'delta_rmse': 2.0,
    'delta_rmse_w50': 1.8,
    'delta_rmse_w100': 2.0,
    'delta_rmse_w250': 1.8,
    'delta_resid_var': 1.5,
    'delta_resid_var_w50': 1.3,
    'delta_resid_var_w100': 1.5,
    'delta_resid_var_w250': 1.3,
    
    # Energy distance - high priority
    'energy': 1.5,
    'energy_w50': 1.3,
    'energy_w100': 1.3,
    'energy_w250': 1.5,
    'wasserstein': 1.2,
    'wasserstein_w250': 1.2,
    
    # AR(1) coefficient changes - medium-high priority
    'delta_ar1_phi': 1.2,
    'delta_ar1_phi_w50': 1.0,
    'delta_ar1_phi_w100': 1.0,
    'delta_ar1_phi_w250': 1.2,
    'acf1_shift': 1.0,
    'acf1_shift_w50': 0.9,
    'acf1_shift_w100': 0.9,
    'acf1_shift_w250': 1.0,
    
    # MAD ratio - medium priority
    'mad_ratio': 1.2,
    'mad_ratio_w50': 1.0,
    'mad_ratio_w100': 1.0,
    'mad_ratio_w250': 1.2,
    
    # Spectral features - medium priority
    'delta_log_low_high': 0.9,
    'delta_log_low_high_w50': 0.8,
    'delta_log_low_high_w100': 0.8,
    'delta_log_low_high_w250': 0.9,
    'delta_spec_entropy': 0.9,
    'delta_spec_entropy_w50': 0.8,
    'delta_spec_entropy_w100': 0.8,
    'delta_spec_entropy_w250': 0.9,
    'delta_peak_ratio': 0.8,
    'delta_flatness': 0.7,
    'delta_flux': 0.7,
    'delta_rolloff50': 0.7,
    'delta_bandwidth': 0.7,
    'delta_hf_power': 0.7,
    
    # Wavelet features - lower priority (exploratory)
    'delta_wav_energy_l1': 0.8,
    'delta_wav_energy_l1_w250': 0.8,
    'delta_wav_energy_l2': 0.7,
    'delta_wav_energy_l2_w250': 0.7,
    'delta_wav_energy_l3': 0.6,
    'delta_wav_energy_l3_w250': 0.6,
    'delta_wav_entropy': 0.7,
    'delta_wav_low_energy_share': 0.6,
    'delta_wav_high_energy_share': 0.6,
}


def compute_break_likelihood(
    feature_df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    mode: str = "rank_mean"
) -> pd.Series:
    """
    Compute break likelihood score from selected features.
    
    Args:
        feature_df: DataFrame with index=id, columns=features (raw values)
        feature_list: List of feature names to use (default: DEFAULT_FEATURE_LIST)
        weights: Dict of feature_name -> weight (default: DEFAULT_WEIGHTS)
        mode: Scoring mode - "rank_mean" or "zscore_logit"
        
    Returns:
        Series with index=id, values in [0, 1] representing break likelihood
    """
    if feature_list is None:
        feature_list = DEFAULT_FEATURE_LIST
    
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    # Select available features
    available_features = [f for f in feature_list if f in feature_df.columns]
    
    if len(available_features) == 0:
        # No features found - return neutral score
        return pd.Series(0.5, index=feature_df.index, name='score')
    
    # Extract selected features
    X_selected = feature_df[available_features].copy()
    
    # Get weights for available features (normalize to sum to 1)
    feature_weights = np.array([weights.get(f, 1.0) for f in available_features])
    feature_weights = feature_weights / feature_weights.sum()
    
    if mode == "rank_mean":
        scores = _rank_mean_score(X_selected, feature_weights)
    elif mode == "zscore_logit":
        scores = _zscore_logit_score(X_selected, feature_weights)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'rank_mean' or 'zscore_logit'")
    
    # Ensure [0, 1] range and proper naming
    scores = scores.clip(0.0, 1.0)
    scores.name = 'score'
    
    return scores


def fit_break_likelihood_scorer(
    X_train: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    mode: str = "rank_mean"
) -> BreakLikelihoodScorer:
    """
    Fit a break likelihood scorer on training data.
    
    This is the fold-safe API for CV. Use this when you need to:
    - Fit on train fold
    - Apply to val/test fold
    
    Args:
        X_train: Training features (raw, not ranked)
        feature_list: List of feature names to use (default: DEFAULT_FEATURE_LIST)
        weights: Dict of feature_name -> weight (default: DEFAULT_WEIGHTS)
        mode: Scoring mode - "rank_mean" or "zscore_logit"
        
    Returns:
        Fitted BreakLikelihoodScorer
    """
    scorer = BreakLikelihoodScorer(
        feature_list=feature_list,
        weights=weights,
        mode=mode
    )
    scorer.fit(X_train)
    return scorer


def _rank_mean_score(X: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Compute weighted mean of rank-normalized features.
    
    Args:
        X: Feature DataFrame
        weights: Array of weights (same length as X.columns)
        
    Returns:
        Series of scores in [0, 1]
    """
    # Rank-normalize each feature to [0, 1]
    X_ranked = X.rank(pct=True, method='average')
    
    # Fill NaN with median rank (0.5)
    X_ranked = X_ranked.fillna(0.5)
    
    # Weighted mean across features
    scores = (X_ranked * weights).sum(axis=1)
    
    return scores


def _zscore_logit_score(X: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Compute weighted z-score sum through sigmoid.
    
    Uses robust z-score: (x - median) / IQR
    
    Args:
        X: Feature DataFrame
        weights: Array of weights (same length as X.columns)
        
    Returns:
        Series of scores in [0, 1]
    """
    # Robust z-score normalization
    X_z = X.copy()
    
    for col in X.columns:
        values = X[col].values
        
        # Skip if all NaN
        if np.isnan(values).all():
            X_z[col] = 0.0
            continue
        
        # Robust statistics
        median = np.nanmedian(values)
        q75 = np.nanpercentile(values, 75)
        q25 = np.nanpercentile(values, 25)
        iqr = q75 - q25
        
        # Z-score with IQR scaling
        if iqr < 1e-10:
            X_z[col] = 0.0
        else:
            X_z[col] = (values - median) / (iqr + 1e-10)
    
    # Fill remaining NaN with 0
    X_z = X_z.fillna(0.0)
    
    # Clip to reasonable range
    X_z = X_z.clip(-6.0, 6.0)
    
    # Weighted sum
    logits = (X_z * weights).sum(axis=1)
    
    # Sigmoid to [0, 1]
    scores = 1.0 / (1.0 + np.exp(-logits))
    
    return scores


def print_feature_diagnostics(
    feature_df: pd.DataFrame,
    y: Optional[pd.Series] = None,
    feature_list: Optional[List[str]] = None
):
    """
    Print diagnostics for features used in break likelihood.
    
    Args:
        feature_df: Feature DataFrame
        y: Labels (optional, for computing AUC)
        feature_list: List of features to check (default: DEFAULT_FEATURE_LIST)
    """
    if feature_list is None:
        feature_list = DEFAULT_FEATURE_LIST
    
    print("\n" + "=" * 70)
    print("BREAK LIKELIHOOD FEATURE DIAGNOSTICS")
    print("=" * 70)
    
    print(f"\nRequested features: {len(feature_list)}")
    
    available = [f for f in feature_list if f in feature_df.columns]
    missing = [f for f in feature_list if f not in feature_df.columns]
    
    print(f"Available features: {len(available)}")
    print(f"Missing features:   {len(missing)}")
    
    if available:
        print(f"\n✓ Available features:")
        for feat in available:
            nan_pct = feature_df[feat].isna().mean() * 100
            print(f"  - {feat:30s} (NaN: {nan_pct:5.1f}%)")
    
    if missing:
        print(f"\n✗ Missing features:")
        for feat in missing:
            print(f"  - {feat}")
    
    if y is not None and len(available) > 0:
        print(f"\n" + "=" * 70)
        print("INDIVIDUAL FEATURE AUC")
        print("=" * 70)
        
        from sklearn.metrics import roc_auc_score
        
        aucs = []
        for feat in available:
            try:
                vals = feature_df[feat].fillna(feature_df[feat].median())
                auc = roc_auc_score(y, vals)
                auc_flipped = roc_auc_score(y, -vals)
                auc_best = max(auc, auc_flipped)
                aucs.append((feat, auc_best))
            except:
                aucs.append((feat, 0.5))
        
        aucs.sort(key=lambda x: x[1], reverse=True)
        
        for feat, auc in aucs[:10]:  # Top 10
            print(f"  {feat:30s}: {auc:.4f}")
        
        if len(aucs) > 10:
            print(f"  ... ({len(aucs) - 10} more)")
    
    print("=" * 70)


if __name__ == "__main__":
    # Self-check: verify scoring works correctly
    print("=" * 70)
    print("BREAK LIKELIHOOD SELF-CHECK")
    print("=" * 70)
    
    # Create toy feature data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate features with different signal strengths
    data = {
        'delta_rmse': np.random.randn(n_samples) * 2 + 1.0,  # Strong signal
        'delta_rmse_w50': np.random.randn(n_samples) * 1.5 + 0.5,
        'energy': np.random.randn(n_samples) + 0.3,
        'delta_resid_var': np.random.randn(n_samples) * 2,
        'acf1_shift': np.random.randn(n_samples) * 0.5,
    }
    
    # Add some NaN values
    data['delta_rmse'][10:15] = np.nan
    
    feature_df = pd.DataFrame(data, index=[f'id_{i}' for i in range(n_samples)])
    
    print(f"\nToy feature data: {feature_df.shape}")
    print(f"Features: {list(feature_df.columns)}")
    print(f"NaN count: {feature_df.isna().sum().sum()}")
    
    # Test rank_mean mode
    print("\n" + "-" * 70)
    print("Testing rank_mean mode...")
    print("-" * 70)
    
    scores_rank = compute_break_likelihood(
        feature_df,
        feature_list=['delta_rmse', 'delta_rmse_w50', 'energy', 'delta_resid_var', 'acf1_shift'],
        mode="rank_mean"
    )
    
    print(f"\nScores computed: {len(scores_rank)}")
    print(f"Score range: [{scores_rank.min():.4f}, {scores_rank.max():.4f}]")
    print(f"Score mean: {scores_rank.mean():.4f}")
    print(f"Score std: {scores_rank.std():.4f}")
    
    print(f"\nTop 5 scores:")
    print(scores_rank.nlargest(5))
    
    print(f"\nBottom 5 scores:")
    print(scores_rank.nsmallest(5))
    
    # Test zscore_logit mode
    print("\n" + "-" * 70)
    print("Testing zscore_logit mode...")
    print("-" * 70)
    
    scores_zscore = compute_break_likelihood(
        feature_df,
        feature_list=['delta_rmse', 'delta_rmse_w50', 'energy', 'delta_resid_var', 'acf1_shift'],
        mode="zscore_logit"
    )
    
    print(f"\nScores computed: {len(scores_zscore)}")
    print(f"Score range: [{scores_zscore.min():.4f}, {scores_zscore.max():.4f}]")
    print(f"Score mean: {scores_zscore.mean():.4f}")
    print(f"Score std: {scores_zscore.std():.4f}")
    
    # Test with missing features
    print("\n" + "-" * 70)
    print("Testing with missing features...")
    print("-" * 70)
    
    scores_missing = compute_break_likelihood(
        feature_df,
        feature_list=['delta_rmse', 'missing_feature_1', 'energy', 'missing_feature_2'],
        mode="rank_mean"
    )
    
    print(f"Requested 4 features, 2 missing")
    print(f"Scores computed: {len(scores_missing)}")
    print(f"Score range: [{scores_missing.min():.4f}, {scores_missing.max():.4f}]")
    
    # Test with no available features
    print("\n" + "-" * 70)
    print("Testing with all missing features...")
    print("-" * 70)
    
    scores_none = compute_break_likelihood(
        feature_df,
        feature_list=['missing_1', 'missing_2', 'missing_3'],
        mode="rank_mean"
    )
    
    print(f"All features missing, should return 0.5")
    print(f"Unique values: {scores_none.unique()}")
    assert (scores_none == 0.5).all(), "Should return 0.5 for all when no features available"
    
    # Test determinism
    print("\n" + "-" * 70)
    print("Testing determinism...")
    print("-" * 70)
    
    scores1 = compute_break_likelihood(feature_df, mode="rank_mean")
    scores2 = compute_break_likelihood(feature_df, mode="rank_mean")
    
    max_diff = (scores1 - scores2).abs().max()
    print(f"Max difference on repeat: {max_diff:.2e}")
    assert max_diff == 0.0, "Should be perfectly deterministic"
    
    print("\n" + "=" * 70)
    print("✓ ALL SELF-CHECKS PASSED")
    print("=" * 70)
    
    print("\nUsage examples:")
    print("  1. In diagnostic scripts:")
    print("     scores = break_likelihood.compute_break_likelihood(X_raw, mode='rank_mean')")
    print("\n  2. With custom features:")
    print("     scores = break_likelihood.compute_break_likelihood(")
    print("         X_raw,")
    print("         feature_list=['delta_rmse', 'energy', 'delta_resid_var'],")
    print("         weights={'delta_rmse': 2.0, 'energy': 1.0, 'delta_resid_var': 1.5},")
    print("         mode='zscore_logit'")
    print("     )")
