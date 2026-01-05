"""
Feature orchestration: compute all features and aggregate them.
"""

import numpy as np
import pandas as pd
from typing import Dict
from .. import io, preprocessing
from . import dist, dynamics, multiscale, spectral, wavelet, break_likelihood, boundary, boundary_dist, boundary_tail_shape
from .model_based import ar_features

# Phase 1 features (winning solutions)
try:
    from . import cv_features, transformations, compression, cusum
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False


def compute_single_series_features(
    x0: np.ndarray,
    x1: np.ndarray,
    use_multiscale: bool = False,
    use_spectral: bool = False,
    use_wavelet: bool = False,
    use_boundary: bool = False,
    use_boundary_dist: bool = False,
    use_boundary_tail_shape: bool = False,
    use_cv: bool = False,
    use_transforms: bool = False,
    use_compression: bool = False,
    use_cusum: bool = False,
    transform_list=None
) -> Dict[str, float]:
    """
    Compute all features for a single time series.
    
    Args:
        x0: Pre-break values
        x1: Post-break values
        use_multiscale: Whether to include multi-scale features
        use_spectral: Whether to include spectral features
        use_wavelet: Whether to include wavelet features
        use_boundary: Whether to include boundary-localized contrast features
        use_boundary_dist: Whether to include boundary distribution distance features
        use_boundary_tail_shape: Whether to include boundary tail-shape features
        use_cv: Whether to include CV features (Phase 1 - "magic" feature)
        use_transforms: Whether to compute features on transformed series (Phase 1)
        use_compression: Whether to include compression features (Phase 1)
        use_cusum: Whether to include CUSUM features (Phase 1)
        transform_list: List of transforms to use (default: ['raw', 'cumsum', 'diff'])
        
    Returns:
        Dictionary of feature values
    """
    if use_multiscale:
        # Multi-scale features (includes base features)
        features = multiscale.compute_multiscale_features(x0, x1, include_spectral=use_spectral, include_wavelet=use_wavelet)
    else:
        # Original single-scale features
        # Apply robust scaling separately to pre and post segments
        x0_scaled = preprocessing.robust_scale(x0)
        x1_scaled = preprocessing.robust_scale(x1)
        
        # Day 1 features: distribution shape
        features = dist.quantile_features(x0_scaled, x1_scaled)
        features["delta_entropy"] = dist.entropy_change(x0_scaled, x1_scaled)
        
        # Day 2 features: transition dynamics
        vol_features = dynamics.volatility_features(x0_scaled, x1_scaled)
        features.update(vol_features)
        
        # Two-sample statistics
        features["energy"] = dist.energy_distance_1d(x0_scaled, x1_scaled)
        features["wasserstein"] = dist.wasserstein_1d(x0_scaled, x1_scaled)
        q_deltas = dist.quantile_deltas(x0_scaled, x1_scaled, qs=[0.1, 0.5, 0.9])
        features.update(q_deltas)
        scale_feats = dist.scale_shift(x0_scaled, x1_scaled)
        features["mad_ratio"] = scale_feats["mad_ratio"]
        features["iqr_ratio_robust"] = scale_feats["iqr_ratio"]
        features["acf1_shift"] = dist.acf1_shift(x0, x1)
        
        # AR(1) predictability features
        ar_feats = ar_features.ar1_features(x0, x1)
        features.update(ar_feats)
    
    # Spectral features (optional, compatible with both single-scale and multi-scale)
    if use_spectral:
        spec_feats = spectral.spectral_features_all(x0, x1, include_v2=True)
        features.update(spec_feats)
    
    # Wavelet features (optional, compatible with both single-scale and multi-scale)
    if use_wavelet and not use_multiscale:
        wav_feats = wavelet.wavelet_features(x0, x1)
        features.update(wav_feats)
    
    # Boundary-localized contrast features (optional)
    if use_boundary:
        from .. import config
        windows = config.MULTI_SCALE_WINDOWS if hasattr(config, 'MULTI_SCALE_WINDOWS') else (25, 50, 100, 250)
        boundary_feats = boundary.compute_boundary_contrast_features(x0, x1, windows=windows)
        features.update(boundary_feats)
    
    # Boundary distribution distance features (optional)
    if use_boundary_dist:
        from .. import config
        windows = config.MULTI_SCALE_WINDOWS if hasattr(config, 'MULTI_SCALE_WINDOWS') else (25, 50, 100, 250)
        boundary_dist_feats = boundary_dist.compute_boundary_dist_features(x0, x1, windows=windows)
        features.update(boundary_dist_feats)
    
    # Boundary tail-shape features (optional)
    if use_boundary_tail_shape:
        from .. import config
        windows = config.MULTI_SCALE_WINDOWS if hasattr(config, 'MULTI_SCALE_WINDOWS') else (25, 50, 100, 250)
        boundary_tail_shape_feats = boundary_tail_shape.compute_boundary_tail_shape_features(x0, x1, windows=windows)
        features.update(boundary_tail_shape_feats)
    
    # === PHASE 1 FEATURES (Winning Solutions) ===
    
    if PHASE1_AVAILABLE:
        # CV features (THE "magic" feature from winning solutions)
        if use_cv:
            from .. import config
            windows = config.MULTI_SCALE_WINDOWS if hasattr(config, 'MULTI_SCALE_WINDOWS') else (50, 100, 250)
            cv_feats = cv_features.compute_cv_features_multiscale(x0, x1, windows=windows)
            features.update(cv_feats)
        
        # Compression features
        if use_compression:
            from .. import config
            windows = config.MULTI_SCALE_WINDOWS if hasattr(config, 'MULTI_SCALE_WINDOWS') else (50, 100, 250)
            comp_feats = compression.compute_compression_features_multiscale(x0, x1, windows=windows)
            features.update(comp_feats)
        
        # CUSUM features
        if use_cusum:
            from .. import config
            windows = config.MULTI_SCALE_WINDOWS if hasattr(config, 'MULTI_SCALE_WINDOWS') else (50, 100, 250)
            cusum_feats = cusum.compute_cusum_features_multiscale(x0, x1, windows=windows)
            features.update(cusum_feats)
        
        # Transformations (compute existing features on transformed series)
        if use_transforms:
            if transform_list is None:
                transform_list = transformations.get_priority_transforms()  # ['raw', 'cumsum', 'diff']
            
            # Create a simplified feature function for transformations
            def basic_features(x0_t, x1_t):
                """Compute a subset of fast features on transformed series."""
                feats = {}
                
                # Only compute if we have enough data
                if len(x0_t) < 5 or len(x1_t) < 5:
                    return feats
                
                # Basic statistics
                feats['mean_diff'] = np.mean(x1_t) - np.mean(x0_t)
                feats['std_ratio'] = np.std(x1_t) / (np.std(x0_t) + 1e-8)
                feats['median_diff'] = np.median(x1_t) - np.median(x0_t)
                
                # Quantiles
                for q in [0.1, 0.5, 0.9]:
                    q0 = np.percentile(x0_t, q*100)
                    q1 = np.percentile(x1_t, q*100)
                    feats[f'q{int(q*100)}_diff'] = q1 - q0
                
                # Wasserstein and energy (key features)
                try:
                    feats['wasserstein'] = dist.wasserstein_1d(x0_t, x1_t)
                    feats['energy'] = dist.energy_distance_1d(x0_t, x1_t)
                except:
                    feats['wasserstein'] = 0.0
                    feats['energy'] = 0.0
                
                return feats
            
            # Compute features on each transformation
            transform_feats = transformations.compute_transform_features(
                x0, x1, basic_features, transform_list
            )
            features.update(transform_feats)
    
    return features


def compute_features(
    df: pd.DataFrame,
    use_multiscale: bool = False,
    use_spectral: bool = False,
    use_wavelet: bool = False,
    use_break_likelihood: bool = False,
    use_boundary: bool = False,
    use_boundary_dist: bool = False,
    use_boundary_tail_shape: bool = False,
    use_cv: bool = False,
    use_transforms: bool = False,
    use_compression: bool = False,
    use_cusum: bool = False,
    transform_list=None
) -> pd.DataFrame:
    """
    Compute features for all time series in the dataset.
    
    Args:
        df: DataFrame with columns [id, period, value]
        use_multiscale: Whether to include multi-scale features
        use_spectral: Whether to include spectral features
        use_wavelet: Whether to include wavelet features
        use_break_likelihood: Whether to add break_likelihood meta-feature
        use_boundary: Whether to include boundary-localized contrast features
        use_boundary_dist: Whether to include boundary distribution distance features
        use_boundary_tail_shape: Whether to include boundary tail-shape features
        use_cv: Whether to include CV features (Phase 1 - "magic" feature)
        use_transforms: Whether to compute features on transformed series (Phase 1)
        use_compression: Whether to include compression features (Phase 1)
        use_cusum: Whether to include CUSUM features (Phase 1)
        transform_list: List of transforms to use (default: ['raw', 'cumsum', 'diff'])
        
    Returns:
        DataFrame with index=id and columns=features
    """
    feature_list = []
    ids = []
    
    for series_id, x0, x1 in io.iter_series(df):
        features = compute_single_series_features(
            x0, x1,
            use_multiscale=use_multiscale,
            use_spectral=use_spectral,
            use_wavelet=use_wavelet,
            use_boundary=use_boundary,
            use_boundary_dist=use_boundary_dist,
            use_boundary_tail_shape=use_boundary_tail_shape,
            use_cv=use_cv,
            use_transforms=use_transforms,
            use_compression=use_compression,
            use_cusum=use_cusum,
            transform_list=transform_list
        )
        feature_list.append(features)
        ids.append(series_id)
    
    feature_df = pd.DataFrame(feature_list, index=ids)
    feature_df.index.name = "id"
    
    # Add break_likelihood placeholder if requested
    # IMPORTANT: Do NOT compute break_likelihood globally here - that would cause leakage in CV
    # Instead, add a NaN placeholder column. The fold-safe computation happens in:
    # - cv_proper.cross_validate_with_rank_norm() (per fold, fit on train)
    # - cv_proper.train_final_model_with_rank_norm() (fit on all training data)
    if use_break_likelihood:
        feature_df['break_likelihood'] = np.nan
    
    return feature_df


def rank_normalize(x: pd.Series) -> pd.Series:
    """
    Rank-based normalization (originality-safe).
    
    Converts values to percentile ranks ∈ [0, 1].
    This is critical for:
    - ROC AUC cares only about ranking
    - Spearman correlation drops dramatically
    - Deterministic by construction
    
    Args:
        x: Input series
        
    Returns:
        Rank-normalized series ∈ [0, 1]
    """
    return x.rank(pct=True)


def rank_normalize_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rank normalization to all feature columns.
    
    CRITICAL: This must be done INSIDE each CV fold to avoid leakage.
    Never rank-normalize using information from the test set.
    
    Args:
        feature_df: DataFrame of raw features
        
    Returns:
        DataFrame with rank-normalized features ∈ [0, 1]
    """
    # Handle NaN values (from small windows in multi-scale)
    # Fill with median rank (0.5) to avoid losing data
    ranked = feature_df.apply(rank_normalize)
    ranked = ranked.fillna(0.5)
    
    return ranked


def aggregate_features(feature_df: pd.DataFrame) -> pd.Series:
    """
    Rank-based aggregation to produce final scores.
    
    This is the Day 1-2 baseline scoring function:
    1. Rank-normalize each feature across all ids
    2. Average the ranks
    
    Args:
        feature_df: DataFrame of raw features
        
    Returns:
        Series of aggregated scores ∈ [0, 1]
    """
    # Rank-normalize each feature column
    ranked = rank_normalize_features(feature_df)
    
    # Average across features
    scores = ranked.mean(axis=1)
    
    return scores


def compute_and_aggregate(
    df: pd.DataFrame,
    use_multiscale: bool = False,
    use_spectral: bool = False,
    use_wavelet: bool = False,
    use_break_likelihood: bool = False,
    use_boundary: bool = False,
    use_boundary_dist: bool = False,
    use_boundary_tail_shape: bool = False,
    use_cv: bool = False,
    use_transforms: bool = False,
    use_compression: bool = False,
    use_cusum: bool = False,
    transform_list=None
) -> pd.Series:
    """
    End-to-end: compute features and produce final scores.
    
    This is the main entry point for the baseline model.
    
    Args:
        df: Raw time series data
        use_multiscale: Whether to use multi-scale features
        use_spectral: Whether to use spectral features
        use_wavelet: Whether to use wavelet features
        use_break_likelihood: Whether to add break_likelihood meta-feature
        use_boundary: Whether to include boundary-localized contrast features
        use_boundary_dist: Whether to include boundary distribution distance features
        use_boundary_tail_shape: Whether to include boundary tail-shape features
        use_cv: Whether to include CV features (Phase 1 - "magic" feature)
        use_transforms: Whether to compute features on transformed series (Phase 1)
        use_compression: Whether to include compression features (Phase 1)
        use_cusum: Whether to include CUSUM features (Phase 1)
        transform_list: List of transforms to use
        
    Returns:
        Series of scores ∈ [0, 1] per id
    """
    features = compute_features(
        df, 
        use_multiscale=use_multiscale, 
        use_spectral=use_spectral, 
        use_wavelet=use_wavelet,
        use_break_likelihood=use_break_likelihood,
        use_boundary=use_boundary,
        use_boundary_dist=use_boundary_dist,
        use_boundary_tail_shape=use_boundary_tail_shape,
        use_cv=use_cv,
        use_transforms=use_transforms,
        use_compression=use_compression,
        use_cusum=use_cusum,
        transform_list=transform_list
    )
    scores = aggregate_features(features)
    return scores
