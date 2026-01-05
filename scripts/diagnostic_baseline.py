"""
Pure statistical baseline diagnostic.

No ML model - just rank aggregation with proper CV.
Goal: Determine if features contain signal or if problem is feature choice.

If this baseline gets ~0.55 AUC:
  → Problem is FEATURES (not GBM, not regularization, not CV)
  
If this baseline gets ~0.75+ AUC:
  → Features are good, problem is modeling/overfitting
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sb import config, data_loader, features
from sb.features import break_likelihood
from sb.cv_proper import RankNormalizer  # Import for fold-safe ranking


def rank_normalize_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Rank-normalize each feature to [0, 1]."""
    X_ranked = X_raw.rank(pct=True, method='average')
    X_ranked = X_ranked.fillna(0.5)
    return X_ranked


def score_feature_auc_train_only(X_train_ranked: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    """
    Compute direction-invariant ROC AUC per feature using TRAIN ONLY.
    Returns a Series indexed by feature name, sorted descending.
    """
    aucs = {}
    for col in X_train_ranked.columns:
        x = X_train_ranked[col]
        if x.isna().all() or x.nunique(dropna=True) <= 1:
            aucs[col] = 0.5
            continue
        try:
            a1 = roc_auc_score(y_train, x)
            a2 = roc_auc_score(y_train, -x)
            aucs[col] = max(a1, a2)
        except Exception:
            aucs[col] = 0.5
    return pd.Series(aucs).sort_values(ascending=False)


def aggregate_features(X_ranked: pd.DataFrame) -> pd.Series:
    """Simple mean aggregation."""
    return X_ranked.mean(axis=1)


def compute_feature_diagnostics(X_raw: pd.DataFrame, y: pd.Series):
    """
    Compute per-feature diagnostics:
    - Individual ROC AUC
    - Correlation with label
    """
    print("\n" + "=" * 70)
    print("PER-FEATURE DIAGNOSTICS")
    print("=" * 70)
    
    diagnostics = []
    
    for col in X_raw.columns:
        feature_values = X_raw[col].values
        
        # Skip if all NaN
        if np.isnan(feature_values).all():
            continue
        
        # Fill NaN for AUC computation
        feature_clean = pd.Series(feature_values, index=X_raw.index).fillna(
            np.nanmedian(feature_values)
        )
        
        # Compute AUC (try both directions)
        try:
            auc = roc_auc_score(y, feature_clean)
            auc_flipped = roc_auc_score(y, -feature_clean)
            auc_best = max(auc, auc_flipped)
        except:
            auc_best = 0.5
        
        # Correlation with label
        corr = np.corrcoef(feature_clean, y)[0, 1]
        
        diagnostics.append({
            'feature': col,
            'auc': auc_best,
            'corr': abs(corr),
            'nan_pct': np.isnan(feature_values).mean() * 100
        })
    
    df_diag = pd.DataFrame(diagnostics).sort_values('auc', ascending=False)
    
    print(f"\nTop 20 features by individual AUC:")
    print(df_diag.head(20).to_string(index=False))
    
    print(f"\nBottom 10 features by individual AUC:")
    print(df_diag.tail(10).to_string(index=False))
    
    print(f"\n" + "=" * 70)
    print(f"Feature AUC statistics:")
    print(f"  Mean:   {df_diag['auc'].mean():.4f}")
    print(f"  Median: {df_diag['auc'].median():.4f}")
    print(f"  Std:    {df_diag['auc'].std():.4f}")
    print(f"  Max:    {df_diag['auc'].max():.4f}")
    print(f"  Min:    {df_diag['auc'].min():.4f}")
    
    # Count features with signal (AUC > 0.55)
    n_signal = (df_diag['auc'] > 0.55).sum()
    print(f"\nFeatures with signal (AUC > 0.55): {n_signal}/{len(df_diag)} ({n_signal/len(df_diag)*100:.1f}%)")
    
    return df_diag


def run_statistical_baseline(X_raw: pd.DataFrame, y: pd.Series, 
                             n_splits: int = 5, multiscale: bool = False,
                             use_break_likelihood_feature: bool = False,
                             topk: int = 0):
    """
    Run pure statistical baseline with proper CV.
    
    Args:
        X_raw: Raw features (not ranked)
        y: Labels
        n_splits: Number of CV folds
        multiscale: Whether multiscale features were used
        use_break_likelihood_feature: Whether to add break_likelihood as a feature
        topk: If >0, select top-K features per fold using train-only AUC
    """
    print("\n" + "=" * 70)
    print("STATISTICAL BASELINE - RANK AGGREGATION")
    print("=" * 70)
    print(f"\nFeature count: {X_raw.shape[1]}")
    print(f"Sample count: {X_raw.shape[0]}")
    print(f"Multi-scale: {'YES' if multiscale else 'NO'}")
    print(f"Break-likelihood as feature: {'YES' if use_break_likelihood_feature else 'NO'}")
    print(f"Top-K selection: {topk if topk > 0 else 'OFF'}")
    print(f"CV folds: {n_splits}")
    
    # Stratified K-fold
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=config.SHUFFLE_CV,
        random_state=config.RANDOM_SEED
    )
    
    fold_aucs = []
    
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION (Proper - Rank Inside Folds)")
    print("=" * 70)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        # Split data
        X_train_raw = X_raw.iloc[train_idx].copy()
        X_val_raw = X_raw.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        # If break_likelihood is enabled, recompute it fold-safely
        if use_break_likelihood_feature and 'break_likelihood' in X_raw.columns:
            # Remove existing break_likelihood (will recompute)
            X_train_base = X_train_raw.drop(columns=['break_likelihood'])
            X_val_base = X_val_raw.drop(columns=['break_likelihood'])
            
            # Fit scorer on train
            bl_scorer = break_likelihood.fit_break_likelihood_scorer(
                X_train_base, mode="rank_mean"
            )
            
            # Transform train and val
            bl_train = bl_scorer.transform(X_train_base)
            bl_val = bl_scorer.transform(X_val_base)
            
            # Add back to feature matrices
            X_train_raw['break_likelihood'] = bl_train
            X_val_raw['break_likelihood'] = bl_val
        
        # FOLD-SAFE IMPUTATION: Fit on train, apply to both train and val
        from sb.cv_proper import MedianImputer
        imputer = MedianImputer()
        X_train_filled = imputer.fit_transform(X_train_raw)
        X_val_filled = imputer.transform(X_val_raw)
        
        # CRITICAL FIX: Rank-normalize using TRAIN-FITTED transform (no leakage)
        # Fit RankNormalizer on train, apply to both train and val
        rn = RankNormalizer().fit(X_train_filled)
        X_train_ranked = rn.transform(X_train_filled)
        X_val_ranked = rn.transform(X_val_filled)
        
        # Aggregate via mean (with optional Top-K feature selection)
        if topk and topk > 0:
            # FOLD-SAFE TOP-K: Select features using TRAIN-ONLY AUC
            feature_auc = score_feature_auc_train_only(X_train_ranked, y_train)
            k = min(topk, len(feature_auc))
            topk_cols = feature_auc.index[:k].tolist()
            
            train_scores = X_train_ranked[topk_cols].mean(axis=1)
            val_scores = X_val_ranked[topk_cols].mean(axis=1)
            
            if fold == 1:
                print("\nTop-K features (fold 1, TRAIN-only):")
                print(feature_auc.head(k).to_string())
        else:
            # Default: aggregate all features
            train_scores = aggregate_features(X_train_ranked)
            val_scores = aggregate_features(X_val_ranked)
        
        # Compute AUC on validation
        val_auc = roc_auc_score(y_val, val_scores)
        fold_aucs.append(val_auc)
        
        print(f"Fold {fold}: Val AUC = {val_auc:.4f}")
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)
    print(f"\nOut-of-sample AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Fold AUCs: {[f'{x:.4f}' for x in fold_aucs]}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if mean_auc < 0.60:
        print("\n❌ POOR BASELINE (AUC < 0.60)")
        print("   → Features contain WEAK signal")
        print("   → Problem is FEATURE CHOICE, not modeling")
        print("   → Need better features or different approach")
        print("\n   Recommendations:")
        print("   1. Check per-feature AUCs above")
        print("   2. Remove features with AUC ≈ 0.50")
        print("   3. Try different feature families")
        print("   4. Check data quality/preprocessing")
        
    elif mean_auc < 0.70:
        print("\n⚠️  WEAK BASELINE (0.60 ≤ AUC < 0.70)")
        print("   → Features contain MODERATE signal")
        print("   → GBM might help, but gains will be limited")
        print("   → Consider feature engineering")
        
    elif mean_auc < 0.80:
        print("\n✅ GOOD BASELINE (0.70 ≤ AUC < 0.80)")
        print("   → Features contain GOOD signal")
        print("   → GBM should improve performance")
        print("   → Focus on proper regularization")
        
    else:
        print("\n🎉 EXCELLENT BASELINE (AUC ≥ 0.80)")
        print("   → Features contain STRONG signal")
        print("   → Simple aggregation already works well")
        print("   → GBM may provide small improvements")
    
    if std_auc > 0.05:
        print(f"\n⚠️  High CV std ({std_auc:.4f}) suggests:")
        print("   - Feature instability across folds")
        print("   - Small sample size in some folds")
        print("   - Need more robust features")
    
    return mean_auc, std_auc, fold_aucs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diagnostic: Pure statistical baseline with proper CV"
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Use multi-scale features"
    )
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Use spectral features"
    )
    parser.add_argument(
        "--wavelet",
        action="store_true",
        help="Use wavelet features"
    )
    parser.add_argument(
        "--break-likelihood",
        action="store_true",
        help="Add break-likelihood as an additional feature (aggregated from hand-selected features)"
    )
    parser.add_argument(
        "--boundary",
        action="store_true",
        help="Use boundary-localized contrast features"
    )
    parser.add_argument(
        "--boundary-dist",
        action="store_true",
        help="Use boundary distribution distance features (Wasserstein/Energy + 180 tail-restricted features)"
    )
    parser.add_argument(
        "--boundary-tail-shape",
        action="store_true",
        help="Use boundary tail-shape features (Hill estimator, quantile curvature, asymmetry)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="If >0, select top-K features per fold using TRAIN-only AUC, then mean-aggregate only those features."
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--learned-agg",
        action="store_true",
        help="Use learned aggregation instead of rank-mean"
    )
    parser.add_argument(
        "--agg-model",
        type=str,
        choices=["lgbm", "logreg"],
        default="lgbm",
        help="Model type for learned aggregation (default: lgbm)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(config.RANDOM_SEED)
    
    print("=" * 70)
    print("DIAGNOSTIC: STATISTICAL BASELINE")
    print("=" * 70)
    print("\nGoal: Determine if features contain signal")
    print("Method: Rank aggregation with proper CV (no ML)" if not args.learned_agg else "Method: Learned aggregation with proper CV")
    print(f"Learned Aggregation: {'YES (' + args.agg_model.upper() + ')' if args.learned_agg else 'NO'}")
    print(f"Multiscale: {'YES' if args.multiscale else 'NO'}")
    print(f"Spectral: {'YES' if args.spectral else 'NO'}")
    print(f"Wavelet: {'YES' if args.wavelet else 'NO'}")
    print(f"Break-likelihood as feature: {'YES' if args.break_likelihood else 'NO'}")
    print(f"Boundary-localized features: {'YES' if args.boundary else 'NO'}")
    print(f"Boundary distribution distances: {'YES' if args.boundary_dist else 'NO'}")
    print(f"Boundary tail-shape features: {'YES' if args.boundary_tail_shape else 'NO'}")
    
    # Load data
    print("\nLoading CrunchDAO dataset...")
    df, y = data_loader.load_for_training("data")
    print(f"Loaded {df['id'].nunique():,} time series")
    print(f"Break rate: {y.mean():.2%}")
    
    # Extract features
    print(f"\nExtracting features...")
    X_raw = features.base.compute_features(
        df, 
        use_multiscale=args.multiscale, 
        use_spectral=args.spectral, 
        use_wavelet=args.wavelet,
        use_break_likelihood=args.break_likelihood,
        use_boundary=args.boundary,
        use_boundary_dist=args.boundary_dist,
        use_boundary_tail_shape=args.boundary_tail_shape
    )
    
    # NOTE: NaN handling now happens fold-safely in CV loop via MedianImputer
    # No global fillna here - that would cause leakage
    n_nan = X_raw.isna().sum().sum()
    if n_nan > 0:
        print(f"Info: {n_nan} NaN values found (will be imputed fold-safely in CV)")
    
    print(f"Feature shape: {X_raw.shape}")
    
    # Print boundary_dist feature diagnostics if enabled
    if args.boundary_dist:
        bl_tail_cols = [c for c in X_raw.columns if c.startswith('bl_tail_')]
        bl_dist_cols = [c for c in X_raw.columns if c.startswith('bl_') and not c.startswith('bl_tail_') and not c.startswith('bl_ts_')]
        if bl_tail_cols:
            n_tail = len(bl_tail_cols)
            nan_pct = X_raw[bl_tail_cols].isna().mean().mean() * 100
            print(f"  → {n_tail} tail-restricted features (bl_tail_*), {nan_pct:.1f}% NaN avg")
        if bl_dist_cols:
            n_dist = len(bl_dist_cols)
            nan_pct = X_raw[bl_dist_cols].isna().mean().mean() * 100
            print(f"  → {n_dist} base distribution distance features (bl_*), {nan_pct:.1f}% NaN avg")
    
    # Print boundary_tail_shape feature diagnostics if enabled
    if args.boundary_tail_shape:
        bl_ts_cols = [c for c in X_raw.columns if c.startswith('bl_ts_')]
        bl_ts_loc_cols = [c for c in bl_ts_cols if '_loc_' in c]
        bl_ts_dod_cols = [c for c in bl_ts_cols if '_dod_' in c]
        if bl_ts_cols:
            n_ts = len(bl_ts_cols)
            n_loc = len(bl_ts_loc_cols)
            n_dod = len(bl_ts_dod_cols)
            nan_pct = X_raw[bl_ts_cols].isna().mean().mean() * 100
            print(f"  → {n_ts} tail-shape features (bl_ts_*), {nan_pct:.1f}% NaN avg")
            print(f"    • {n_loc} localization features (window vs full)")
            print(f"    • {n_dod} DoD features (statistics across windows)")
    
    # Verify break_likelihood feature if requested
    if args.break_likelihood:
        if 'break_likelihood' in X_raw.columns:
            bl_scores = X_raw['break_likelihood']
            print(f"✓ break_likelihood feature present: range [{bl_scores.min():.3f}, {bl_scores.max():.3f}]")
        else:
            print("⚠️  Warning: --break-likelihood flag set but feature not found!")
    
    # Per-feature diagnostics
    if not args.break_likelihood:
        df_diag = compute_feature_diagnostics(X_raw, y)
    else:
        # Print break-likelihood feature diagnostics instead
        break_likelihood.print_feature_diagnostics(X_raw, y)
        df_diag = None
    
    # Run statistical baseline with proper CV
    if args.learned_agg:
        # Use learned aggregation
        print("\n" + "=" * 70)
        print("LEARNED AGGREGATION MODE")
        print("=" * 70)
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from sb.models.learned_agg import AggregatorConfig
        from sb import cv_proper
        
        agg_config = AggregatorConfig(
            model_type=args.agg_model,
            max_features=300,
            correlation_threshold=0.98,
            random_state=config.RANDOM_SEED
        )
        
        mean_auc, std_auc, fold_aucs, _ = cv_proper.cross_validate_with_learned_agg(
            X_raw=X_raw,
            y=y,
            agg_config=agg_config,
            n_splits=args.n_folds,
            random_state=config.RANDOM_SEED,
            verbose=True,
            return_oof_scores=False
        )
        
        print("\n" + "=" * 70)
        print("LEARNED AGGREGATION RESULTS")
        print("=" * 70)
        print(f"\nOut-of-sample AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Fold AUCs: {[f'{x:.4f}' for x in fold_aucs]}")
    else:
        # Use rank-mean aggregation
        mean_auc, std_auc, fold_aucs = run_statistical_baseline(
            X_raw, y, 
            n_splits=args.n_folds,
            multiscale=args.multiscale,
            use_break_likelihood_feature=args.break_likelihood,
            topk=args.topk
        )
    
    # Final summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(f"\nStatistical Baseline AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    if df_diag is not None:
        print(f"Best single feature AUC:  {df_diag['auc'].max():.4f}")
        print(f"Mean single feature AUC:  {df_diag['auc'].mean():.4f}")
    
    print("\nNext steps:")
    if mean_auc < 0.60:
        print("  1. ❌ Current features are weak - need better feature engineering")
        print("  2. Review per-feature AUCs above")
        print("  3. Try --break-likelihood flag to use hand-selected features")
        print("  4. Check if data preprocessing is correct")
    elif mean_auc < 0.75:
        print("  1. ⚠️  Features have moderate signal")
        print("  2. Try GBM with heavy regularization")
        print("  3. Try --break-likelihood flag for focused feature set")
    else:
        print("  1. ✅ Features are good!")
        print("  2. Try GBM for additional gains")
        flags = []
        if args.multiscale:
            flags.append('--multiscale')
        if args.spectral:
            flags.append('--spectral')
        if args.wavelet:
            flags.append('--wavelet')
        if args.boundary:
            flags.append('--boundary')
        if args.boundary_dist:
            flags.append('--boundary-dist')
        flag_str = ' '.join(flags) if flags else ''
        print(f"  3. Run: python scripts/train_local.py --mode gbm {flag_str}")


if __name__ == "__main__":
    main()
