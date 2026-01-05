"""
Local training script with PROPER cross-validation.

This script implements:
1. Stratified K-fold CV (splitting by id, not time)
2. Rank normalization INSIDE each fold (no leakage)
3. Heavily regularized LightGBM (to prevent overfitting)
4. Multi-scale features (same features at different windows)
5. Full determinism (reproducible results)

NO in-sample evaluation. Only out-of-sample CV results are reported.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sb import config, data_loader, features, models, cv_proper


def create_model_fn_gbm():
    """
    Create a model factory function for GBM CV.
    
    Returns a function that takes (X_train, y_train) and returns a trained model.
    """
    def train_model(X_train, y_train):
        model = models.gbm.StructuralBreakGBM(
            params=config.LIGHTGBM_PARAMS
        )
        model.train(X_train, y_train)
        return model
    
    return train_model


def create_model_fn_arkf():
    """
    Create a model factory function for AR/Kalman features mode.
    
    Uses heavily regularized logistic regression.
    """
    def train_model(X_train, y_train):
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0],
            cv=3,
            scoring='roc_auc',
            max_iter=500,
            random_state=config.RANDOM_SEED,
            n_jobs=1
        )
        model.fit(X_train, y_train)
        return model
    
    return train_model


def create_model_fn():
    """Legacy wrapper for backward compatibility."""
    return create_model_fn_gbm()


def verify_spectral_columns(X_raw: pd.DataFrame, multiscale: bool, spectral: bool, windows=None):
    """
    Verify spectral features are present when --spectral flag is used.
    
    Args:
        X_raw: Feature DataFrame
        multiscale: Whether multiscale mode is enabled
        spectral: Whether spectral flag is enabled
        windows: List of window sizes (default from config)
    """
    if not spectral:
        return  # Nothing to verify
    
    if windows is None:
        windows = config.MULTI_SCALE_WINDOWS
    
    # Base spectral v1 (6 features)
    expected_base_v1 = {
        'log_low_high_pre', 'log_low_high_post', 'delta_log_low_high',
        'spec_entropy_pre', 'spec_entropy_post', 'delta_spec_entropy'
    }
    
    # Base spectral v2 (6 features, all deltas)
    expected_base_v2 = {
        'delta_peak_ratio', 'delta_flatness', 'delta_flux',
        'delta_rolloff50', 'delta_bandwidth', 'delta_hf_power'
    }
    
    # Windowed spectral (deltas only: 2 from v1 + 6 from v2 = 8 per window)
    expected_windowed_deltas = {
        'delta_log_low_high', 'delta_spec_entropy',  # v1 deltas
        'delta_peak_ratio', 'delta_flatness', 'delta_flux',  # v2 deltas
        'delta_rolloff50', 'delta_bandwidth', 'delta_hf_power'  # v2 deltas
    }
    
    # Check base spectral features (full scale only)
    base_cols = set(X_raw.columns) & (expected_base_v1 | expected_base_v2)
    if len(base_cols) != 12:
        missing = (expected_base_v1 | expected_base_v2) - base_cols
        raise ValueError(
            f"Spectral wiring failed! Expected 12 base spectral features (6 v1 + 6 v2) but found {len(base_cols)}.\n"
            f"Missing: {missing}\n"
            f"Check that spectral_features_all() is properly called in base.py"
        )
    
    # If multiscale, check windowed spectral features (deltas only)
    if multiscale:
        for w in windows:
            expected_windowed = {f"{feat}_w{w}" for feat in expected_windowed_deltas}
            windowed_cols = set(X_raw.columns) & expected_windowed
            if len(windowed_cols) != 8:
                missing = expected_windowed - windowed_cols
                raise ValueError(
                    f"Spectral wiring failed for window {w}! Expected 8 windowed spectral deltas but found {len(windowed_cols)}.\n"
                    f"Missing: {missing}\n"
                    f"Check that spectral_features_deltas_only() is properly called in multiscale.py"
                )
    
    # Success message
    if multiscale:
        total_spectral = 12 + (8 * len(windows))  # base (v1+v2) + windowed deltas
        print(f"✓ Verified {total_spectral} spectral features present (12 base + {8*len(windows)} windowed deltas)")
    else:
        print(f"✓ Verified 12 spectral features present (6 v1 + 6 v2)")


def verify_wavelet_columns(X_raw: pd.DataFrame, multiscale: bool, wavelet: bool, windows=None):
    """
    Verify wavelet features are present when --wavelet flag is used.
    
    Args:
        X_raw: Feature DataFrame
        multiscale: Whether multiscale mode is enabled
        wavelet: Whether wavelet flag is enabled
        windows: List of window sizes (default from config)
    """
    if not wavelet:
        return  # Nothing to verify
    
    if windows is None:
        windows = config.MULTI_SCALE_WINDOWS
    
    # Base wavelet features (12)
    expected_base_wavelet = {
        'wav_entropy_pre', 'wav_entropy_post', 'delta_wav_entropy',
        'wav_low_energy_share_pre', 'wav_low_energy_share_post', 'delta_wav_low_energy_share',
        'wav_high_energy_share_pre', 'wav_high_energy_share_post', 'delta_wav_high_energy_share',
        'delta_wav_energy_l1', 'delta_wav_energy_l2', 'delta_wav_energy_l3'
    }
    
    # Check base wavelet features (full scale only)
    base_cols = set(X_raw.columns) & expected_base_wavelet
    if len(base_cols) != 12:
        missing = expected_base_wavelet - base_cols
        raise ValueError(
            f"Wavelet wiring failed! Expected 12 base wavelet features but found {len(base_cols)}.\n"
            f"Missing: {missing}\n"
            f"Check that wavelet_features() is properly called in base.py or multiscale.py"
        )
    
    # If multiscale, check for one boundary window (largest window)
    if multiscale and len(windows) > 0:
        boundary_window = max(windows)
        expected_windowed = {f"{feat}_w{boundary_window}" for feat in expected_base_wavelet}
        windowed_cols = set(X_raw.columns) & expected_windowed
        if len(windowed_cols) != 12:
            missing = expected_windowed - windowed_cols
            raise ValueError(
                f"Wavelet wiring failed for boundary window {boundary_window}! Expected 12 windowed wavelet features but found {len(windowed_cols)}.\n"
                f"Missing: {missing}\n"
                f"Check that multiscale.py computes wavelet features for boundary window"
            )
    
    # Success message
    if multiscale and len(windows) > 0:
        total_wavelet = 12 + 12  # base + one boundary window
        print(f"✓ Verified {total_wavelet} wavelet features present (12 base + 12 boundary window)")
    else:
        print(f"✓ Verified 12 wavelet features present")


def main():
    parser = argparse.ArgumentParser(
        description="Train with proper CV (no in-sample eval)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training CSV/parquet file (optional if using CrunchDAO format in data/)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "gbm", "arkf"],
        default="gbm",
        help="Training mode: baseline (rank aggregation), gbm (LightGBM), or arkf (AR/Kalman features)"
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Use multi-scale features (same features at different windows)"
    )
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Use spectral features (frequency-domain analysis)"
    )
    parser.add_argument(
        "--wavelet",
        action="store_true",
        help="Use wavelet features (time-frequency analysis)"
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
        help="Use boundary distribution distance features (Wasserstein/Energy + tail-restricted features)"
    )
    parser.add_argument(
        "--boundary-tail-shape",
        action="store_true",
        help="Use boundary tail-shape features (Hill estimator, quantile curvature, asymmetry)"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use Coefficient of Variation features (Phase 1 - 'magic' feature from winners)"
    )
    parser.add_argument(
        "--transforms",
        action="store_true",
        help="Compute features on transformed series: CUMSUM, DIFF, RANK (Phase 1)"
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help="Use compression-based features: Z-lib, Lempel-Ziv complexity (Phase 1)"
    )
    parser.add_argument(
        "--cusum",
        action="store_true",
        help="Use CUSUM features: elbow detection, shape categories (Phase 1)"
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Enable ALL Phase 1 features (CV + transforms + compression + CUSUM)"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="models/trained_model.joblib",
        help="Path to save trained GBM model (for --mode gbm)"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--learned-agg",
        action="store_true",
        help="Use learned feature aggregation instead of rank-mean (leakage-safe)"
    )
    parser.add_argument(
        "--agg-model",
        type=str,
        choices=["lgbm", "logreg"],
        default="lgbm",
        help="Model type for learned aggregation: lgbm (default) or logreg"
    )
    parser.add_argument(
        "--agg-oof-feature",
        action="store_true",
        help="Generate OOF meta-feature 'agg_score' for downstream GBM stacking"
    )
    
    args = parser.parse_args()
    
    # Phase 1 convenience flag
    if args.phase1:
        args.cv = True
        args.transforms = True
        args.compression = True
        args.cusum = True
    
    # Set all random seeds for determinism
    np.random.seed(config.RANDOM_SEED)
    
    print("=" * 70)
    print("STRUCTURAL BREAK DETECTION - PROPER CROSS-VALIDATION")
    print("=" * 70)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Multi-scale: {'YES' if args.multiscale else 'NO'}")
    print(f"Spectral: {'YES' if args.spectral else 'NO'}")
    print(f"Wavelet: {'YES' if args.wavelet else 'NO'}")
    print(f"Break Likelihood Feature: {'YES' if args.break_likelihood else 'NO'}")
    print(f"Boundary-localized features: {'YES' if args.boundary else 'NO'}")
    print(f"Boundary distribution distances: {'YES' if args.boundary_dist else 'NO'}")
    print(f"Boundary tail-shape features: {'YES' if args.boundary_tail_shape else 'NO'}")
    print(f"\n--- Phase 1 Features (Winning Solutions) ---")
    print(f"CV features (magic): {'YES' if args.cv else 'NO'}")
    print(f"Transformations: {'YES' if args.transforms else 'NO'}")
    print(f"Compression: {'YES' if args.compression else 'NO'}")
    print(f"CUSUM: {'YES' if args.cusum else 'NO'}")
    print(f"Boundary tail-shape features: {'YES' if args.boundary_tail_shape else 'NO'}")
    print(f"Learned Aggregation: {'YES' if args.learned_agg else 'NO'}")
    if args.learned_agg:
        print(f"  - Aggregation Model: {args.agg_model.upper()}")
        print(f"  - OOF Feature: {'YES' if args.agg_oof_feature else 'NO'}")
    print(f"CV Folds: {args.n_folds}")
    print(f"Random Seed: {config.RANDOM_SEED}")
    
    # Load data (always use CrunchDAO format if no data path specified)
    if args.data is None:
        print("\nUsing CrunchDAO dataset from data/ directory...")
        df, y = data_loader.load_for_training("data")
    else:
        print(f"\nLoading data from {args.data}...")
        from sb import io
        df = io.load_data(args.data)
        
        if "label" not in df.columns:
            print("Error: No 'label' column found")
            return
        
        y = df.groupby("id")["label"].first()
    
    print(f"Loaded {df['id'].nunique():,} time series")
    print(f"Label distribution: {dict(y.value_counts().sort_index())}")
    print(f"Break rate: {y.mean():.2%}")
    
    # Extract features (RAW features, not ranked)
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION")
    print("=" * 70)
    
    if args.mode == "arkf":
        print(f"\nComputing AR/Kalman features...")
        from sb.features.ar_kalman import extract_features
        from sb import io
        
        if args.spectral:
            print("Note: --spectral is ignored in arkf mode (arkf uses sb.features.ar_kalman)")
        
        # Load data in proper format
        X_train, y = io.load_train("data")
        X_raw = extract_features(X_train, window_sizes=[25, 50, 100], n_jobs=4, verbose=True)
        
    else:
        print(f"\nComputing {'multi-scale' if args.multiscale else 'single-scale'} features...")
        X_raw = features.base.compute_features(
            df, 
            use_multiscale=args.multiscale, 
            use_spectral=args.spectral, 
            use_wavelet=args.wavelet,
            use_break_likelihood=args.break_likelihood,
            use_boundary=args.boundary,
            use_boundary_dist=args.boundary_dist,
            use_boundary_tail_shape=args.boundary_tail_shape,
            use_cv=args.cv,
            use_transforms=args.transforms,
            use_compression=args.compression,
            use_cusum=args.cusum
        )
    
    print(f"Feature shape: {X_raw.shape}")
    print(f"Features: {list(X_raw.columns)[:6]}{'...' if len(X_raw.columns) > 6 else ''}")
    
    # Verify spectral features if requested
    if args.mode != "arkf":
        verify_spectral_columns(X_raw, multiscale=args.multiscale, spectral=args.spectral)
        verify_wavelet_columns(X_raw, multiscale=args.multiscale, wavelet=args.wavelet)
    
    # Verify break_likelihood feature if requested
    if args.break_likelihood:
        if 'break_likelihood' in X_raw.columns:
            bl_scores = X_raw['break_likelihood']
            print(f"\n✓ break_likelihood feature present: range [{bl_scores.min():.3f}, {bl_scores.max():.3f}]")
        else:
            print("\n⚠️  Warning: --break-likelihood flag set but feature not found!")
    
    # NOTE: NaN handling now happens fold-safely in cv_proper.py via MedianImputer
    # No global fillna here - that would cause leakage in CV
    n_nan = X_raw.isna().sum().sum()
    if n_nan > 0:
        print(f"\nInfo: {n_nan} NaN values found (will be imputed fold-safely in CV)")
    
    # Run based on mode
    if args.mode == "baseline":
        print("\n" + "=" * 70)
        print("MODE: BASELINE (Rank Aggregation)")
        print("=" * 70)
        
        if args.learned_agg:
            print("\nUsing LEARNED aggregation (fold-safe)")
            print(f"Model type: {args.agg_model.upper()}")
            print("This replaces simple rank-mean with a learned model.")
            print("\nPerforming cross-validation...")
            
            # Import learned aggregation components
            from sb.models.learned_agg import AggregatorConfig
            
            # Create config
            agg_config = AggregatorConfig(
                model_type=args.agg_model,
                max_features=300,
                correlation_threshold=0.98,
                random_state=config.RANDOM_SEED
            )
            
            # Run CV with learned aggregation
            mean_auc, std_auc, fold_aucs, _ = cv_proper.cross_validate_with_learned_agg(
                X_raw=X_raw,
                y=y,
                agg_config=agg_config,
                n_splits=args.n_folds,
                random_state=config.RANDOM_SEED,
                verbose=True,
                return_oof_scores=False
            )
            
            print(f"\nLearned Aggregation CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            
            # Train final aggregator on all data
            print("\nTraining final aggregator on all data...")
            final_agg = cv_proper.train_final_learned_agg(
                X_raw=X_raw,
                y=y,
                agg_config=agg_config,
                verbose=True
            )
            
            # Save aggregator
            import joblib
            agg_path = Path("models/learned_agg.joblib")
            agg_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_agg, agg_path)
            print(f"\n✅ Learned aggregator saved to: {agg_path}")
            
            # Show feature importance if available
            importance = final_agg.get_feature_importance()
            if importance is not None:
                print("\nTop 10 features:")
                for feat, imp in importance.head(10).items():
                    print(f"  {feat:30s}: {imp:.4f}")
        
        elif args.break_likelihood:
            print("\nUsing break-likelihood scoring (hand-selected features)")
            print("Features used: delta_rmse, delta_resid_var, energy, delta_ar1_phi, etc.")
            
            # Use break-likelihood scorer
            scores = features.break_likelihood.compute_break_likelihood(
                X_raw, mode="rank_mean"
            )
        else:
            print("\nUsing simple rank aggregation (no ML, no CV needed)")
            print("This is the Day 1-2 baseline for comparison.")
            
            # Baseline just ranks and averages
            scores = features.base.aggregate_features(X_raw)
        
        # Evaluate on full dataset (baseline is deterministic, no overfitting risk)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, scores)
        
        print(f"\nBaseline ROC AUC: {auc:.4f}")
        print("\nNote: This is in-sample AUC (ok for baseline since no fitting occurs)")
        
    elif args.mode == "arkf":
        print("\n" + "=" * 70)
        print("MODE: AR/KALMAN FEATURES WITH LOGISTIC REGRESSION")
        print("=" * 70)
        print("\nUsing heavily regularized logistic regression on AR/Kalman features")
        
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION (Rank normalization inside folds)")
        print("=" * 70)
        print("\nRunning stratified K-fold CV...")
        print("Each fold:")
        print("  1. Split by id (stratified by label)")
        print("  2. Impute NaNs using train fold median")
        print("  3. Rank-normalize using train fold distribution")
        print("  4. Train LogisticRegressionCV")
        print("  5. Predict on ranked val")
        print("  6. Compute AUC\n")
        
        model_fn = create_model_fn_arkf()
        
        mean_auc, std_auc, fold_aucs = cv_proper.cross_validate_with_rank_norm(
            X_raw=X_raw,
            y=y,
            model_fn=model_fn,
            n_splits=args.n_folds,
            random_state=config.RANDOM_SEED,
            verbose=True,
            recompute_break_likelihood=args.break_likelihood
        )
        
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 70)
        print(f"\nOut-of-sample ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Fold AUCs: {[f'{x:.4f}' for x in fold_aucs]}")
        
        # Train final model on ALL data for deployment
        print("\n" + "=" * 70)
        print("TRAINING FINAL MODEL (on all data)")
        print("=" * 70)
        
        bundle = cv_proper.train_final_model_with_rank_norm(
            X_raw=X_raw,
            y=y,
            model_fn=model_fn,
            fit_break_likelihood=args.break_likelihood
        )
        
        # Save bundle (model + rank_normalizer + feature_columns)
        import joblib
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.save_model)
        
        print(f"\n✅ Model bundle saved to: {args.save_model}")
        print("   Bundle contains: model, rank_normalizer, feature_columns")
        print("   This ensures inference uses the SAME rank transformation as training.")
        
    else:  # gbm mode
        print("\n" + "=" * 70)
        print("MODE: LIGHTGBM WITH PROPER CROSS-VALIDATION")
        print("=" * 70)
        print("\nLightGBM parameters (heavily regularized):")
        for k, v in config.LIGHTGBM_PARAMS.items():
            if k != "verbose":
                print(f"  {k:20s}: {v}")
        
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION (Rank normalization inside folds)")
        print("=" * 70)
        
        model_fn = create_model_fn_gbm()
        
        # Check if we should use learned aggregation as OOF feature
        if args.agg_oof_feature:
            print("\nTwo-stage training with learned aggregation OOF feature:")
            print("  STAGE 1: Generate OOF 'agg_score' from all features")
            print("  STAGE 2: Train GBM with 'agg_score' as additional feature\n")
            
            # Import learned aggregation components
            from sb.models.learned_agg import AggregatorConfig
            
            # Create config for aggregator
            agg_config = AggregatorConfig(
                model_type=args.agg_model,
                max_features=300,
                correlation_threshold=0.98,
                random_state=config.RANDOM_SEED
            )
            
            # Run two-stage CV
            mean_auc, std_auc, fold_aucs, oof_agg_scores = cv_proper.cross_validate_with_learned_agg_feature(
                X_raw=X_raw,
                y=y,
                model_fn=model_fn,
                agg_config=agg_config,
                n_splits=args.n_folds,
                random_state=config.RANDOM_SEED,
                verbose=True
            )
            
            # For final training, we need to:
            # 1. Train final aggregator on all data
            # 2. Generate agg_score on all data
            # 3. Train final GBM with agg_score included
            
            print("\n" + "=" * 70)
            print("TRAINING FINAL MODELS (on all data)")
            print("=" * 70)
            
            # Train final aggregator
            final_agg = cv_proper.train_final_learned_agg(
                X_raw=X_raw,
                y=y,
                agg_config=agg_config,
                verbose=True
            )
            
            # Generate agg_score on all data
            final_agg_scores = final_agg.predict_proba(X_raw)[:, 1]
            
            # Add to features
            X_augmented = X_raw.copy()
            X_augmented['meta_agg_score'] = final_agg_scores
            
            # Train final GBM with augmented features
            bundle = cv_proper.train_final_model_with_rank_norm(
                X_raw=X_augmented,
                y=y,
                model_fn=model_fn,
                fit_break_likelihood=args.break_likelihood
            )
            
            # Save aggregator separately
            import joblib
            agg_path = Path("models/learned_agg.joblib")
            agg_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_agg, agg_path)
            print(f"\n✅ Learned aggregator saved to: {agg_path}")
            
        else:
            # Standard single-stage CV
            print("\nRunning stratified K-fold CV...")
            print("Each fold:")
            print("  1. Split by id (stratified by label)")
            print("  2. Rank-normalize train features (using only train data)")
            print("  3. Rank-normalize val features (using only val data)")
            print("  4. Train LightGBM on ranked train")
            print("  5. Predict on ranked val")
            print("  6. Compute AUC\n")
            
            mean_auc, std_auc, fold_aucs = cv_proper.cross_validate_with_rank_norm(
                X_raw=X_raw,
                y=y,
                model_fn=model_fn,
                n_splits=args.n_folds,
                random_state=config.RANDOM_SEED,
                verbose=True,
                recompute_break_likelihood=args.break_likelihood
            )
            
            # Train final model on ALL data for deployment
            print("\n" + "=" * 70)
            print("TRAINING FINAL MODEL (on all data)")
            print("=" * 70)
            
            bundle = cv_proper.train_final_model_with_rank_norm(
                X_raw=X_raw,
                y=y,
                model_fn=model_fn,
                fit_break_likelihood=args.break_likelihood
            )
        
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 70)
        print(f"\nOut-of-sample ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Fold AUCs: {[f'{x:.4f}' for x in fold_aucs]}")
        
        if std_auc > 0.03:
            print(f"\n⚠️  Warning: High CV std ({std_auc:.4f}) suggests instability")
            print("   Consider: more regularization or more training data")
        
        # Train final model on ALL data for deployment
        print("\n" + "=" * 70)
        print("TRAINING FINAL MODEL (on all data)")
        print("=" * 70)
        
        bundle = cv_proper.train_final_model_with_rank_norm(
            X_raw=X_raw,
            y=y,
            model_fn=model_fn,
            fit_break_likelihood=args.break_likelihood
        )
        
        print("\nFeature importances (top 10):")
        importances = bundle.model.get_feature_importance()
        for feat, imp in importances.head(10).items():
            print(f"  {feat:30s}: {imp:.3f}")
        
        # Save bundle (model + rank_normalizer + feature_columns)
        import joblib
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.save_model)
        
        print(f"\n✅ Model bundle saved to: {args.save_model}")
        print("   Bundle contains: model, rank_normalizer, feature_columns")
        print("   This ensures inference uses the SAME rank transformation as training.")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    if args.mode == "gbm":
        print(f"\n✅ Out-of-sample CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"✅ Model saved to: {args.save_model}")
        print("\nNext steps:")
        print(f"  1. Test on held-out set: python scripts/infer_local.py --mode gbm")
        print("  2. Compare with baseline to verify improvement")
        print("  3. If CV AUC is good (>0.80), submit to platform")
        
        if args.multiscale:
            print(f"\nUsing {len(config.MULTI_SCALE_WINDOWS) + 1} scales:")
            print(f"  - Full segments (baseline)")
            for w in config.MULTI_SCALE_WINDOWS:
                print(f"  - Last/first {w} points (boundary-focused)")
    else:
        print(f"\nBaseline AUC: {auc:.4f}")
        print("\nNext steps:")
        print("  1. Try GBM mode: python scripts/train_local.py --mode gbm")
        print("  2. Try multi-scale: python scripts/train_local.py --mode gbm --multiscale")


if __name__ == "__main__":
    main()
