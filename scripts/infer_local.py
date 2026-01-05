"""
Local inference script with PROPER rank normalization.

This script implements inference matching the training pipeline:
1. Extract raw features from test data
2. Rank-normalize features (using test data statistics only)
3. Predict with trained model
4. Generate predictions ∈ [0, 1]

For baseline mode: Uses simple rank aggregation (no model needed)
For GBM mode: Uses trained LightGBM model with proper ranking
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src and parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sb import config, io, data_loader, features, cv_proper


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
        description="Inference with proper rank normalization"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to test CSV/parquet file (optional if using CrunchDAO format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save predictions (default: predictions.csv)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Path to labels CSV file (optional, for evaluation)"
    )
    parser.add_argument(
        "--crunchdao",
        action="store_true",
        help="Use CrunchDAO dataset format from data/ directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "gbm", "arkf"],
        default="gbm",
        help="Inference mode: baseline (rank aggregation), gbm (LightGBM), or arkf (AR/Kalman)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/trained_model.joblib",
        help="Path to trained GBM model (for --mode gbm)"
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Use multi-scale features (must match training)"
    )
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Use spectral features (must match training)"
    )
    parser.add_argument(
        "--wavelet",
        action="store_true",
        help="Use wavelet features (must match training)"
    )
    parser.add_argument(
        "--break-likelihood",
        action="store_true",
        help="Add break-likelihood as an additional feature (must match training)"
    )
    parser.add_argument(
        "--boundary",
        action="store_true",
        help="Use boundary-localized contrast features (must match training)"
    )
    parser.add_argument(
        "--boundary-dist",
        action="store_true",
        help="Use boundary distribution distance features (Wasserstein/Energy + tail features; must match training)"
    )
    parser.add_argument(
        "--boundary-tail-shape",
        action="store_true",
        help="Use boundary tail-shape features (must match training)"
    )
    
    args = parser.parse_args()
    
    # Set all random seeds for determinism
    np.random.seed(config.RANDOM_SEED)
    
    print("=" * 70)
    print("STRUCTURAL BREAK DETECTION - INFERENCE")
    print("=" * 70)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Multi-scale: {'YES' if args.multiscale else 'NO'}")
    print(f"Spectral: {'YES' if args.spectral else 'NO'}")
    print(f"Wavelet: {'YES' if args.wavelet else 'NO'}")
    print(f"Break Likelihood Feature: {'YES' if args.break_likelihood else 'NO'}")
    print(f"Boundary-localized features: {'YES' if args.boundary else 'NO'}")
    print(f"Boundary distribution distances: {'YES' if args.boundary_dist else 'NO'}")
    print(f"Boundary tail-shape features: {'YES' if args.boundary_tail_shape else 'NO'}")
    
    # Load test data
    if args.crunchdao or (args.data is None):
        print("\nUsing CrunchDAO test dataset from data/ directory...")
        df, y = data_loader.load_for_testing("data", with_labels=True)
        print(f"Loaded {df['id'].nunique():,} time series")
    else:
        print(f"\nLoading test data from {args.data}...")
        df = io.load_data(args.data)
        print(f"Loaded {df['id'].nunique():,} time series")
        y = None
        
        # Load labels if provided
        if args.labels:
            print(f"\nLoading labels from {args.labels}...")
            y_df = pd.read_csv(args.labels)
            y = y_df.set_index("id")["label"]
    
    # Extract features (RAW features, not ranked)
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION")
    print("=" * 70)
    
    if args.mode == "arkf":
        print(f"\nComputing AR/Kalman features...")
        from sb.features.ar_kalman import extract_features
        
        if args.spectral:
            print("Note: --spectral is ignored in arkf mode (arkf uses sb.features.ar_kalman)")
        
        # Load data in proper format
        X_test = io.load_test("data")
        X_raw = extract_features(X_test, window_sizes=[25, 50, 100], n_jobs=4, verbose=True)
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
            use_boundary_tail_shape=args.boundary_tail_shape
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
    
    # NOTE: NaN handling is done by saved imputer in ModelBundle
    # No global fillna here - use train-fitted imputer from bundle
    n_nan = X_raw.isna().sum().sum()
    if n_nan > 0:
        print(f"\nInfo: {n_nan} NaN values found (will use saved imputer from bundle)")
    
    # Run inference based on mode
    if args.mode == "baseline":
        print("\n" + "=" * 70)
        print("MODE: BASELINE (Rank Aggregation)")
        print("=" * 70)
        
        if args.break_likelihood:
            print("\nUsing break-likelihood scoring (hand-selected features)")
            predictions = features.break_likelihood.compute_break_likelihood(
                X_raw, mode="rank_mean"
            )
        else:
            print("\nUsing simple rank aggregation (no ML, no model needed)")
            
            # Baseline just ranks and averages
            predictions = features.base.aggregate_features(X_raw)
        
        # Ensure predictions is a Series with proper index
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions, index=X_raw.index)
        
    elif args.mode == "arkf":
        print("\n" + "=" * 70)
        print("MODE: AR/KALMAN FEATURES WITH LOGISTIC REGRESSION")
        print("=" * 70)
        
        # Load trained model
        import joblib
        model_path = Path(args.model)
        
        if not model_path.exists():
            print(f"\n❌ Error: Model file not found: {args.model}")
            print("Train a model first with:")
            print("  python scripts/train_local.py --mode arkf")
            return
        
        print(f"Loading model bundle from: {args.model}")
        bundle = joblib.load(args.model)
        
        # Check if it's a ModelBundle or legacy model
        if isinstance(bundle, cv_proper.ModelBundle):
            print(f"✅ Loaded ModelBundle (model + rank_normalizer + {len(bundle.feature_columns)} features)")
        else:
            print(f"✅ Loaded model (legacy format - will use independent rank normalization)")
        
        # Rank normalize and predict
        print("\nRank-normalizing test features...")
        predictions = cv_proper.predict_with_rank_norm(
            model_or_bundle=bundle,
            X_test_raw=X_raw
        )
        
        # Convert to Series with proper index
        predictions = pd.Series(predictions, index=X_raw.index)
        
    else:  # gbm
        print("\n" + "=" * 70)
        print("MODE: LIGHTGBM WITH PROPER RANK NORMALIZATION")
        print("=" * 70)
        
        # Load trained model
        import joblib
        model_path = Path(args.model)
        
        if not model_path.exists():
            print(f"\n❌ Error: Model file not found: {args.model}")
            print("Train a model first with:")
            print("  python scripts/train_local.py --mode gbm")
            if args.multiscale:
                print("  python scripts/train_local.py --mode gbm --multiscale")
            return
        
        print(f"Loading model bundle from: {args.model}")
        bundle = joblib.load(args.model)
        
        # Check if it's a ModelBundle or legacy model
        if isinstance(bundle, cv_proper.ModelBundle):
            print(f"✅ Loaded ModelBundle (model + rank_normalizer + {len(bundle.feature_columns)} features)")
        else:
            print(f"✅ Loaded model (legacy format - will use independent rank normalization)")
        
        # Rank normalize and predict
        print("\nRank-normalizing test features...")
        predictions = cv_proper.predict_with_rank_norm(
            model_or_bundle=bundle,
            X_test_raw=X_raw
        )
        
        # Convert to Series with proper index
        predictions = pd.Series(predictions, index=X_raw.index)
    
    # Save predictions
    print("\n" + "=" * 70)
    print("SAVING PREDICTIONS")
    print("=" * 70)
    
    print(f"\nSaving predictions to {args.output}...")
    io.save_predictions(predictions, args.output)
    
    print(f"✅ Predictions saved to: {args.output}")
    print(f"   {len(predictions):,} predictions generated")
    print(f"\nPrediction statistics:")
    print(f"   Mean: {predictions.mean():.4f}")
    print(f"   Std:  {predictions.std():.4f}")
    print(f"   Min:  {predictions.min():.4f}")
    print(f"   Max:  {predictions.max():.4f}")
    
    # Check if predictions look reasonable
    if predictions.std() < 0.01:
        print("\n⚠️  Warning: Very low prediction variance")
        print("   Model may not be discriminating well")
    
    if predictions.mean() < 0.1 or predictions.mean() > 0.9:
        print(f"\n⚠️  Warning: Extreme mean prediction ({predictions.mean():.4f})")
        print("   Model may be biased")
    
    # Optional evaluation
    if y is not None:
        print("\n" + "=" * 70)
        print("EVALUATION (Test Set Performance)")
        print("=" * 70)
        from sb import cv
        cv.print_evaluation_summary(y, predictions)
    
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    
    if args.mode == "gbm":
        print("\nNext steps:")
        print("  1. Check test set AUC (if labels available)")
        print("  2. Compare with CV AUC from training")
        print("  3. If performance is good, submit to platform")


if __name__ == "__main__":
    main()
