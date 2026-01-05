"""
Run AR/Kalman feature diagnostics with leakage-safe cross-validation.

This script:
1. Loads training data
2. Computes AR/Kalman features (leakage-safe by design)
3. Runs stratified K-fold CV with proper fold-wise transformations
4. Reports per-feature AUCs and overall baseline AUC
5. Saves results to CSV

Run: python scripts/run_ar_kalman_diagnostics.py
     python scripts/run_ar_kalman_diagnostics.py --window-sizes 25 50 100 --cv-folds 5 --jobs 4
     python scripts/run_ar_kalman_diagnostics.py --fast --no-parallel
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sb import io
from sb.features.ar_kalman import extract_features
from sb.eval.cv_diagnostics import run_cv_diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="AR/Kalman Feature Diagnostics - Leakage-Safe Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing X_train.parquet and y_train.parquet (default: data)'
    )
    parser.add_argument(
        '--window-sizes',
        type=int,
        nargs='+',
        default=[25, 50, 100],
        help='Boundary window sizes (default: 25 50 100)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=4,
        help='Number of parallel jobs for feature extraction (default: 4, use 1 for strict determinism)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Force single-threaded execution (strict determinism)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: subsample to 1000 series for quick iteration'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory for output files (default: results)'
    )
    
    args = parser.parse_args()
    
    # Override jobs if no-parallel
    if args.no_parallel:
        args.jobs = 1
    
    # Fixed random seed for reproducibility
    np.random.seed(42)
    
    print("=" * 70)
    print("AR/KALMAN DIAGNOSTICS - LEAKAGE-SAFE PIPELINE")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Window sizes: {args.window_sizes}")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Parallel jobs: {args.jobs}")
    print(f"  Fast mode: {args.fast}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: 42")
    
    # Step 1: Load data
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)
    
    X_train, y_train = io.load_train(args.data_dir)
    
    # Handle MultiIndex format
    if isinstance(X_train.index, pd.MultiIndex):
        n_series = X_train.index.get_level_values(0).nunique()
    else:
        n_series = X_train['id'].nunique()
    
    print(f"Loaded {len(X_train):,} observations across {n_series:,} series")
    print(f"Break rate: {y_train.mean():.2%}")
    
    # Fast mode: subsample
    if args.fast:
        unique_ids = X_train.index.get_level_values(0).unique()[:1000]
        X_train = X_train.loc[unique_ids]
        y_train = y_train.loc[unique_ids]
        n_series = len(unique_ids)
        print(f"\n⚡ FAST MODE: Subsampled to {n_series} series")
    
    # Step 2: Compute features
    print("\n" + "=" * 70)
    print("STEP 2: COMPUTE AR/KALMAN FEATURES")
    print("=" * 70)
    print("\n⚠️  ANTI-LEAKAGE: All preprocessing uses ONLY pre-segment statistics")
    print("   - Winsorization: quantiles from PRE only")
    print("   - Standardization: mean/std from PRE only")
    print("   - AR/Kalman params: estimated on PRE only")
    print("   - Cross-prediction: PRE model applied to POST")
    
    t_start = time.time()
    features_df = extract_features(
        X_train,
        window_sizes=args.window_sizes,
        n_jobs=args.jobs,
        verbose=True
    )
    t_elapsed = time.time() - t_start
    
    print(f"\n✅ Features computed: {features_df.shape}")
    print(f"   Samples: {features_df.shape[0]:,}")
    print(f"   Features: {features_df.shape[1]}")
    print(f"   Time: {t_elapsed:.1f}s ({t_elapsed / n_series * 1000:.1f}ms per series)")
    print(f"   Throughput: {n_series / t_elapsed:.0f} series/sec")
    
    # Step 2.5: Determinism check
    print("\n" + "=" * 70)
    print("STEP 2.5: DETERMINISM CHECK")
    print("=" * 70)
    
    # Select random subset for recomputation
    test_ids = features_df.index[:min(100, len(features_df))]
    X_test_subset = X_train.loc[test_ids] if isinstance(X_train.index, pd.MultiIndex) else X_train[X_train.index.get_level_values(0).isin(test_ids)]
    
    print(f"Recomputing features for {len(test_ids)} series...")
    features_df2 = extract_features(
        X_test_subset,
        window_sizes=args.window_sizes,
        n_jobs=1,
        verbose=False
    )
    
    # Compare
    common_cols = features_df.columns.intersection(features_df2.columns)
    diff = (features_df.loc[test_ids, common_cols] - features_df2.loc[test_ids, common_cols]).abs()
    max_diff = diff.max().max()
    
    print(f"Max absolute difference: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✅ DETERMINISM VERIFIED: Features are deterministic!")
    else:
        print(f"⚠️  WARNING: Non-deterministic behavior detected (max diff = {max_diff:.2e})")
        print(f"   This may be due to parallel processing or numerical precision")
    print(f"   Features: {features_df.shape[1]}")
    
    # Check for NaNs
    nan_counts = features_df.isna().sum()
    n_features_with_nans = (nan_counts > 0).sum()
    if n_features_with_nans > 0:
        print(f"\n⚠️  {n_features_with_nans} features have NaN values (will be imputed in CV)")
        print(f"   Total NaNs: {nan_counts.sum():,}")
    
    # Step 3: Run CV diagnostics
    print("\n" + "=" * 70)
    print("STEP 3: CROSS-VALIDATION DIAGNOSTICS")
    print("=" * 70)
    print("\n⚠️  ANTI-LEAKAGE: All fold-wise transformations fit on TRAIN only")
    print("   - NaN imputation: medians from TRAIN fold")
    print("   - Rank normalization: percentiles from TRAIN fold")
    print("   - No global statistics computed on full dataset!")
    
    # Run with rank normalization
    results_rank = run_cv_diagnostics(
        X=features_df,
        y=y_train,
        n_splits=args.cv_folds,
        random_state=42,
        use_rank_normalization=True,
        aggregation='rank_mean',
        verbose=True
    )
    
    # Optionally run with logistic regression
    print("\n" + "=" * 70)
    print("BONUS: LOGISTIC REGRESSION BASELINE")
    print("=" * 70)
    
    results_logistic = run_cv_diagnostics(
        X=features_df,
        y=y_train,
        n_splits=args.cv_folds,
        random_state=42,
        use_rank_normalization=True,
        aggregation='logistic',
        verbose=False
    )
    
    print(f"\nLogistic regression AUC: {results_logistic['mean_auc']:.4f} ± {results_logistic['std_auc']:.4f}")
    
    # Step 4: Save results
    print("\n" + "=" * 70)
    print("STEP 4: SAVE RESULTS")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save per-feature AUCs
    feature_auc_summary = pd.DataFrame({
        'feature': results_rank['per_feature_aucs'].index,
        'mean_auc': results_rank['per_feature_aucs'].values,
        'auc_std': results_rank['per_feature_auc_df'].std().values
    }).sort_values('mean_auc', ascending=False)
    
    feature_auc_file = output_dir / "ar_kalman_feature_aucs.csv"
    feature_auc_summary.to_csv(feature_auc_file, index=False)
    print(f"\n✅ Saved feature AUCs to: {feature_auc_file}")
    
    # Save CV summary
    cv_summary_data = {
        'method': ['rank_mean', 'logistic'],
        'mean_auc': [results_rank['mean_auc'], results_logistic['mean_auc']],
        'std_auc': [results_rank['std_auc'], results_logistic['std_auc']],
    }
    
    # Add fold columns dynamically
    for i in range(args.cv_folds):
        cv_summary_data[f'fold_{i+1}'] = [results_rank['fold_aucs'][i], results_logistic['fold_aucs'][i]]
    
    cv_summary = pd.DataFrame(cv_summary_data)
    
    cv_summary_file = output_dir / "ar_kalman_cv_summary.csv"
    cv_summary.to_csv(cv_summary_file, index=False)
    print(f"✅ Saved CV summary to: {cv_summary_file}")
    
    # Save full feature matrix (for inspection)
    features_file = output_dir / "ar_kalman_features.parquet"
    features_df.to_parquet(features_file)
    print(f"✅ Saved features to: {features_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nBaseline AUC (rank aggregation): {results_rank['mean_auc']:.4f} ± {results_rank['std_auc']:.4f}")
    print(f"Best single feature AUC: {results_rank['per_feature_aucs'].max():.4f}")
    print(f"Median single feature AUC: {results_rank['per_feature_aucs'].median():.4f}")
    
    # Count features with signal
    n_signal = (results_rank['per_feature_aucs'] > 0.55).sum()
    n_total = len(results_rank['per_feature_aucs'])
    print(f"\nFeatures with signal (AUC > 0.55): {n_signal}/{n_total} ({n_signal/n_total*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    if results_rank['mean_auc'] < 0.60:
        print("\n❌ Baseline AUC < 0.60: Features need improvement")
        print("   1. Check per-feature AUCs in results/ar_kalman_feature_aucs.csv")
        print("   2. Try different preprocessing (winsorization, detrending)")
        print("   3. Try different AR models (AR(2), ARMA)")
        print("   4. Check if break types match AR/Kalman assumptions")
    elif results_rank['mean_auc'] < 0.75:
        print("\n⚠️  Baseline AUC 0.60-0.75: Moderate signal")
        print("   1. Features contain some information")
        print("   2. Feature selection may help (remove weak features)")
        print("   3. Try combining with other feature families")
        print("   4. ML model (GBM) may provide improvements")
    else:
        print("\n✅ Baseline AUC ≥ 0.75: Good signal!")
        print("   1. AR/Kalman features are working well")
        print("   2. Can use rank aggregation as baseline model")
        print("   3. ML model (GBM) may provide small improvements")
        print("   4. Ready for submission")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
