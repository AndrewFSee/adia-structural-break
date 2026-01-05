"""
Diagnostic script for spectral features.

Evaluates spectral features on CrunchDAO training data:
- Computes all features (including new spectral features)
- Reports per-feature AUC (single-feature ranking)
- Computes 5-fold CV with rank-aggregation baseline
- Deterministic and leakage-safe
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sb import io
from sb.features import base


def compute_per_feature_auc(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list
) -> pd.DataFrame:
    """
    Compute ROC AUC for each feature individually.
    
    Args:
        X: Feature matrix
        y: Binary labels
        feature_cols: List of feature column names
        
    Returns:
        DataFrame with columns [feature, auc] sorted by AUC descending
    """
    aucs = []
    
    for col in feature_cols:
        feature_vals = X[col].values
        
        # Skip if all NaN or constant
        if np.all(np.isnan(feature_vals)) or np.std(feature_vals) < 1e-10:
            aucs.append({'feature': col, 'auc': 0.5})
            continue
        
        # Rank-normalize to handle NaNs (fill with median rank)
        ranks = pd.Series(feature_vals).rank(pct=True, na_option='keep')
        ranks = ranks.fillna(0.5)
        
        # Compute AUC
        try:
            auc = roc_auc_score(y, ranks)
        except ValueError:
            auc = 0.5
        
        aucs.append({'feature': col, 'auc': auc})
    
    auc_df = pd.DataFrame(aucs)
    auc_df = auc_df.sort_values('auc', ascending=False).reset_index(drop=True)
    
    return auc_df


def rank_normalize_fold_safe(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank-normalize features using TRAIN statistics only.
    
    ⚠️ ANTI-LEAKAGE: Percentiles computed from TRAIN fold only!
    
    Args:
        X_train: Training fold features
        X_val: Validation fold features
        
    Returns:
        X_train_ranked, X_val_ranked
    """
    X_train_ranked = pd.DataFrame(index=X_train.index, columns=X_train.columns)
    X_val_ranked = pd.DataFrame(index=X_val.index, columns=X_val.columns)
    
    for col in X_train.columns:
        train_vals = X_train[col].values
        val_vals = X_val[col].values
        
        # Compute percentiles from TRAIN only
        train_sorted = np.sort(train_vals[~np.isnan(train_vals)])
        
        if len(train_sorted) == 0:
            # All NaN in training - fill with 0.5
            X_train_ranked[col] = 0.5
            X_val_ranked[col] = 0.5
            continue
        
        # Rank train values
        train_ranks = pd.Series(train_vals).rank(pct=True, na_option='keep').fillna(0.5)
        X_train_ranked[col] = train_ranks
        
        # Map val values to train percentiles
        val_ranks = np.searchsorted(train_sorted, val_vals, side='right') / len(train_sorted)
        val_ranks = np.clip(val_ranks, 0, 1)
        
        # Fill NaNs with median
        val_ranks = np.where(np.isnan(val_vals), 0.5, val_ranks)
        X_val_ranked[col] = val_ranks
    
    return X_train_ranked, X_val_ranked


def run_cv_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """
    Run stratified K-fold CV with rank-aggregation baseline.
    
    Args:
        X: Feature matrix
        y: Binary labels
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with CV results
    """
    # Get unique IDs for stratification
    ids = X.index.values
    
    # Stratified K-fold by ID
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_aucs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(ids, y)):
        # Split data
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        # Rank-normalize using TRAIN fold only (leakage-safe)
        X_train_ranked, X_val_ranked = rank_normalize_fold_safe(X_train, X_val)
        
        # Aggregate: simple mean of ranks
        train_scores = X_train_ranked.mean(axis=1)
        val_scores = X_val_ranked.mean(axis=1)
        
        # Compute fold AUC
        try:
            fold_auc = roc_auc_score(y_val, val_scores)
        except ValueError:
            fold_auc = 0.5
        
        fold_aucs.append(fold_auc)
        
        print(f"  Fold {fold_idx+1}/{n_splits}: AUC = {fold_auc:.4f}")
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    return {
        'fold_aucs': fold_aucs,
        'mean_auc': mean_auc,
        'std_auc': std_auc
    }


def main():
    parser = argparse.ArgumentParser(description='Spectral features diagnostic')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top features to display')
    parser.add_argument('--spectral', action='store_true', default=True,
                        help='Use spectral features (default: True for this diagnostic)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("SPECTRAL FEATURES DIAGNOSTIC")
    print("=" * 80)
    print(f"\nSpectral: {'YES' if args.spectral else 'NO'}")
    
    # Load training data
    print(f"\n1. Loading data from {args.data_dir}...")
    X_train, y_train = io.load_train(args.data_dir)
    print(f"   Loaded {len(X_train)} series")
    print(f"   Label distribution: {y_train.value_counts().to_dict()}")
    
    # Compute features (includes spectral features now)
    print("\n2. Computing features (includes spectral)...")
    feature_df = base.compute_features(X_train, use_spectral=args.spectral)
    print(f"   Feature matrix shape: {feature_df.shape}")
    
    # Check for spectral features (exactly 6 expected)
    spectral_cols = [col for col in feature_df.columns if 'spec_entropy' in col or 'log_low_high' in col]
    print(f"   Spectral features found: {len(spectral_cols)} (expected: 6)")
    if len(spectral_cols) > 0:
        print(f"   Spectral features: {sorted(spectral_cols)}")
    
    # Compute per-feature AUC
    print(f"\n3. Computing per-feature AUC (top {args.top_k})...")
    feature_aucs = compute_per_feature_auc(feature_df, y_train, feature_df.columns.tolist())
    
    print("\n   Top features by single-feature AUC:")
    print("   " + "-" * 60)
    for idx, row in feature_aucs.head(args.top_k).iterrows():
        is_spectral = any(kw in row['feature'] for kw in ['spec_entropy', 'log_low_high'])
        marker = " [SPECTRAL]" if is_spectral else ""
        print(f"   {idx+1:2d}. {row['feature']:35s} AUC = {row['auc']:.4f}{marker}")
    
    # Count spectral in top-K
    top_k_features = feature_aucs.head(args.top_k)['feature'].values
    spectral_in_top_k = sum(1 for f in top_k_features if any(kw in f for kw in ['spec_entropy', 'log_low_high']))
    print(f"\n   Spectral features in top-{args.top_k}: {spectral_in_top_k}/{args.top_k}")
    
    # Run CV diagnostics
    print(f"\n4. Running {args.cv_folds}-fold CV with rank-aggregation baseline...")
    cv_results = run_cv_diagnostics(feature_df, y_train, n_splits=args.cv_folds)
    
    print(f"\n   CV Results:")
    print(f"   Mean AUC:  {cv_results['mean_auc']:.4f}")
    print(f"   Std AUC:   {cv_results['std_auc']:.4f}")
    print(f"   Fold AUCs: {[f'{auc:.4f}' for auc in cv_results['fold_aucs']]}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    
    # Summary interpretation
    print("\nInterpretation:")
    if cv_results['mean_auc'] >= 0.75:
        print("  ✓ Strong signal - features capture structural breaks well")
    elif cv_results['mean_auc'] >= 0.60:
        print("  ~ Moderate signal - consider feature selection or engineering")
    else:
        print("  ✗ Weak signal - features may need improvement")
    
    if spectral_in_top_k >= 5:
        print(f"  ✓ Spectral features performing well ({spectral_in_top_k} in top-{args.top_k})")
    elif spectral_in_top_k >= 2:
        print(f"  ~ Some spectral features useful ({spectral_in_top_k} in top-{args.top_k})")
    else:
        print(f"  ✗ Spectral features not ranking high ({spectral_in_top_k} in top-{args.top_k})")


if __name__ == '__main__':
    main()
