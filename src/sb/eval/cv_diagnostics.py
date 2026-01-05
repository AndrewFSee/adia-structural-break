"""
Cross-validation diagnostics with strict leakage prevention.

KEY ANTI-LEAKAGE MEASURES:
1. Feature computation is leakage-safe by design (uses only per-series data)
2. NaN imputation fit on TRAIN fold only, applied to VAL
3. Rank normalization fit on TRAIN fold only, applied to VAL
4. Any scaling fit on TRAIN fold only, applied to VAL

Never compute global statistics on full dataset before splitting!
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from typing import Tuple, Dict, List
import warnings


def impute_nans_fold_safe(X_train: pd.DataFrame, X_val: pd.DataFrame,
                          strategy: str = 'median') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute NaNs using TRAIN statistics only.
    
    ⚠️ ANTI-LEAKAGE: Compute imputation values from TRAIN, apply to both TRAIN and VAL.
    
    Args:
        X_train: Training features (may contain NaN)
        X_val: Validation features (may contain NaN)
        strategy: 'median' or 'mean'
        
    Returns:
        X_train_imputed, X_val_imputed
    """
    if strategy == 'median':
        fill_values = X_train.median()
    elif strategy == 'mean':
        fill_values = X_train.mean()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Fill any remaining NaNs with 0 (e.g., if entire column is NaN in train)
    fill_values = fill_values.fillna(0.0)
    
    X_train_imputed = X_train.fillna(fill_values)
    X_val_imputed = X_val.fillna(fill_values)
    
    return X_train_imputed, X_val_imputed


def rank_normalize_fold_safe(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank-normalize features using TRAIN ranks only.
    
    ⚠️ ANTI-LEAKAGE: Compute percentile ranks from TRAIN distribution.
    VAL values are ranked relative to TRAIN distribution.
    
    Args:
        X_train: Training features
        X_val: Validation features
        
    Returns:
        X_train_ranked, X_val_ranked (values in [0, 1])
    """
    X_train_ranked = X_train.rank(pct=True, method='average')
    
    # For validation, rank relative to training distribution
    X_val_ranked = X_val.copy()
    
    for col in X_train.columns:
        train_sorted = np.sort(X_train[col].values)
        val_values = X_val[col].values
        
        # Find where each val value would rank in train distribution
        ranks = np.searchsorted(train_sorted, val_values, side='right')
        percentiles = ranks / len(train_sorted)
        
        # Clip to [0, 1]
        percentiles = np.clip(percentiles, 0.0, 1.0)
        
        X_val_ranked[col] = percentiles
    
    return X_train_ranked, X_val_ranked


def compute_per_feature_auc(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Compute AUC for each feature individually.
    
    Args:
        X: Feature matrix
        y: Binary labels
        
    Returns:
        Series of AUCs per feature
    """
    aucs = {}
    
    for col in X.columns:
        feature_values = X[col].values
        
        # Handle NaN
        if np.isnan(feature_values).all():
            aucs[col] = 0.5
            continue
        
        # Try both directions and take max
        try:
            auc_pos = roc_auc_score(y, feature_values)
            auc_neg = roc_auc_score(y, -feature_values)
            aucs[col] = max(auc_pos, auc_neg)
        except:
            aucs[col] = 0.5
    
    return pd.Series(aucs)


def aggregate_score_rank_mean(X: pd.DataFrame, feature_subset: List[str] = None) -> np.ndarray:
    """
    Simple baseline scorer: mean of rank-normalized features.
    
    Args:
        X: Feature matrix (already rank-normalized)
        feature_subset: Optional list of features to use (default: all)
        
    Returns:
        Array of scores (one per sample)
    """
    if feature_subset is not None:
        X_subset = X[feature_subset]
    else:
        X_subset = X
    
    # Mean of ranks
    scores = X_subset.mean(axis=1).values
    
    return scores


def aggregate_score_logistic(X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame) -> np.ndarray:
    """
    Baseline scorer using heavily regularized logistic regression.
    
    ⚠️ ANTI-LEAKAGE: Trained on TRAIN fold only.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        
    Returns:
        Array of predicted probabilities for validation set
    """
    # Strong L2 regularization
    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1],
        cv=3,
        scoring='roc_auc',
        max_iter=500,
        random_state=42,
        n_jobs=1
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf.fit(X_train, y_train)
    
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    
    return y_pred_proba


def run_cv_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    use_rank_normalization: bool = True,
    aggregation: str = 'rank_mean',
    verbose: bool = True
) -> Dict:
    """
    Run cross-validation diagnostics with proper leakage prevention.
    
    ⚠️ ANTI-LEAKAGE: All transformations fit on TRAIN fold only!
    
    Args:
        X: Feature matrix (index=id)
        y: Binary labels (index=id)
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        use_rank_normalization: Whether to rank-normalize features
        aggregation: 'rank_mean' or 'logistic'
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results
    """
    if verbose:
        print("=" * 70)
        print("CROSS-VALIDATION DIAGNOSTICS (LEAKAGE-SAFE)")
        print("=" * 70)
        print(f"Samples: {len(X):,}")
        print(f"Features: {X.shape[1]}")
        print(f"CV folds: {n_splits}")
        print(f"Break rate: {y.mean():.2%}")
        print(f"Rank normalization: {use_rank_normalization}")
        print(f"Aggregation: {aggregation}")
    
    # Stratified K-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_aucs = []
    all_per_feature_aucs = []
    
    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING CV FOLDS")
        print("=" * 70)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split data
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        # Step 1: Impute NaNs (train stats only!)
        X_train_imp, X_val_imp = impute_nans_fold_safe(X_train_raw, X_val_raw, strategy='median')
        
        # Step 2: Rank normalize (train stats only!)
        if use_rank_normalization:
            X_train_proc, X_val_proc = rank_normalize_fold_safe(X_train_imp, X_val_imp)
        else:
            X_train_proc = X_train_imp
            X_val_proc = X_val_imp
        
        # Step 3: Compute per-feature AUCs on validation
        per_feature_aucs = compute_per_feature_auc(X_val_proc, y_val)
        all_per_feature_aucs.append(per_feature_aucs)
        
        # Step 4: Aggregate score
        if aggregation == 'rank_mean':
            val_scores = aggregate_score_rank_mean(X_val_proc)
        elif aggregation == 'logistic':
            val_scores = aggregate_score_logistic(X_train_proc, y_train, X_val_proc)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Step 5: Compute overall AUC
        val_auc = roc_auc_score(y_val, val_scores)
        fold_aucs.append(val_auc)
        
        if verbose:
            print(f"Fold {fold}: Val AUC = {val_auc:.4f} (n_val={len(y_val)})")
    
    # Aggregate results
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    # Average per-feature AUCs across folds
    per_feature_auc_df = pd.DataFrame(all_per_feature_aucs)
    mean_per_feature_aucs = per_feature_auc_df.mean().sort_values(ascending=False)
    
    if verbose:
        print("\n" + "=" * 70)
        print("CV RESULTS")
        print("=" * 70)
        print(f"Out-of-sample AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Fold AUCs: {[f'{x:.4f}' for x in fold_aucs]}")
        
        print("\n" + "=" * 70)
        print("TOP 20 FEATURES BY MEAN AUC")
        print("=" * 70)
        print(mean_per_feature_aucs.head(20).to_string())
        
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        
        if mean_auc < 0.60:
            print("❌ POOR: AUC < 0.60 - Features have weak signal")
            print("   → Problem is likely feature engineering")
            print("   → Try different feature families or preprocessing")
        elif mean_auc < 0.70:
            print("⚠️  MODERATE: 0.60 ≤ AUC < 0.70 - Some signal present")
            print("   → Features contain moderate information")
            print("   → ML model may help but gains limited")
        elif mean_auc < 0.80:
            print("✅ GOOD: 0.70 ≤ AUC < 0.80 - Features contain good signal")
            print("   → Simple baseline works reasonably well")
            print("   → ML model should provide improvements")
        else:
            print("🎉 EXCELLENT: AUC ≥ 0.80 - Strong signal")
            print("   → Features are highly informative")
            print("   → Simple aggregation already works well")
    
    return {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'fold_aucs': fold_aucs,
        'per_feature_aucs': mean_per_feature_aucs,
        'per_feature_auc_df': per_feature_auc_df
    }
