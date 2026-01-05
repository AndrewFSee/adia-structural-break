"""
Cross-validation and evaluation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import Tuple, List
from . import config


def stratified_split_by_id(
    feature_df: pd.DataFrame,
    y: pd.Series,
    test_size: float = None,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split by series id (not by time).
    
    Args:
        feature_df: Feature DataFrame with index=id
        y: Labels with index=id
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    from sklearn.model_selection import train_test_split
    
    ids = feature_df.index.values
    train_ids, test_ids = train_test_split(
        ids,
        test_size=test_size,
        stratify=y.values,
        random_state=random_state
    )
    
    X_train = feature_df.loc[train_ids]
    X_test = feature_df.loc[test_ids]
    y_train = y.loc[train_ids]
    y_test = y.loc[test_ids]
    
    return X_train, X_test, y_train, y_test


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute ROC AUC score.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        ROC AUC score
    """
    return roc_auc_score(y_true, y_pred)


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_fn,
    n_splits: int = None,
    random_state: int = None
) -> Tuple[float, List[float]]:
    """
    Stratified k-fold cross-validation.
    
    Args:
        X: Features
        y: Labels
        model_fn: Function that takes (X_train, y_train, X_val) and returns predictions
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        (mean_auc, fold_aucs)
    """
    if n_splits is None:
        n_splits = config.N_SPLITS
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        y_pred = model_fn(X_train, y_train, X_val)
        auc = evaluate_predictions(y_val.values, y_pred)
        fold_aucs.append(auc)
        
        print(f"Fold {fold}: AUC = {auc:.4f}")
    
    mean_auc = np.mean(fold_aucs)
    print(f"\nMean CV AUC: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
    
    return mean_auc, fold_aucs


def print_evaluation_summary(y_true: pd.Series, scores: pd.Series) -> None:
    """
    Print summary statistics of predictions.
    
    Args:
        y_true: True labels
        scores: Predicted scores
    """
    auc = evaluate_predictions(y_true.values, scores.values)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"ROC AUC: {auc:.4f}")
    print(f"\nScore distribution:")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  25%:    {scores.quantile(0.25):.4f}")
    print(f"  Median: {scores.median():.4f}")
    print(f"  75%:    {scores.quantile(0.75):.4f}")
    print(f"  Max:    {scores.max():.4f}")
    print(f"\nLabel distribution:")
    print(f"  No break (0): {(y_true == 0).sum()} ({(y_true == 0).mean()*100:.1f}%)")
    print(f"  Break (1):    {(y_true == 1).sum()} ({(y_true == 1).mean()*100:.1f}%)")
    print("=" * 60)
