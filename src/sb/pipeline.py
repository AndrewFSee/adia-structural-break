"""
End-to-end pipeline orchestration.
"""

import pandas as pd
from typing import Optional
from . import io, features, cv
from .models import gbm


def run_baseline_pipeline(
    df: pd.DataFrame, 
    y: Optional[pd.Series] = None, 
    use_multiscale: bool = False,
    use_spectral: bool = False
) -> pd.Series:
    """
    Run the Day 1-2 baseline pipeline: features + rank aggregation (no ML).
    
    This is deterministic, fast, and originality-safe.
    
    Args:
        df: Raw time series data
        y: Labels (optional, for evaluation only)
        use_multiscale: Whether to include multi-scale features (default: False)
        use_spectral: Whether to include spectral features (default: False)
        
    Returns:
        Scores per id
    """
    print("Computing features...")
    scores = features.base.compute_and_aggregate(df, use_multiscale=use_multiscale, use_spectral=use_spectral)
    
    if y is not None:
        print("\nEvaluating baseline...")
        cv.print_evaluation_summary(y, scores)
    
    return scores


def run_gbm_pipeline(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_test: Optional[pd.DataFrame] = None,
    use_multiscale: bool = False,
    use_spectral: bool = False
) -> tuple:
    """
    Run enhanced pipeline with LightGBM.
    
    Use this after the baseline is working well.
    
    Args:
        df_train: Training time series data
        y_train: Training labels
        df_test: Test time series data (optional)
        use_multiscale: Whether to use multi-scale features
        use_spectral: Whether to use spectral features
        
    Returns:
        (trained_model, train_predictions, test_predictions)
    """
    print("Computing training features...")
    X_train = features.base.compute_features(df_train, use_multiscale=use_multiscale, use_spectral=use_spectral)
    
    print("Training LightGBM model...")
    model = gbm.train_gbm(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    print("\nFeature importance:")
    print(model.get_feature_importance().head(10))
    
    test_preds = None
    if df_test is not None:
        print("\nComputing test features...")
        X_test = features.base.compute_features(df_test, use_multiscale=use_multiscale, use_spectral=use_spectral)
        test_preds = model.predict(X_test)
    
    return model, train_preds, test_preds
