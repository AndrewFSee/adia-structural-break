"""
SOLUTION FILE FOR CRUNCHDAO SUBMISSION - REFACTORED

This file contains the train() and infer() functions required by the platform.

Refactored Strategy (addressing overfitting):
- Proper cross-validation with rank normalization inside folds
- Heavy regularization (LightGBM with constrained parameters)
- Multi-scale features (boundary-focused windows)
- Full determinism (reproducible results)

The pipeline now uses proper CV to assess out-of-sample performance during training,
and applies the same rank normalization process during inference.
"""

import numpy as np
import pandas as pd
from sb import config, features, cv_proper


# Global model storage
_model = None
_use_multiscale = False


def train(X_train: pd.DataFrame, y_train: pd.Series, use_multiscale: bool = False) -> None:
    """
    Train function called by the CrunchDAO platform.
    
    This implements proper cross-validation with:
    1. Stratified K-fold splitting (by id, not time)
    2. Rank normalization INSIDE each fold (no leakage)
    3. Heavily regularized LightGBM
    4. Optional multi-scale features
    
    The model is trained on ALL data after CV validation,
    but CV results provide realistic out-of-sample performance estimate.
    
    Args:
        X_train: Training data with columns [id, period, value]
        y_train: Training labels (0=no break, 1=break)
        use_multiscale: Whether to use multi-scale boundary-focused features
    """
    global _model, _use_multiscale
    
    # Set random seeds for determinism
    np.random.seed(config.RANDOM_SEED)
    
    _use_multiscale = use_multiscale
    
    print("=" * 70)
    print("TRAINING WITH PROPER CROSS-VALIDATION")
    print("=" * 70)
    print(f"Training data shape: {X_train.shape}")
    print(f"Label distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"Break rate: {y_train.mean():.2%}")
    print(f"Multi-scale features: {'YES' if use_multiscale else 'NO'}")
    
    # Extract raw features
    print("\nExtracting features...")
    X_raw = features.base.compute_features(X_train, use_multiscale=use_multiscale)
    print(f"Feature shape: {X_raw.shape}")
    
    # Handle NaN values
    n_nan = X_raw.isna().sum().sum()
    if n_nan > 0:
        print(f"Warning: {n_nan} NaN values found, filled with median")
        X_raw = X_raw.fillna(X_raw.median())
    
    # Create model factory
    from sb import models
    
    def model_fn(X_train_fold, y_train_fold):
        model = models.gbm.StructuralBreakGBM(params=config.LIGHTGBM_PARAMS)
        model.train(X_train_fold, y_train_fold)
        return model
    
    # Run proper cross-validation
    print("\nRunning cross-validation (rank normalization inside folds)...")
    mean_auc, std_auc, fold_aucs = cv_proper.cross_validate_with_rank_norm(
        X_raw=X_raw,
        y=y_train,
        model_fn=model_fn,
        n_splits=5,
        random_state=config.RANDOM_SEED,
        verbose=True
    )
    
    print(f"\nOut-of-sample CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Fold AUCs: {[f'{x:.4f}' for x in fold_aucs]}")
    
    if mean_auc < 0.75:
        print("⚠️  Warning: CV AUC is low (<0.75). Model may not generalize well.")
    elif mean_auc > 0.85:
        print("✅ Excellent CV AUC (>0.85). Model is performing well.")
    
    # Train final model on ALL data
    print("\nTraining final model on all data...")
    _model = cv_proper.train_final_model_with_rank_norm(
        X_raw=X_raw,
        y=y_train,
        model_fn=model_fn
    )
    
    print("✅ Training complete.")


def infer(X_test: pd.DataFrame) -> pd.Series:
    """
    Inference function called by the CrunchDAO platform.
    
    Returns a score ∈ [0, 1] for each series id, where higher scores
    indicate higher probability of structural break.
    
    This applies the same rank normalization process as training,
    but using only the test data statistics (no leakage).
    
    Args:
        X_test: Test data with columns [id, period, value]
        
    Returns:
        Series with index=id and values=predicted scores
    """
    global _model, _use_multiscale
    
    # Set random seeds for determinism
    np.random.seed(config.RANDOM_SEED)
    
    print(f"Running inference on {X_test['id'].nunique()} series...")
    
    if _model is None:
        # Fallback to baseline if no model trained
        print("No model found, using baseline (rank aggregation)...")
        X_raw = features.base.compute_features(X_test, use_multiscale=False)
        scores = features.base.aggregate_features(X_raw)
    else:
        # Extract raw features
        print(f"Extracting {'multi-scale' if _use_multiscale else 'single-scale'} features...")
        X_raw = features.base.compute_features(X_test, use_multiscale=_use_multiscale)
        
        # Handle NaN values
        n_nan = X_raw.isna().sum().sum()
        if n_nan > 0:
            print(f"Warning: {n_nan} NaN values found, filled with median")
            X_raw = X_raw.fillna(X_raw.median())
        
        # Predict with proper rank normalization
        print("Predicting with rank normalization...")
        scores = cv_proper.predict_with_rank_norm(
            model=_model,
            X_raw=X_raw
        )
    
    print(f"Inference complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return scores


# For local testing
if __name__ == "__main__":
    print("This is the solution.py file for CrunchDAO submission.")
    print("Use scripts/train_local.py and scripts/infer_local.py for local testing.")

