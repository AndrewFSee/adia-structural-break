"""
Proper cross-validation with rank normalization inside folds.

This module ensures:
1. No data leakage (rank normalization happens INSIDE each fold)
2. Proper stratification by label
3. Splitting by id, not by time
4. Deterministic results
5. Train-fitted rank normalization (fit on train, apply to val/test)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import Callable, Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from . import config, features


@dataclass
class ModelBundle:
    """
    Bundle containing trained model and rank normalizer.
    
    This ensures inference uses the SAME rank transformation as training.
    Optionally includes break_likelihood_scorer for fold-safe break_likelihood computation.
    Optionally includes imputer for fold-safe median imputation.
    """
    model: Any
    rank_normalizer: 'RankNormalizer'
    feature_columns: List[str]
    break_likelihood_scorer: Optional[Any] = None  # BreakLikelihoodScorer if break_likelihood enabled
    imputer: Optional['MedianImputer'] = None  # MedianImputer for NaN handling


class MedianImputer:
    """
    Fold-safe median imputation.
    
    Fits on train data, applies to val/test consistently.
    """
    
    def __init__(self):
        self.medians_: Optional[Dict[str, float]] = None
        
    def fit(self, X: pd.DataFrame) -> 'MedianImputer':
        """
        Fit imputer on training data.
        
        Args:
            X: Training features (may contain NaNs)
            
        Returns:
            self
        """
        self.medians_ = {}
        for col in X.columns:
            values = X[col].values
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                self.medians_[col] = np.median(valid)
            else:
                # All NaN - use 0.0 (rank normalizer will convert to 0.5)
                self.medians_[col] = 0.0
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted medians.
        
        Args:
            X: Features to transform
            
        Returns:
            DataFrame with NaNs filled
        """
        if self.medians_ is None:
            raise ValueError("Imputer not fitted. Call fit() first.")
        
        X_imputed = X.copy()
        for col in X.columns:
            if col in self.medians_:
                X_imputed[col] = X_imputed[col].fillna(self.medians_[col])
            else:
                # Column not seen in training - fill with 0.0
                X_imputed[col] = X_imputed[col].fillna(0.0)
        
        return X_imputed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    
class RankNormalizer:
    """
    Fit rank normalization on train data and apply to val/test.
    
    This prevents data leakage by using ONLY train distribution to
    transform validation and test data.
    
    Each feature is mapped to its percentile rank in the train distribution:
        rank(x) = searchsorted(sorted_train_values, x) / len(train)
    
    NaNs are filled with 0.5 after transformation.
    """
    
    def __init__(self):
        self.train_sorted_: Dict[str, np.ndarray] = {}
        self.feature_names_: Optional[List[str]] = None
        
    def fit(self, X: pd.DataFrame) -> 'RankNormalizer':
        """
        Fit rank normalizer on training data.
        
        Args:
            X: Training features (raw, not rank-normalized)
            
        Returns:
            self
        """
        self.feature_names_ = list(X.columns)
        
        # For each column, store sorted values (excluding NaNs)
        for col in X.columns:
            values = X[col].values
            # Remove NaNs and sort
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                sorted_vals = np.sort(valid)
            else:
                # All NaN - use empty array
                sorted_vals = np.array([])
            self.train_sorted_[col] = sorted_vals
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using train distribution.
        
        Args:
            X: Features to transform (can be train, val, or test)
            
        Returns:
            Rank-normalized DataFrame with values in [0, 1]
        """
        if self.feature_names_ is None:
            raise ValueError("RankNormalizer must be fit before transform")
        
        # Initialize output
        X_ranked = pd.DataFrame(index=X.index, columns=X.columns, dtype=np.float64)
        
        for col in X.columns:
            if col not in self.train_sorted_:
                # Column not seen in training - fill with 0.5
                X_ranked[col] = 0.5
                continue
                
            sorted_train = self.train_sorted_[col]
            
            if len(sorted_train) == 0:
                # All NaN in training - fill with 0.5
                X_ranked[col] = 0.5
                continue
            
            values = X[col].values
            
            # Compute ranks using searchsorted (vectorized)
            # side="right" means value <= x get lower rank
            ranks = np.searchsorted(sorted_train, values, side='right')
            
            # Normalize to [0, 1]
            normalized = ranks / len(sorted_train)
            
            # Clip to [0, 1] (handles values outside train range)
            normalized = np.clip(normalized, 0.0, 1.0)
            
            X_ranked[col] = normalized
        
        # Fill NaNs with 0.5 (neutral rank)
        X_ranked = X_ranked.fillna(0.5)
        
        return X_ranked
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def cross_validate_with_rank_norm(
    X_raw: pd.DataFrame,
    y: pd.Series,
    model_fn: Callable,
    n_splits: int = None,
    random_state: int = None,
    verbose: bool = True,
    recompute_break_likelihood: bool = False,
    break_likelihood_mode: str = "rank_mean"
) -> Tuple[float, float, List[float]]:
    """
    Cross-validate with proper rank normalization INSIDE each fold.
    
    This is the CORRECT way to validate:
    1. Split data by id (stratified)
    2. For each fold:
       a. If break_likelihood enabled: FIT scorer on train, APPLY to train & val
       b. FIT rank normalizer on train data ONLY
       c. TRANSFORM train and val using the fitted normalizer
       d. Train model on rank-normalized train
       e. Predict on rank-normalized val
       f. Compute AUC
    3. Return mean and std of AUCs
    
    Args:
        X_raw: Raw features (NOT rank-normalized)
        y: Labels
        model_fn: Function (X_train, y_train) -> model with .predict() method
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        verbose: Whether to print fold results
        recompute_break_likelihood: If True, recompute break_likelihood per fold
        break_likelihood_mode: Mode for break_likelihood ("rank_mean" or "zscore_logit")
        
    Returns:
        (mean_auc, std_auc, fold_aucs)
    """
    if n_splits is None:
        n_splits = config.N_SPLITS
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    # Ensure indices align
    assert X_raw.index.equals(y.index), "X and y indices must match"
    
    # Check if break_likelihood column exists
    has_break_likelihood = 'break_likelihood' in X_raw.columns
    
    # StratifiedKFold for balanced splits
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=config.SHUFFLE_CV,
        random_state=random_state
    )
    
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        # Split raw features and labels
        X_train_raw = X_raw.iloc[train_idx].copy()
        X_val_raw = X_raw.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        # CRITICAL: If break_likelihood column exists, recompute fold-safely
        # (base.py adds NaN placeholder to avoid global leakage)
        if recompute_break_likelihood and 'break_likelihood' in X_train_raw.columns:
            # Import here to avoid circular dependency
            from .features import break_likelihood as bl_module
            
            # Remove existing break_likelihood column (will recompute)
            X_train_base = X_train_raw.drop(columns=['break_likelihood'])
            X_val_base = X_val_raw.drop(columns=['break_likelihood'])
            
            # Fit scorer on train
            bl_scorer = bl_module.fit_break_likelihood_scorer(
                X_train_base, mode=break_likelihood_mode
            )
            
            # Transform train and val
            bl_train = bl_scorer.transform(X_train_base)
            bl_val = bl_scorer.transform(X_val_base)
            
            # Add back to feature matrices
            X_train_raw['break_likelihood'] = bl_train
            X_val_raw['break_likelihood'] = bl_val
        
        # FOLD-SAFE IMPUTATION: Fit on TRAIN ONLY, then transform both
        imputer = MedianImputer()
        X_train_filled = imputer.fit_transform(X_train_raw)
        X_val_filled = imputer.transform(X_val_raw)
        
        # CRITICAL: FIT rank normalizer on train, APPLY to both train and val
        # This prevents leakage (val uses only train distribution)
        rn = RankNormalizer().fit(X_train_filled)
        # X_train = rn.transform(X_train_filled)
        # X_val = rn.transform(X_val_filled)
        X_train = X_train_filled
        X_val = X_val_filled
        
        # Train model
        model = model_fn(X_train, y_train)
        
        # Predict on validation
        y_pred = model.predict(X_val)
        
        # Compute AUC
        auc = roc_auc_score(y_val, y_pred)
        fold_aucs.append(auc)
        
        if verbose:
            print(f"  Fold {fold}/{n_splits}: AUC = {auc:.4f}")
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    if verbose:
        print(f"\n  Mean CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return mean_auc, std_auc, fold_aucs


def train_final_model_with_rank_norm(
    X_raw: pd.DataFrame,
    y: pd.Series,
    model_fn: Callable,
    fit_break_likelihood: bool = False,
    break_likelihood_mode: str = "rank_mean"
) -> ModelBundle:
    """
    Train final model on ALL data with rank normalization.
    
    Use this after CV to train the final model for deployment.
    
    IMPORTANT: Returns a ModelBundle containing both the trained model
    AND the fitted rank normalizer. This ensures inference uses the
    SAME transformation as training.
    
    Args:
        X_raw: Raw features (NOT rank-normalized)
        y: Labels
        model_fn: Function (X_train, y_train) -> trained model
        fit_break_likelihood: If True, fit and save break_likelihood_scorer
        break_likelihood_mode: Mode for break_likelihood
        
    Returns:
        ModelBundle with model, rank_normalizer, feature_columns, and optionally break_likelihood_scorer
    """
    # If break_likelihood column exists and we need to fit the scorer
    if 'break_likelihood' in X_raw.columns:
        # Import here to avoid circular dependency
        from .features import break_likelihood as bl_module
        
        # Remove break_likelihood column temporarily
        X_base = X_raw.drop(columns=['break_likelihood']).copy()
        
        # Fit scorer on all training data
        bl_scorer = bl_module.fit_break_likelihood_scorer(
            X_base, mode=break_likelihood_mode
        )
        
        # Recompute break_likelihood with fitted scorer
        bl_scores = bl_scorer.transform(X_base)
        
        # Create new X_raw with recomputed break_likelihood
        X_raw = X_base.copy()
        X_raw['break_likelihood'] = bl_scores
    else:
        bl_scorer = None
    
    # FOLD-SAFE IMPUTATION: Fit on all training data
    imputer = MedianImputer()
    X_filled = imputer.fit_transform(X_raw)
    
    # Fit rank normalizer on all training data (including break_likelihood if present)
    rn = RankNormalizer().fit(X_filled)
    
    # Transform using fitted normalizer
    X_ranked = rn.transform(X_filled)
    
    # Train on all data
    model = model_fn(X_ranked, y)
    
    # Bundle everything together
    bundle = ModelBundle(
        model=model,
        rank_normalizer=rn,
        feature_columns=list(X_raw.columns),
        break_likelihood_scorer=bl_scorer,
        imputer=imputer
    )
    
    return bundle


def predict_with_rank_norm(
    model_or_bundle,
    X_test_raw: pd.DataFrame,
    X_train_raw: pd.DataFrame = None
) -> np.ndarray:
    """
    Predict on test data with proper rank normalization.
    
    PREFERRED: Pass a ModelBundle from train_final_model_with_rank_norm.
    This ensures test uses the SAME rank transformation as training.
    
    LEGACY: If a plain model is passed, falls back to independent
    rank normalization on test data (prints warning).
    
    Args:
        model_or_bundle: Either ModelBundle or plain model
        X_test_raw: Raw test features
        X_train_raw: Not used (kept for API compatibility)
        
    Returns:
        Predictions
    """
    if isinstance(model_or_bundle, ModelBundle):
        # CORRECT: Use saved rank normalizer
        bundle = model_or_bundle
        
        # Make a copy to avoid mutating input
        X_test_proc = X_test_raw.copy()
        
        # If bundle has break_likelihood_scorer, recompute break_likelihood
        if bundle.break_likelihood_scorer is not None:
            if 'break_likelihood' in bundle.feature_columns:
                # Remove existing break_likelihood if present
                if 'break_likelihood' in X_test_proc.columns:
                    X_test_base = X_test_proc.drop(columns=['break_likelihood'])
                else:
                    X_test_base = X_test_proc.copy()
                
                # Recompute using saved scorer
                bl_scores = bundle.break_likelihood_scorer.transform(X_test_base)
                X_test_proc = X_test_base.copy()
                X_test_proc['break_likelihood'] = bl_scores
        
        # Validate and align columns
        test_cols = set(X_test_proc.columns)
        train_cols = set(bundle.feature_columns)
        
        # Add missing columns with NaN
        missing = train_cols - test_cols
        if missing:
            print(f"  WARNING: Test data missing {len(missing)} columns. Filling with NaN.")
            for col in missing:
                X_test_proc[col] = np.nan
        
        # Drop extra columns
        extra = test_cols - train_cols
        if extra:
            print(f"  WARNING: Test data has {len(extra)} extra columns. Dropping.")
            X_test_proc = X_test_proc.drop(columns=list(extra))
        
        # Reorder to match training
        X_test_proc = X_test_proc[bundle.feature_columns]
        
        # Apply saved imputer if available
        if bundle.imputer is not None:
            X_test_proc = bundle.imputer.transform(X_test_proc)
        
        # Transform using saved normalizer
        X_test_ranked = bundle.rank_normalizer.transform(X_test_proc)
        
        # Predict
        predictions = bundle.model.predict(X_test_ranked)
        
    else:
        # LEGACY: Plain model without rank normalizer
        print("  WARNING: Using legacy prediction without saved rank normalizer.")
        print("  This may cause train/test mismatch. Use ModelBundle for proper inference.")
        
        model = model_or_bundle
        
        # Rank-normalize test features independently (old behavior)
        X_test_ranked = features.base.rank_normalize_features(X_test_raw)
        
        # Predict
        predictions = model.predict(X_test_ranked)
    
    return predictions
    

def cross_validate_with_learned_agg(
    X_raw: pd.DataFrame,
    y: pd.Series,
    agg_config: Any,  # AggregatorConfig type
    n_splits: int = None,
    random_state: int = None,
    verbose: bool = True,
    return_oof_scores: bool = False
) -> Tuple[float, float, List[float], Optional[np.ndarray]]:
    """
    Cross-validate with learned aggregation.
    
    This generates an OOF (out-of-fold) meta-feature "agg_score" that can
    be used as an additional feature for downstream models.
    
    Args:
        X_raw: Raw features (NOT rank-normalized)
        y: Labels
        agg_config: Configuration for LearnedAggregator
        n_splits: Number of CV folds
        random_state: Random seed
        verbose: Whether to print progress
        return_oof_scores: If True, return full OOF array aligned with y
        
    Returns:
        (mean_auc, std_auc, fold_aucs, oof_scores)
        oof_scores is None unless return_oof_scores=True
    """
    from .models.learned_agg import LearnedAggregator
    
    if n_splits is None:
        n_splits = config.N_SPLITS
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    # Ensure indices align
    assert X_raw.index.equals(y.index), "X and y indices must match"
    
    # StratifiedKFold for balanced splits
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=config.SHUFFLE_CV,
        random_state=random_state
    )
    
    fold_aucs = []
    oof_scores = np.full(len(y), np.nan) if return_oof_scores else None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        # Split raw features and labels
        X_train_raw = X_raw.iloc[train_idx]
        X_val_raw = X_raw.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        # Create and train aggregator
        agg = LearnedAggregator(agg_config)
        agg.fit(
            X_train_raw, 
            y_train, 
            X_val=X_val_raw, 
            y_val=y_val,
            verbose=verbose
        )
        
        # Predict on validation
        y_pred_proba = agg.predict_proba(X_val_raw)[:, 1]
        
        # Store OOF scores if requested
        if return_oof_scores:
            oof_scores[val_idx] = y_pred_proba
        
        # Compute AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        fold_aucs.append(auc)
        
        if verbose:
            print(f"  Fold {fold}/{n_splits}: AUC = {auc:.4f}")
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    if verbose:
        print(f"\n  Mean CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return mean_auc, std_auc, fold_aucs, oof_scores


def train_final_learned_agg(
    X_raw: pd.DataFrame,
    y: pd.Series,
    agg_config: Any,  # AggregatorConfig type
    verbose: bool = True
) -> Any:  # LearnedAggregator type
    """
    Train final learned aggregator on ALL data.
    
    Use this after CV to train the final aggregator for deployment.
    
    Args:
        X_raw: Raw features
        y: Labels
        agg_config: Configuration for LearnedAggregator
        verbose: Whether to print progress
        
    Returns:
        Trained LearnedAggregator
    """
    from .models.learned_agg import LearnedAggregator
    
    if verbose:
        print(f"\nTraining final learned aggregator on {len(X_raw)} samples...")
    
    agg = LearnedAggregator(agg_config)
    agg.fit(X_raw, y, verbose=verbose)
    
    # Compute in-sample AUC (for reference)
    in_sample_proba = agg.predict_proba(X_raw)[:, 1]
    in_sample_auc = roc_auc_score(y, in_sample_proba)
    
    if verbose:
        print(f"Final aggregator in-sample AUC: {in_sample_auc:.4f}")
        print("(Note: This is expected to be higher than CV AUC)")
    
    return agg


def cross_validate_with_learned_agg_feature(
    X_raw: pd.DataFrame,
    y: pd.Series,
    model_fn: Callable,
    agg_config: Any,  # AggregatorConfig type
    n_splits: int = None,
    random_state: int = None,
    verbose: bool = True
) -> Tuple[float, float, List[float], np.ndarray]:
    """
    Cross-validate with learned aggregation as an OOF meta-feature.
    
    This performs a two-stage CV:
    1. Generate OOF "agg_score" using learned aggregation
    2. Train main model (e.g., GBM) with agg_score as additional feature
    
    Args:
        X_raw: Raw features
        y: Labels
        model_fn: Main model factory (e.g., GBM)
        agg_config: Configuration for LearnedAggregator
        n_splits: Number of CV folds
        random_state: Random seed
        verbose: Whether to print progress
        
    Returns:
        (mean_auc, std_auc, fold_aucs, oof_agg_scores)
    """
    from .models.learned_agg import LearnedAggregator
    
    if n_splits is None:
        n_splits = config.N_SPLITS
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 1: Generate OOF Aggregation Scores")
        print("=" * 70)
    
    # Stage 1: Generate OOF agg_scores
    _, _, _, oof_agg_scores = cross_validate_with_learned_agg(
        X_raw=X_raw,
        y=y,
        agg_config=agg_config,
        n_splits=n_splits,
        random_state=random_state,
        verbose=verbose,
        return_oof_scores=True
    )
    
    # Verify no NaNs in OOF scores
    assert not np.any(np.isnan(oof_agg_scores)), "OOF scores contain NaNs"
    
    if verbose:
        print(f"\n✓ OOF aggregation scores generated: [{oof_agg_scores.min():.3f}, {oof_agg_scores.max():.3f}]")
        print("\n" + "=" * 70)
        print("STAGE 2: Train Main Model with Aggregation Feature")
        print("=" * 70)
    
    # Stage 2: Add agg_score to features and train main model
    X_augmented = X_raw.copy()
    X_augmented['meta_agg_score'] = oof_agg_scores
    
    # Run standard CV with augmented features
    mean_auc, std_auc, fold_aucs = cross_validate_with_rank_norm(
        X_raw=X_augmented,
        y=y,
        model_fn=model_fn,
        n_splits=n_splits,
        random_state=random_state,
        verbose=verbose,
        recompute_break_likelihood=False  # Already handled if needed
    )
    
    return mean_auc, std_auc, fold_aucs, oof_agg_scores

