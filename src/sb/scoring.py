"""
Simple scoring function using AR/Kalman features without complex ML.

This provides a baseline predictor for structural breaks using:
- Rank aggregation of top features
- Or calibrated logistic regression with strong L2

⚠️ ANTI-LEAKAGE: Must be trained on train set, applied to test set.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from typing import Optional, List
import warnings


class ARKalmanScorer:
    """
    Simple scorer for structural breaks using AR/Kalman features.
    
    Two modes:
    1. 'rank_mean': Simple rank aggregation (no training needed)
    2. 'logistic': Regularized logistic regression (needs training)
    """
    
    def __init__(self, mode: str = 'rank_mean', top_k_features: Optional[int] = None):
        """
        Initialize scorer.
        
        Args:
            mode: 'rank_mean' or 'logistic'
            top_k_features: If provided, use only top K features by AUC
        """
        self.mode = mode
        self.top_k_features = top_k_features
        self.model = None
        self.feature_subset = None
        self.train_medians = None
        self.train_percentiles = None
        
    def _impute_and_rank(self, X: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Impute NaNs and rank-normalize features.
        
        ⚠️ ANTI-LEAKAGE: Uses train statistics stored during fit.
        
        Args:
            X: Feature matrix
            is_train: Whether this is training data (fit statistics)
            
        Returns:
            Processed feature matrix
        """
        if is_train:
            # Fit imputation and ranking on train
            self.train_medians = X.median()
            self.train_medians = self.train_medians.fillna(0.0)
            
            X_imputed = X.fillna(self.train_medians)
            X_ranked = X_imputed.rank(pct=True, method='average')
            
            # Store percentiles for test ranking
            self.train_percentiles = {}
            for col in X_imputed.columns:
                self.train_percentiles[col] = np.sort(X_imputed[col].values)
            
            return X_ranked
        else:
            # Apply train statistics to test
            X_imputed = X.fillna(self.train_medians)
            
            # Rank relative to train distribution
            X_ranked = X_imputed.copy()
            for col in X_imputed.columns:
                train_sorted = self.train_percentiles[col]
                test_values = X_imputed[col].values
                
                ranks = np.searchsorted(train_sorted, test_values, side='right')
                percentiles = ranks / len(train_sorted)
                percentiles = np.clip(percentiles, 0.0, 1.0)
                
                X_ranked[col] = percentiles
            
            return X_ranked
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the scorer on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Process features
        X_train_proc = self._impute_and_rank(X_train, is_train=True)
        
        # Feature selection (optional)
        if self.top_k_features is not None:
            from sklearn.metrics import roc_auc_score
            
            feature_aucs = {}
            for col in X_train_proc.columns:
                try:
                    auc = roc_auc_score(y_train, X_train_proc[col])
                    auc_flipped = roc_auc_score(y_train, -X_train_proc[col])
                    feature_aucs[col] = max(auc, auc_flipped)
                except:
                    feature_aucs[col] = 0.5
            
            # Select top K
            top_features = sorted(feature_aucs.items(), key=lambda x: x[1], reverse=True)
            self.feature_subset = [f[0] for f in top_features[:self.top_k_features]]
            
            X_train_proc = X_train_proc[self.feature_subset]
        
        # Train model if logistic
        if self.mode == 'logistic':
            self.model = LogisticRegressionCV(
                Cs=[0.001, 0.01, 0.1],
                cv=3,
                scoring='roc_auc',
                max_iter=500,
                random_state=42,
                n_jobs=1
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.model.fit(X_train_proc, y_train)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for test data.
        
        Args:
            X_test: Test features
            
        Returns:
            Array of probabilities (0 to 1)
        """
        # Process features (using train statistics!)
        X_test_proc = self._impute_and_rank(X_test, is_train=False)
        
        # Apply feature subset if selected
        if self.feature_subset is not None:
            X_test_proc = X_test_proc[self.feature_subset]
        
        # Predict
        if self.mode == 'rank_mean':
            # Simple mean of ranks
            probs = X_test_proc.mean(axis=1).values
        elif self.mode == 'logistic':
            probs = self.model.predict_proba(X_test_proc)[:, 1]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return probs


def score_series_id(df: pd.DataFrame, series_id: str, 
                    scorer: ARKalmanScorer) -> float:
    """
    Score a single series for structural break probability.
    
    Args:
        df: Full dataset with MultiIndex (id, time)
        series_id: ID of series to score
        scorer: Fitted ARKalmanScorer
        
    Returns:
        Probability of structural break (0 to 1)
    """
    # Extract series
    series_data = df.loc[series_id]
    
    # Compute features (this is leakage-safe by design)
    from sb.features.ar_kalman_features import compute_ar_kalman_features_single, split_pre_post
    
    pre, post = split_pre_post(series_data)
    features = compute_ar_kalman_features_single(pre, post, window_sizes=[25, 50, 100])
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features], index=[series_id])
    
    # Score
    prob = scorer.predict_proba(features_df)[0]
    
    return prob


# Example usage function
def create_baseline_scorer(X_train: pd.DataFrame, y_train: pd.Series,
                          mode: str = 'rank_mean') -> ARKalmanScorer:
    """
    Create and fit a baseline scorer.
    
    Example:
        from sb.features.ar_kalman_features import compute_ar_kalman_features
        from sb.scoring import create_baseline_scorer
        
        # Compute features
        X_train_feats = compute_ar_kalman_features(X_train)
        
        # Create scorer
        scorer = create_baseline_scorer(X_train_feats, y_train, mode='rank_mean')
        
        # Score test data
        X_test_feats = compute_ar_kalman_features(X_test)
        probs = scorer.predict_proba(X_test_feats)
    
    Args:
        X_train: Training features
        y_train: Training labels
        mode: 'rank_mean' or 'logistic'
        
    Returns:
        Fitted scorer
    """
    scorer = ARKalmanScorer(mode=mode, top_k_features=None)
    scorer.fit(X_train, y_train)
    return scorer
