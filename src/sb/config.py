"""
Configuration constants for structural break detection.
All magic numbers and seeds defined here for reproducibility.
"""

# Random seeds for determinism
RANDOM_SEED = 42
NUMPY_SEED = 42
LIGHTGBM_SEED = 42

# Feature computation parameters
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
ENTROPY_BINS = 20
ROLLING_WINDOW = 50  # For volatility slope computation
MIN_WINDOW_MULTIPLIER = 2  # Minimum series length = ROLLING_WINDOW * this

# Multi-scale windows for boundary-focused features
# These windows look at points near the break point
MULTI_SCALE_WINDOWS = [50, 100, 250]  # Last/first N points around boundary

# Robust scaling
MAD_EPSILON = 1e-8  # To avoid division by zero

# Kalman/variance parameters
DIFF_ORDER = 1  # For first-order differences

# Model parameters - BALANCED REGULARIZATION
# Updated 2026-01-01: Relaxed from HEAVILY REGULARIZED to improve performance
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 500,            # More trees (was 300)
    "learning_rate": 0.05,          # Slightly faster (was 0.03)
    "max_depth": 5,                 # Deeper trees (was 4)
    "num_leaves": 31,               # More leaves (was 15)
    "min_data_in_leaf": 100,        # Lower minimum (was 200)
    "feature_fraction": 0.8,        # Column subsampling (unchanged)
    "bagging_fraction": 0.8,        # Row subsampling (unchanged)
    "bagging_freq": 1,              # Subsample every iteration (unchanged)
    "lambda_l2": 0.5,               # Lighter L2 (was 1.0)
    "verbose": -1,
    "random_state": LIGHTGBM_SEED,
    "deterministic": True,
    "force_col_wise": True,         # Deterministic column ordering
}

# Cross-validation - PROPER VALIDATION, NO IN-SAMPLE EVAL
N_SPLITS = 5
SHUFFLE_CV = True
TEST_SIZE = 0.2

# Data validation
REQUIRED_COLUMNS = ["id", "period", "value"]
EXPECTED_PERIODS = [0, 1]
