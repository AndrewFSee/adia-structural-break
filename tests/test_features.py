"""Unit tests for feature extraction."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sb import config, preprocessing
from sb.features import dist, dynamics


def test_robust_scale():
    """Test robust scaling."""
    x = np.array([1, 2, 3, 4, 5, 100])  # Outlier at end
    scaled = preprocessing.robust_scale(x)
    
    # Should have median ~ 0
    assert abs(np.median(scaled)) < 0.5
    print("✅ test_robust_scale passed")


def test_quantile_features():
    """Test quantile-based features."""
    x0 = np.random.randn(100)
    x1 = np.random.randn(100) * 2 + 1  # Different distribution
    
    features = dist.quantile_features(x0, x1)
    
    assert "delta_q_slope" in features
    assert "median_shift" in features
    assert "iqr_ratio" in features
    assert all(v >= 0 for v in features.values())
    
    print("✅ test_quantile_features passed")


def test_entropy():
    """Test entropy calculation."""
    # Uniform distribution should have high entropy
    x_uniform = np.random.uniform(0, 1, 1000)
    # Constant should have low entropy (but may be zero)
    x_constant = np.ones(1000)
    
    ent_uniform = dist.entropy(x_uniform)
    ent_constant = dist.entropy(x_constant)
    
    assert ent_uniform > ent_constant
    print("✅ test_entropy passed")


def test_rolling_var_slope():
    """Test rolling variance slope."""
    # Increasing volatility
    x = np.concatenate([
        np.random.randn(100) * 0.5,
        np.random.randn(100) * 2.0
    ])
    
    slope = dynamics.rolling_var_slope(x)
    assert slope >= 0  # Should detect increase
    
    print("✅ test_rolling_var_slope passed")


def test_kalman_variance():
    """Test Kalman variance proxy."""
    x = np.random.randn(100)
    var = dynamics.kalman_level_variance(x)
    
    assert var >= 0
    assert not np.isnan(var)
    
    print("✅ test_kalman_variance passed")


def test_determinism():
    """Test that features are deterministic."""
    np.random.seed(42)
    x0 = np.random.randn(100)
    x1 = np.random.randn(100)
    
    features1 = dist.quantile_features(x0, x1)
    features2 = dist.quantile_features(x0, x1)
    
    assert features1 == features2
    
    print("✅ test_determinism passed")


if __name__ == "__main__":
    print("Running feature tests...\n")
    
    test_robust_scale()
    test_quantile_features()
    test_entropy()
    test_rolling_var_slope()
    test_kalman_variance()
    test_determinism()
    
    print("\n✅ All tests passed!")
