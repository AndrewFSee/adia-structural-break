"""
Sanity checks for the submission.

Verifies:
1. Determinism: Same input → same output
2. Speed: Can process 100 series quickly
3. Monotonicity: Higher scores correlate with more breaks
4. Score range: All predictions in [0, 1]
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add src and parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sb import config, io
from solution import train, infer


def generate_synthetic_data(n_series: int = 100, n_points: int = 200) -> pd.DataFrame:
    """
    Generate synthetic time series for testing.
    
    Args:
        n_series: Number of series to generate
        n_points: Points per series per period
        
    Returns:
        DataFrame in expected format
    """
    np.random.seed(config.RANDOM_SEED)
    
    data = []
    
    for i in range(n_series):
        series_id = f"test_{i:04d}"
        
        # Period 0: baseline
        x0 = np.random.randn(n_points)
        
        # Period 1: add a break for half the series
        if i < n_series // 2:
            # No break
            x1 = np.random.randn(n_points)
        else:
            # With break: shift mean and/or variance
            x1 = np.random.randn(n_points) * 1.5 + 0.5
        
        for t, val in enumerate(x0):
            data.append({"id": series_id, "period": 0, "time": t, "value": val})
        
        for t, val in enumerate(x1):
            data.append({"id": series_id, "period": 1, "time": t, "value": val})
    
    return pd.DataFrame(data)


def test_determinism():
    """Test that inference is deterministic."""
    print("\n" + "=" * 70)
    print("TEST 1: DETERMINISM")
    print("=" * 70)
    
    df = generate_synthetic_data(n_series=50)
    
    print("Running inference twice on same data...")
    preds1 = infer(df)
    preds2 = infer(df)
    
    if np.allclose(preds1.values, preds2.values):
        print("✅ PASS: Predictions are identical")
        return True
    else:
        max_diff = np.max(np.abs(preds1.values - preds2.values))
        print(f"❌ FAIL: Predictions differ (max diff: {max_diff})")
        return False


def test_speed():
    """Test inference speed."""
    print("\n" + "=" * 70)
    print("TEST 2: SPEED")
    print("=" * 70)
    
    df = generate_synthetic_data(n_series=100)
    
    print("Timing inference on 100 series...")
    start = time.time()
    preds = infer(df)
    elapsed = time.time() - start
    
    per_series = elapsed / 100
    extrapolated = per_series * 16000
    
    print(f"Time for 100 series: {elapsed:.3f}s")
    print(f"Time per series: {per_series*1000:.1f}ms")
    print(f"Extrapolated to 16k series: {extrapolated:.1f}s")
    
    if elapsed < 10.0:
        print("✅ PASS: Fast enough for submission")
        return True
    else:
        print("⚠️ WARNING: May be too slow for large submissions")
        return False


def test_monotonicity():
    """Test that scores correlate with breaks."""
    print("\n" + "=" * 70)
    print("TEST 3: MONOTONICITY")
    print("=" * 70)
    
    n_series = 100
    df = generate_synthetic_data(n_series=n_series)
    
    # Create labels (first half no break, second half break)
    labels = pd.Series([0] * (n_series // 2) + [1] * (n_series // 2))
    labels.index = [f"test_{i:04d}" for i in range(n_series)]
    
    preds = infer(df)
    
    # Check if mean score is higher for break cases
    no_break_scores = preds[labels == 0].mean()
    break_scores = preds[labels == 1].mean()
    
    print(f"Mean score (no break): {no_break_scores:.4f}")
    print(f"Mean score (break):    {break_scores:.4f}")
    
    if break_scores > no_break_scores:
        print("✅ PASS: Scores are higher for series with breaks")
        return True
    else:
        print("❌ FAIL: Scores not monotonic with breaks")
        return False


def test_score_range():
    """Test that all scores are in [0, 1]."""
    print("\n" + "=" * 70)
    print("TEST 4: SCORE RANGE")
    print("=" * 70)
    
    df = generate_synthetic_data(n_series=100)
    preds = infer(df)
    
    min_score = preds.min()
    max_score = preds.max()
    
    print(f"Score range: [{min_score:.4f}, {max_score:.4f}]")
    
    if min_score >= 0 and max_score <= 1:
        print("✅ PASS: All scores in [0, 1]")
        return True
    else:
        print("❌ FAIL: Scores outside [0, 1] range")
        return False


def test_no_missing():
    """Test that no predictions are missing."""
    print("\n" + "=" * 70)
    print("TEST 5: NO MISSING VALUES")
    print("=" * 70)
    
    df = generate_synthetic_data(n_series=100)
    preds = infer(df)
    
    n_missing = preds.isna().sum()
    
    print(f"Missing predictions: {n_missing}")
    
    if n_missing == 0:
        print("✅ PASS: No missing predictions")
        return True
    else:
        print("❌ FAIL: Found missing predictions")
        return False


def main():
    print("=" * 70)
    print("SANITY CHECKS FOR STRUCTURAL BREAK SUBMISSION")
    print("=" * 70)
    
    results = {
        "Determinism": test_determinism(),
        "Speed": test_speed(),
        "Monotonicity": test_monotonicity(),
        "Score Range": test_score_range(),
        "No Missing": test_no_missing(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready for submission!")
    else:
        print("⚠️ SOME TESTS FAILED - Review before submitting")
    print("=" * 70)


if __name__ == "__main__":
    main()
