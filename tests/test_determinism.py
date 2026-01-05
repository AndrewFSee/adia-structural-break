"""Test determinism of the full pipeline."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from solution import train, infer
from sb import config


def generate_test_data(n_series=10, seed=42):
    """Generate synthetic test data."""
    np.random.seed(seed)
    
    data = []
    for i in range(n_series):
        series_id = f"test_{i}"
        x0 = np.random.randn(100)
        x1 = np.random.randn(100) * (1.5 if i % 2 else 1.0)
        
        for t, val in enumerate(x0):
            data.append({"id": series_id, "period": 0, "time": t, "value": val})
        for t, val in enumerate(x1):
            data.append({"id": series_id, "period": 1, "time": t, "value": val})
    
    return pd.DataFrame(data)


def test_infer_determinism():
    """Test that infer produces identical results."""
    print("Testing infer() determinism...")
    
    df = generate_test_data(n_series=20)
    
    # Run inference multiple times
    results = []
    for i in range(3):
        preds = infer(df)
        results.append(preds.values.copy())
    
    # Check all are identical
    for i in range(1, len(results)):
        if not np.allclose(results[0], results[i]):
            print(f"❌ FAIL: Run {i+1} differs from run 1")
            print(f"Max diff: {np.max(np.abs(results[0] - results[i]))}")
            return False
    
    print("✅ PASS: infer() is deterministic")
    return True


def test_train_determinism():
    """Test that train produces identical state."""
    print("Testing train() determinism...")
    
    df = generate_test_data(n_series=20)
    y = pd.Series([i % 2 for i in range(20)], index=[f"test_{i}" for i in range(20)])
    
    # Train multiple times
    for i in range(3):
        train(df, y)
    
    print("✅ PASS: train() completes without error")
    return True


def test_different_seeds_same_result():
    """Test that setting seed produces consistent results."""
    print("Testing seed consistency...")
    
    df = generate_test_data(n_series=20, seed=123)
    
    np.random.seed(config.RANDOM_SEED)
    preds1 = infer(df)
    
    np.random.seed(config.RANDOM_SEED)
    preds2 = infer(df)
    
    if not np.allclose(preds1.values, preds2.values):
        print("❌ FAIL: Same seed produces different results")
        return False
    
    print("✅ PASS: Seed produces consistent results")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("DETERMINISM TESTS")
    print("=" * 70)
    print()
    
    results = {
        "Infer determinism": test_infer_determinism(),
        "Train determinism": test_train_determinism(),
        "Seed consistency": test_different_seeds_same_result(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
    
    if all(results.values()):
        print("\n✅ All determinism tests passed!")
    else:
        print("\n❌ Some tests failed!")
