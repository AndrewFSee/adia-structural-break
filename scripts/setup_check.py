"""
Quick initialization script to verify the setup.
Run this after installation to ensure everything works.
"""

import sys
from pathlib import Path

print("=" * 70)
print("ADIA STRUCTURAL BREAK - SETUP VERIFICATION")
print("=" * 70)

# Test imports
print("\n1. Testing imports...")
try:
    import numpy as np
    print("   ✅ numpy")
except ImportError as e:
    print(f"   ❌ numpy: {e}")

try:
    import pandas as pd
    print("   ✅ pandas")
except ImportError as e:
    print(f"   ❌ pandas: {e}")

try:
    import sklearn
    print("   ✅ scikit-learn")
except ImportError as e:
    print(f"   ❌ scikit-learn: {e}")

try:
    import lightgbm
    print("   ✅ lightgbm")
except ImportError as e:
    print(f"   ❌ lightgbm: {e}")

# Test package
print("\n2. Testing sb package...")
try:
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    import sb
    print(f"   ✅ sb package (version {sb.__version__})")
    
    from sb import config, io, preprocessing, features, models, cv, pipeline
    print("   ✅ All modules imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test solution.py
print("\n3. Testing solution.py...")
try:
    from solution import train, infer
    print("   ✅ train() and infer() functions found")
except ImportError as e:
    print(f"   ❌ Solution import failed: {e}")
    sys.exit(1)

# Test with synthetic data
print("\n4. Running quick functional test...")
try:
    np.random.seed(42)
    
    # Generate tiny synthetic dataset
    data = []
    for i in range(5):
        series_id = f"test_{i}"
        x0 = np.random.randn(50)
        x1 = np.random.randn(50)
        
        for t, val in enumerate(x0):
            data.append({"id": series_id, "period": 0, "time": t, "value": val})
        for t, val in enumerate(x1):
            data.append({"id": series_id, "period": 1, "time": t, "value": val})
    
    df = pd.DataFrame(data)
    
    # Test infer
    predictions = infer(df)
    
    assert len(predictions) == 5, "Wrong number of predictions"
    assert predictions.min() >= 0, "Predictions below 0"
    assert predictions.max() <= 1, "Predictions above 1"
    assert not predictions.isna().any(), "Missing predictions"
    
    print("   ✅ Functional test passed")
    print(f"   Sample predictions: {predictions.values[:3]}")
    
except Exception as e:
    print(f"   ❌ Functional test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ SETUP VERIFICATION COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("1. Read QUICKSTART.md for usage instructions")
print("2. Run: python scripts/sanity_check.py")
print("3. Prepare your data and run: python scripts/train_local.py --data train.csv")
print("\nYou're ready to start! 🚀")
