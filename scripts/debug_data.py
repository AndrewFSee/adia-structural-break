"""
Check for data quality issues that might explain low AUC.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sb import data_loader, io

print("Loading data...")
df, y = data_loader.load_for_training("data")

print(f"\n{'='*70}")
print("DATA QUALITY CHECK")
print(f"{'='*70}")

print(f"\nBasic Stats:")
print(f"  Total series: {len(y)}")
print(f"  Breaks (label=1): {y.sum()} ({y.mean():.1%})")
print(f"  No breaks (label=0): {(1-y).sum()} ({(1-y).mean():.1%})")

print(f"\nSeries Length Stats:")
lengths_per_series = df.groupby('id').size()
print(f"  Mean: {lengths_per_series.mean():.1f}")
print(f"  Min: {lengths_per_series.min()}")
print(f"  Max: {lengths_per_series.max()}")
print(f"  Median: {lengths_per_series.median():.1f}")

print(f"\nValue Stats (per period):")
for period in [0, 1]:
    period_data = df[df['period'] == period]['value']
    print(f"  Period {period}:")
    print(f"    Mean: {period_data.mean():.4f}")
    print(f"    Std: {period_data.std():.4f}")
    print(f"    Min: {period_data.min():.4f}")
    print(f"    Max: {period_data.max():.4f}")
    print(f"    NaN count: {period_data.isna().sum()}")

print(f"\nChecking for obvious patterns:")

# Check if breaks correlate with simple statistics
print(f"\nPeriod 1 mean by label:")
for label in [0, 1]:
    ids = y[y == label].index
    period1_data = df[(df['id'].isin(ids)) & (df['period'] == 1)]['value']
    print(f"  Label {label}: mean={period1_data.mean():.4f}, std={period1_data.std():.4f}")

# Check if break series are different in obvious ways
print(f"\nSeries characteristics by label:")
for label in [0, 1]:
    ids = y[y == label].index
    subset = df[df['id'].isin(ids)]
    
    # Mean absolute value
    mean_abs = subset.groupby('id')['value'].apply(lambda x: np.abs(x).mean()).mean()
    
    # Variance
    mean_var = subset.groupby('id')['value'].var().mean()
    
    print(f"  Label {label}:")
    print(f"    Mean abs value: {mean_abs:.4f}")
    print(f"    Mean variance: {mean_var:.4f}")

print(f"\n{'='*70}")
print("DIAGNOSIS")
print(f"{'='*70}")

# Check for class imbalance
if y.mean() < 0.1 or y.mean() > 0.9:
    print("\n⚠️  Severe class imbalance detected!")
    print("   This makes the problem harder.")
elif y.mean() < 0.2 or y.mean() > 0.8:
    print("\n⚠️  Moderate class imbalance detected.")
    print("   Consider stratified sampling.")
else:
    print("\n✅ Class balance is reasonable.")

# Check for series length
if lengths_per_series.min() < 100:
    print(f"\n⚠️  Some series are very short (min={lengths_per_series.min()})!")
    print("   Short series may not have enough data for break detection.")
else:
    print(f"\n✅ Series lengths are adequate (min={lengths_per_series.min()}).")

# Check for data range issues
all_values = df['value'].values
if np.abs(all_values).max() < 1e-3:
    print("\n⚠️  Values are very small!")
    print("   This might cause numerical issues.")
elif np.abs(all_values).max() > 1e6:
    print("\n⚠️  Values are very large!")
    print("   This might cause numerical issues.")
else:
    print("\n✅ Value range is reasonable.")

print(f"\n{'='*70}")
