"""
Inspect the CrunchDAO dataset.

Run this to understand your data before training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sb import data_loader


def main():
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)
    
    # Get dataset statistics
    print("\nLoading dataset statistics...")
    stats = data_loader.get_dataset_stats("data")
    
    print("\n" + "=" * 70)
    print("TRAINING SET")
    print("=" * 70)
    print(f"Number of series:     {stats['train']['n_series']:,}")
    print(f"Total data points:    {stats['train']['total_points']:,}")
    print(f"Series with breaks:   {stats['train']['n_breaks']:,}")
    print(f"Break rate:           {stats['train']['break_rate']:.1%}")
    
    print("\n" + "=" * 70)
    print("TEST SET")
    print("=" * 70)
    print(f"Number of series:     {stats['test']['n_series']:,}")
    print(f"Total data points:    {stats['test']['total_points']:,}")
    if stats['test']['n_breaks'] is not None:
        print(f"Series with breaks:   {stats['test']['n_breaks']:,}")
        print(f"Break rate:           {stats['test']['break_rate']:.1%}")
    else:
        print("Labels not available")
    
    # Load a sample
    print("\n" + "=" * 70)
    print("SAMPLE DATA (first series)")
    print("=" * 70)
    
    df_train, y_train = data_loader.load_for_training("data")
    
    # Show first series
    first_id = df_train["id"].iloc[0]
    sample = df_train[df_train["id"] == first_id]
    
    print(f"\nSeries ID: {first_id}")
    print(f"Label (has break): {y_train.loc[first_id]}")
    print(f"Total points: {len(sample)}")
    
    period_0 = sample[sample["period"] == 0]
    period_1 = sample[sample["period"] == 1]
    
    print(f"\nPeriod 0 (pre-break):")
    print(f"  Points: {len(period_0)}")
    print(f"  Value range: [{period_0['value'].min():.4f}, {period_0['value'].max():.4f}]")
    print(f"  Mean: {period_0['value'].mean():.4f}")
    print(f"  Std: {period_0['value'].std():.4f}")
    
    print(f"\nPeriod 1 (post-break):")
    print(f"  Points: {len(period_1)}")
    print(f"  Value range: [{period_1['value'].min():.4f}, {period_1['value'].max():.4f}]")
    print(f"  Mean: {period_1['value'].mean():.4f}")
    print(f"  Std: {period_1['value'].std():.4f}")
    
    print("\nFirst few rows:")
    print(sample.head(10))
    
    print("\n" + "=" * 70)
    print("DATA FORMAT")
    print("=" * 70)
    print("\nColumns:", list(df_train.columns))
    print("Shape:", df_train.shape)
    print("\nLabel format:")
    print(f"  Index: {y_train.index.name}")
    print(f"  Values: {y_train.name if hasattr(y_train, 'name') else 'Series'}")
    print(f"  Unique values: {sorted(y_train.unique())}")


if __name__ == "__main__":
    main()
