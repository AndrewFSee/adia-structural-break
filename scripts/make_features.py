"""
Feature extraction helper - compute and cache features for analysis.

Use this to compute features once and analyze them without recomputing.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sb import io, features


def main():
    parser = argparse.ArgumentParser(description="Extract and cache features")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features.csv",
        help="Path to save features (default: features.csv)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FEATURE EXTRACTION")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    df = io.load_data(args.data)
    print(f"Loaded {df['id'].nunique()} time series")
    
    # Compute features
    print("\nComputing features...")
    feature_df = features.base.compute_features(df)
    
    print(f"\nFeature summary:")
    print(feature_df.describe())
    
    # Save features
    print(f"\nSaving features to {args.output}...")
    feature_df.to_csv(args.output)
    
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Features saved to: {args.output}")
    print(f"Feature shape: {feature_df.shape}")
    print(f"Feature columns: {list(feature_df.columns)}")


if __name__ == "__main__":
    main()
