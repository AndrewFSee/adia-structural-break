"""
Data loading utilities for CrunchDAO structural break dataset.

The dataset has the following structure:
- X_train.parquet: Training data (10,000 time series)
- y_train.parquet: Training labels
- X_test.reduced.parquet: Test data (100 time series)
- y_test.reduced.parquet: Test labels

X format:
- MultiIndex: (id, time)
- Columns: value, period
  - period=0: before break point
  - period=1: after break point

y format:
- Index: id
- Column: structural_breakpoint (bool: True=break, False=no break)
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def load_crunchdao_data(
    data_dir: str = "data",
    subset: str = "train"
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load CrunchDAO structural break dataset.
    
    Args:
        data_dir: Directory containing the parquet files
        subset: "train" or "test"
        
    Returns:
        (X, y) where:
        - X: DataFrame with MultiIndex (id, time), columns (value, period)
        - y: Series with index=id, values=structural_breakpoint (or None if test)
    """
    data_path = Path(data_dir)
    
    if subset == "train":
        X = pd.read_parquet(data_path / "X_train.parquet")
        y = pd.read_parquet(data_path / "y_train.parquet")["structural_breakpoint"]
        # Convert boolean to int (0/1) for consistency
        y = y.astype(int)
    elif subset == "test":
        X = pd.read_parquet(data_path / "X_test.reduced.parquet")
        # Try to load test labels if they exist
        y_path = data_path / "y_test.reduced.parquet"
        if y_path.exists():
            y = pd.read_parquet(y_path)["structural_breakpoint"]
            y = y.astype(int)
        else:
            y = None
    else:
        raise ValueError(f"subset must be 'train' or 'test', got '{subset}'")
    
    return X, y


def convert_to_standard_format(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CrunchDAO format to our standard format.
    
    CrunchDAO format:
    - MultiIndex: (id, time)
    - Columns: value, period
    
    Our standard format:
    - Flat DataFrame
    - Columns: id, time, value, period
    
    Args:
        X: DataFrame in CrunchDAO format
        
    Returns:
        DataFrame in standard format
    """
    # Reset the MultiIndex to make id and time regular columns
    df = X.reset_index()
    return df


def load_for_training(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data in our standard format.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        (df, y) where:
        - df: DataFrame with columns [id, time, value, period]
        - y: Series with index=id, values=label (0/1)
    """
    X, y = load_crunchdao_data(data_dir, subset="train")
    df = convert_to_standard_format(X)
    return df, y


def load_for_testing(
    data_dir: str = "data",
    with_labels: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load test data in our standard format.
    
    Args:
        data_dir: Directory containing data files
        with_labels: Whether to try loading test labels
        
    Returns:
        (df, y) where:
        - df: DataFrame with columns [id, time, value, period]
        - y: Series with index=id, values=label (0/1), or None
    """
    X, y = load_crunchdao_data(data_dir, subset="test")
    df = convert_to_standard_format(X)
    
    if not with_labels:
        y = None
    
    return df, y


def get_dataset_stats(data_dir: str = "data") -> dict:
    """
    Get statistics about the dataset.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with dataset statistics
    """
    X_train, y_train = load_crunchdao_data(data_dir, "train")
    X_test, y_test = load_crunchdao_data(data_dir, "test")
    
    stats = {
        "train": {
            "n_series": len(X_train.index.get_level_values("id").unique()),
            "total_points": len(X_train),
            "n_breaks": y_train.sum() if y_train is not None else None,
            "break_rate": y_train.mean() if y_train is not None else None,
        },
        "test": {
            "n_series": len(X_test.index.get_level_values("id").unique()),
            "total_points": len(X_test),
            "n_breaks": y_test.sum() if y_test is not None else None,
            "break_rate": y_test.mean() if y_test is not None else None,
        }
    }
    
    return stats
