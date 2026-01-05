"""
I/O utilities for loading and segmenting time series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Iterator, Dict
from . import config


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame has required columns and structure.
    
    Args:
        df: Input DataFrame (can be MultiIndex with (id, time) or columns)
        
    Raises:
        ValueError: If validation fails
    """
    # Handle MultiIndex format (id, time) -> convert to columns for validation
    if isinstance(df.index, pd.MultiIndex):
        if df.index.names != ['id', 'time']:
            raise ValueError(f"Expected MultiIndex with names ['id', 'time'], got {df.index.names}")
        # Check required columns in values
        required = {'value', 'period'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    else:
        # Standard column format
        missing = set(config.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    if not df["period"].isin(config.EXPECTED_PERIODS).all():
        raise ValueError(f"Period column must contain only {config.EXPECTED_PERIODS}")


def split_series(df_id: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Canonical segmentation function.
    Extract pre-break (period=0) and post-break (period=1) values.
    
    ⚠️ Do not trust time ordering blindly — always rely on period.
    
    Args:
        df_id: DataFrame for a single id (can have MultiIndex or regular index)
        
    Returns:
        x0: Pre-break values (period=0)
        x1: Post-break values (period=1)
    """
    # Handle both MultiIndex and regular DataFrame
    x0 = df_id.loc[df_id["period"] == 0, "value"].values
    x1 = df_id.loc[df_id["period"] == 1, "value"].values
    return x0, x1


def iter_series(df: pd.DataFrame) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Iterate over all time series in the dataset.
    
    Args:
        df: Full DataFrame with multiple ids (MultiIndex or regular)
        
    Yields:
        (id, x0, x1) tuples where:
            id: series identifier
            x0: pre-break values
            x1: post-break values
    """
    # Handle MultiIndex format
    if isinstance(df.index, pd.MultiIndex):
        # Group by first level of index (id)
        for series_id in df.index.get_level_values(0).unique():
            df_group = df.loc[series_id]
            x0, x1 = split_series(df_group)
            yield series_id, x0, x1
    else:
        # Regular format with 'id' column
        for series_id, df_group in df.groupby("id", sort=False):
            x0, x1 = split_series(df_group)
            yield series_id, x0, x1


def load_data(filepath: str, validate: bool = True) -> pd.DataFrame:
    """
    Load time series data from CSV or Parquet.
    
    Args:
        filepath: Path to CSV or Parquet file
        validate: Whether to validate DataFrame structure
        
    Returns:
        Loaded DataFrame
    """
    # Auto-detect format from extension
    if filepath.lower().endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        # Try CSV as default
        df = pd.read_csv(filepath)
    
    if validate:
        validate_dataframe(df)
    
    return df


def load_train(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data from directory.
    
    Args:
        data_dir: Directory containing X_train.parquet and y_train.parquet
        
    Returns:
        X_train: Feature DataFrame with columns [id, time, value, period]
        y_train: Target Series with index=id
    """
    from pathlib import Path
    
    data_path = Path(data_dir)
    
    X_train = pd.read_parquet(data_path / "X_train.parquet")
    y_train = pd.read_parquet(data_path / "y_train.parquet").squeeze()
    
    # Validate
    validate_dataframe(X_train)
    
    return X_train, y_train


def load_test(data_dir: str = "data") -> pd.DataFrame:
    """
    Load test data from directory.
    
    Args:
        data_dir: Directory containing X_test.parquet (or X_test.reduced.parquet)
        
    Returns:
        X_test: Feature DataFrame with columns [id, time, value, period]
    """
    from pathlib import Path
    
    data_path = Path(data_dir)
    
    # Try full test file first, fall back to reduced
    if (data_path / "X_test.parquet").exists():
        X_test = pd.read_parquet(data_path / "X_test.parquet")
    elif (data_path / "X_test.reduced.parquet").exists():
        X_test = pd.read_parquet(data_path / "X_test.reduced.parquet")
    else:
        raise FileNotFoundError(f"No test file found in {data_dir}")
    
    # Validate
    validate_dataframe(X_test)
    
    return X_test


def save_predictions(predictions: pd.Series, filepath: str) -> None:
    """
    Save predictions to CSV in submission format.
    
    Args:
        predictions: Series with index=id, values=prediction
        filepath: Output path
    """
    predictions.name = "prediction"
    predictions.index.name = "id"
    predictions.to_csv(filepath, header=True)
