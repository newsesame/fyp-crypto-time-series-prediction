"""Data loading and preprocessing utilities

This module provides functions for loading Bitcoin price data from CSV files
and performing basic preprocessing operations.
"""

import numpy as np
import pandas as pd


def load_and_rename(csv_path: str) -> pd.DataFrame:
    """Load CSV data and rename columns to more descriptive names
    
    Args:
        csv_path: Path to the CSV file containing Bitcoin price data
    
    Returns:
        DataFrame with renamed columns
    """
    data = pd.read_csv(csv_path)
    # Map abbreviated column names to descriptive names
    columns_dict = {
        't': 'Unix_timestamp',
        'o': 'Opening_price',
        'h': 'Highest_price',
        'l': 'Lowest_price',
        'c': 'Closing_price',
        'v': 'Volume_of_transactions',
    }
    return data.rename(columns=columns_dict)


def sort_and_convert_timestamp(data: pd.DataFrame) -> pd.DataFrame:
    """Sort data by timestamp and convert Unix timestamp to datetime
    
    Args:
        data: DataFrame with Unix_timestamp column
    
    Returns:
        DataFrame with sorted data and converted timestamp
    """
    # Sort by timestamp to ensure chronological order
    data = data.sort_values('Unix_timestamp', ignore_index=True)
    # Convert Unix timestamp (milliseconds) to datetime and adjust for timezone
    data['Unix_timestamp'] = pd.to_datetime(data['Unix_timestamp'], unit='ms') + pd.Timedelta('08:00:00')
    # Rename to more descriptive column name
    data = data.rename(columns={'Unix_timestamp': 'Timestamp'})
    return data


__all__ = ["load_and_rename", "sort_and_convert_timestamp"]


