import numpy as np
import pandas as pd


def load_and_rename(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
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
    data = data.sort_values('Unix_timestamp', ignore_index=True)
    data['Unix_timestamp'] = pd.to_datetime(data['Unix_timestamp'], unit='ms') + pd.Timedelta('08:00:00')
    data = data.rename(columns={'Unix_timestamp': 'Timestamp'})
    return data


__all__ = ["load_and_rename", "sort_and_convert_timestamp"]


