import numpy as np
import pandas as pd
import sklearn.preprocessing as pp

from .compat import apply_pandas_finta_compatibility


def add_finta_feature(data: pd.DataFrame, data_finta: pd.DataFrame, feature_names, both_columns_features):
    apply_pandas_finta_compatibility()
    from finta import TA

    for feature_name in feature_names:
        feature_func = getattr(TA, feature_name)
        finta_feature = feature_func(data_finta)
        if getattr(finta_feature, 'ndim', 1) > 1:
            if feature_name in both_columns_features:
                data[f"{feature_name}_1"] = finta_feature.iloc[:, 0]
                data[f"{feature_name}_2"] = finta_feature.iloc[:, 1]
            else:
                data[feature_name] = finta_feature.iloc[:, 0]
        else:
            data[feature_name] = finta_feature
    return data


def build_finta_input(data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df['open'] = data['Opening_price']
    df['high'] = data['Highest_price']
    df['low'] = data['Lowest_price']
    df['close'] = data['Closing_price']
    df['volume'] = data['Volume_of_transactions']
    return df


def drop_until_last_nan_and_reorder(data_min: pd.DataFrame, new_columns_order):
    max_index = -np.inf
    for column in data_min.columns:
        nan_indices = data_min[column].index[data_min[column].apply(np.isnan)]
        max_index_column = np.max(nan_indices) if len(nan_indices) > 0 else -np.inf
        if max_index_column > max_index:
            max_index = max_index_column
    start_index = int(max_index) + 1 if np.isfinite(max_index) else 0
    data_min = data_min.iloc[start_index:, :].reset_index(drop=True)
    if 'Timestamp' in data_min.columns:
        data_min = data_min.drop(['Timestamp'], axis=1)
    data_min = data_min[new_columns_order]
    return data_min, start_index


def normalize_split_and_batch(data_min: pd.DataFrame, num_features: int, val_percentage: float, test_percentage: float,
                              scaler_name: str, train_batch_size: int, eval_batch_size: int, device) -> tuple:
    import torch
    # split
    data_length = len(data_min)
    val_length = int(data_length * val_percentage)
    test_length = int(data_length * test_percentage)
    train_length = data_length - val_length - test_length
    train_df = data_min.iloc[:train_length]
    val_df = data_min.iloc[train_length:train_length + val_length] if val_length > 0 else None
    test_df = data_min.iloc[train_length + val_length:]

    # scaler
    scaler = pp.StandardScaler() if scaler_name == 'standard' else pp.MinMaxScaler()
    train = train_df.iloc[:, :num_features].to_numpy()
    val = None if val_df is None else val_df.iloc[:, :num_features].to_numpy()
    test = test_df.iloc[:, :num_features].to_numpy()
    fitted = scaler.fit(train)
    train = torch.tensor(fitted.transform(train))
    val = None if val is None else torch.tensor(fitted.transform(val))
    test = torch.tensor(fitted.transform(test))

    # batchify
    def batchify(data, batch_size):
        seq_len = data.size(0) // batch_size
        data = data[:seq_len * batch_size, :]
        data = data.view(batch_size, seq_len, -1)
        data = torch.transpose(data, 0, 1).contiguous()
        return data.to(device)

    train_data = batchify(train, train_batch_size).float()
    val_data = None if val is None else batchify(val, eval_batch_size).float()
    test_data = batchify(test, eval_batch_size).float()
    return train_df, val_df, test_df, train_data, val_data, test_data, fitted


__all__ = [
    'add_finta_feature', 'build_finta_input', 'drop_until_last_nan_and_reorder', 'normalize_split_and_batch'
]


