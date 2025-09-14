import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def create_classification_labels(data: pd.DataFrame, open_price_col: str = 'Opening_price') -> pd.DataFrame:
    """Calculate returns based on opening price and generate classification labels
    
    Args:
        data: DataFrame containing price data
        open_price_col: Name of the opening price column
    
    Returns:
        DataFrame containing return and label columns
    """
    df = data.copy()
    
    # Calculate return = Open_{t+1} / Open_t - 1
    df['return'] = df[open_price_col].shift(-1) / df[open_price_col] - 1
    
    # Generate labels: 1 for positive return, -1 for negative return
    df['label'] = np.where(df['return'] > 0, 1, -1)
    
    # Remove the last row (no future price available)
    df = df[:-1]
    
    return df


def prepare_classification_data(data: pd.DataFrame, num_features: int, scaler, device) -> TensorDataset:
    """Prepare data for classification task
    
    Args:
        data: DataFrame containing features and labels
        num_features: Number of features to use
        scaler: Fitted scaler for normalization
        device: Computing device
    
    Returns:
        TensorDataset containing (features, labels)
    """
    # Extract features and labels
    features = data.iloc[:, :num_features].values
    labels = data['label'].values
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Convert to tensors
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    # Convert labels from [-1, 1] to [0, 1] for cross-entropy loss
    labels_tensor = (labels_tensor + 1) // 2  # -1 -> 0, 1 -> 1
    
    return TensorDataset(features_tensor, labels_tensor)


def create_sequence_dataset(dataset: TensorDataset, seq_len: int = 10) -> TensorDataset:
    """Convert data to sequence format for classification
    
    Args:
        dataset: Original TensorDataset
        seq_len: Sequence length
    
    Returns:
        TensorDataset in sequence format
    """
    features, labels = dataset.tensors
    
    # Create sequences
    sequences = []
    sequence_labels = []
    
    for i in range(len(features) - seq_len + 1):
        seq = features[i:i + seq_len]  # (seq_len, features)
        label = labels[i + seq_len - 1]  # Use label from the last time step of the sequence
        
        sequences.append(seq)
        sequence_labels.append(label)
    
    if len(sequences) == 0:
        # If there is not enough data to create sequences, return empty dataset
        empty_features = torch.empty(0, seq_len, features.shape[1])  # (0, seq_len, features)
        empty_labels = torch.empty(0, dtype=labels.dtype)
        return TensorDataset(empty_features, empty_labels)
    
    sequences_tensor = torch.stack(sequences)  # (n_sequences, seq_len, features)
    labels_tensor = torch.stack(sequence_labels)  # (n_sequences,)
    
    # Keep (n_sequences, seq_len, features) shape for DataLoader to handle batching
    return TensorDataset(sequences_tensor, labels_tensor)


__all__ = ["create_classification_labels", "prepare_classification_data", "create_sequence_dataset"]
