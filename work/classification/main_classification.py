import torch
import sklearn.preprocessing as pp
import numpy as np
import pandas as pd

try:
    from ..config import (
        DATA_PATH, num_features, scaler_name, train_batch_size, eval_batch_size, epochs,
        num_encoder_layers, periodic_features, out_features, nhead, dim_feedforward, dropout, activation,
        gru_hidden_size, gru_num_layers, gru_dropout,
    )
    from ..compat import apply_pandas_finta_compatibility
    from ..data_loader import load_and_rename, sort_and_convert_timestamp
    from ..features import add_finta_feature, build_finta_input, drop_until_last_nan_and_reorder, normalize_split_and_batch
    from ..models import TransformerClassifier, GRUClassifier
    from .classification_data import create_classification_labels, prepare_classification_data, create_sequence_dataset
    from .train_classification import train_classifier, evaluate_classifier
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from work.config import (
        DATA_PATH, num_features, scaler_name, train_batch_size, eval_batch_size, epochs,
        num_encoder_layers, periodic_features, out_features, nhead, dim_feedforward, dropout, activation,
        gru_hidden_size, gru_num_layers, gru_dropout,
    )
    from work.compat import apply_pandas_finta_compatibility
    from work.data_loader import load_and_rename, sort_and_convert_timestamp
    from work.features import add_finta_feature, build_finta_input, drop_until_last_nan_and_reorder, normalize_split_and_batch
    from work.models import TransformerClassifier, GRUClassifier
    from work.classification.classification_data import create_classification_labels, prepare_classification_data, create_sequence_dataset
    from work.classification.train_classification import train_classifier, evaluate_classifier


def run_transformer_classification(device=None):
    """Execute Transformer classification task
    
    Args:
        device: Computing device (CPU/GPU)
    
    Returns:
        dict: Dictionary containing model results and metrics
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
    apply_pandas_finta_compatibility()

    # Load and process data
    data = load_and_rename(str(DATA_PATH))
    data = sort_and_convert_timestamp(data)

    data_min = data.copy()
    data_finta = build_finta_input(data_min)
    extra_features = ['TRIX', 'VWAP', 'MACD', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI', 'ADX', 'STOCHRSI',
                      'MI', 'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP', 'BASP', 'BASPN', 'WTO', 'SQZMI', 'VFI', 'STC']
    both_columns_features = ["DMI", "EBBP", "BASP", "BASPN"]
    data_min = add_finta_feature(data_min, data_finta, extra_features, both_columns_features)

    new_columns_order = ['Closing_price', 'Volume_of_transactions', 'Opening_price', 'Highest_price', 'Lowest_price','TRIX', 'VWAP', 'MACD',
                         'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI_1', 'DMI_2', 'ADX', 'STOCHRSI', 'MI', 'CHAIKIN', 
                         'VZO', 'PZO', 'EFI', 'EBBP_1', 'EBBP_2', 'BASP_1', 'BASP_2', 'BASPN_1', 'BASPN_2', 'WTO', 'SQZMI', 'VFI', 'STC']
    data_min, start_index = drop_until_last_nan_and_reorder(data_min, new_columns_order)

    # Create classification labels
    data_with_labels = create_classification_labels(data_min, 'Opening_price')

    # Split data, ensuring each dataset has enough data to create sequences
    seq_len = 10
    n = len(data_with_labels)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    # Ensure each dataset has at least seq_len samples
    train_end = max(train_end, seq_len)
    val_end = max(val_end, train_end + seq_len)
    
    train_df = data_with_labels.iloc[:train_end]
    val_df = data_with_labels.iloc[train_end:val_end] if val_end < n else data_with_labels.iloc[train_end:]
    test_df = data_with_labels.iloc[val_end:] if val_end < n else pd.DataFrame()

    # Prepare scaler
    scaler = pp.StandardScaler() if scaler_name == 'standard' else pp.MinMaxScaler()
    scaler.fit(train_df.iloc[:, :num_features])

    # Prepare datasets
    train_data = prepare_classification_data(train_df, num_features, scaler, device)
    val_data = prepare_classification_data(val_df, num_features, scaler, device) if len(val_df) > 0 else None
    test_data = prepare_classification_data(test_df, num_features, scaler, device) if len(test_df) > 0 else None

    # Convert to sequence format
    train_seq = create_sequence_dataset(train_data, seq_len)
    val_seq = create_sequence_dataset(val_data, seq_len) if val_data else None
    test_seq = create_sequence_dataset(test_data, seq_len) if test_data else None

    # Build model
    model = TransformerClassifier(
        num_encoder_layers=num_encoder_layers,
        in_features=num_features,
        periodic_features=periodic_features,
        out_features=out_features,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    ).to(device)

    # Training
    best_model, train_loss_hist, valid_acc_hist = train_classifier(
        model, train_seq, val_seq, epochs=epochs, lr=0.001, batch_size=train_batch_size, device=device
    )

    # Evaluation
    if test_seq is not None:
        test_acc, test_f1, predictions, targets = evaluate_classifier(best_model, test_seq, device, eval_batch_size)
        print(f'Test Accuracy: {test_acc:.2f}%')
        print(f'Test F1 Score: {test_f1:.4f}')
    else:
        test_acc, test_f1, predictions, targets = 0.0, 0.0, np.array([]), np.array([])
        print('No test data available')

    return {
        'model_name': 'Transformer',
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'train_loss': train_loss_hist,
        'valid_accuracy': valid_acc_hist,
        'predictions': predictions,
        'targets': targets,
    }


def run_gru_classification(device=None):
    """Execute GRU classification task
    
    Args:
        device: Computing device (CPU/GPU)
    
    Returns:
        dict: Dictionary containing model results and metrics
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
    apply_pandas_finta_compatibility()

    # Load and process data (same as Transformer)
    data = load_and_rename(str(DATA_PATH))
    data = sort_and_convert_timestamp(data)

    data_min = data.copy()
    data_finta = build_finta_input(data_min)
    extra_features = ['TRIX', 'VWAP', 'MACD', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI', 'ADX', 'STOCHRSI',
                      'MI', 'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP', 'BASP', 'BASPN', 'WTO', 'SQZMI', 'VFI', 'STC']
    both_columns_features = ["DMI", "EBBP", "BASP", "BASPN"]
    data_min = add_finta_feature(data_min, data_finta, extra_features, both_columns_features)

    new_columns_order = ['Closing_price', 'Volume_of_transactions', 'Opening_price', 'Highest_price', 'Lowest_price','TRIX', 'VWAP', 'MACD',
                         'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI_1', 'DMI_2', 'ADX', 'STOCHRSI', 'MI', 'CHAIKIN', 
                         'VZO', 'PZO', 'EFI', 'EBBP_1', 'EBBP_2', 'BASP_1', 'BASP_2', 'BASPN_1', 'BASPN_2', 'WTO', 'SQZMI', 'VFI', 'STC']
    data_min, start_index = drop_until_last_nan_and_reorder(data_min, new_columns_order)

    # Create classification labels
    data_with_labels = create_classification_labels(data_min, 'Opening_price')

    # Split data, ensuring each dataset has enough data to create sequences
    seq_len = 10
    n = len(data_with_labels)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    # Ensure each dataset has at least seq_len samples
    train_end = max(train_end, seq_len)
    val_end = max(val_end, train_end + seq_len)
    
    train_df = data_with_labels.iloc[:train_end]
    val_df = data_with_labels.iloc[train_end:val_end] if val_end < n else data_with_labels.iloc[train_end:]
    test_df = data_with_labels.iloc[val_end:] if val_end < n else pd.DataFrame()

    # Prepare scaler
    scaler = pp.StandardScaler() if scaler_name == 'standard' else pp.MinMaxScaler()
    scaler.fit(train_df.iloc[:, :num_features])

    # Prepare datasets
    train_data = prepare_classification_data(train_df, num_features, scaler, device)
    val_data = prepare_classification_data(val_df, num_features, scaler, device) if len(val_df) > 0 else None
    test_data = prepare_classification_data(test_df, num_features, scaler, device) if len(test_df) > 0 else None

    # Convert to sequence format
    train_seq = create_sequence_dataset(train_data, seq_len)
    val_seq = create_sequence_dataset(val_data, seq_len) if val_data else None
    test_seq = create_sequence_dataset(test_data, seq_len) if test_data else None

    # Build model
    model = GRUClassifier(
        in_features=num_features,
        hidden_size=gru_hidden_size,
        num_layers=gru_num_layers,
        dropout=gru_dropout,
    ).to(device)

    # Training
    best_model, train_loss_hist, valid_acc_hist = train_classifier(
        model, train_seq, val_seq, epochs=epochs, lr=0.001, batch_size=train_batch_size, device=device
    )

    # Evaluation
    if test_seq is not None:
        test_acc, test_f1, predictions, targets = evaluate_classifier(best_model, test_seq, device, eval_batch_size)
        print(f'Test Accuracy: {test_acc:.2f}%')
        print(f'Test F1 Score: {test_f1:.4f}')
    else:
        test_acc, test_f1, predictions, targets = 0.0, 0.0, np.array([]), np.array([])
        print('No test data available')

    return {
        'model_name': 'GRU',
        'test_accuracy': test_acc,
        'train_loss': train_loss_hist,
        'valid_accuracy': valid_acc_hist,
        'predictions': predictions,
        'targets': targets,
    }


def run_both_classifications(device=None):
    """Execute both classification models and compare results
    
    Args:
        device: Computing device (CPU/GPU)
    
    Returns:
        tuple: (transformer_results, gru_results)
    """
    print("Training Transformer Classifier...")
    tfm_result = run_transformer_classification(device)
    
    print("\nTraining GRU Classifier...")
    gru_result = run_gru_classification(device)
    
    print(f"\nClassification Results:")
    print(f"Transformer Test Accuracy: {tfm_result['test_accuracy']:.2f}%")
    print(f"Transformer Test F1 Score: {tfm_result['test_f1']:.4f}")
    print(f"GRU Test Accuracy: {gru_result['test_accuracy']:.2f}%")
    print(f"GRU Test F1 Score: {gru_result['test_f1']:.4f}")
    
    return tfm_result, gru_result


if __name__ == '__main__':
    run_both_classifications()
