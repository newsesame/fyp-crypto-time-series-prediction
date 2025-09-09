import pandas as pd
import torch

try:
    from .main_classification import run_transformer_classification, run_gru_classification
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from work.classification.main_classification import run_transformer_classification, run_gru_classification


def evaluate_classification_models(device=None) -> pd.DataFrame:
    """Evaluate classification models and return comparison table
    
    Args:
        device: Computing device (CPU/GPU)
    
    Returns:
        dict: Dictionary containing model_info and metrics DataFrames
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device

    print("Training Transformer Classifier...")
    tfm_result = run_transformer_classification(device=device)
    
    print("\nTraining GRU Classifier...")
    gru_result = run_gru_classification(device=device)

    # Get model configuration information
    try:
        from ..config import (
            num_encoder_layers, nhead, dim_feedforward, dropout,
            gru_hidden_size, gru_num_layers, gru_dropout,
            epochs, lr, train_batch_size, num_features, scaler_name
        )
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from work.config import (
            num_encoder_layers, nhead, dim_feedforward, dropout,
            gru_hidden_size, gru_num_layers, gru_dropout,
            epochs, lr, train_batch_size, num_features, scaler_name
        )

    # Model information table
    model_info_rows = [
        {
            'Model': 'Transformer',
            'Architecture': f'Enc:{num_encoder_layers}, Heads:{nhead}, FF:{dim_feedforward}, Dropout:{dropout}',
            'Training': f'Epochs:{epochs}, LR:{lr}, Batch:{train_batch_size}',
            'Features': f'Features:{num_features}, Scaler:{scaler_name}',
        },
        {
            'Model': 'GRU',
            'Architecture': f'Hidden:{gru_hidden_size}, Layers:{gru_num_layers}, Dropout:{gru_dropout}',
            'Training': f'Epochs:{epochs}, LR:{lr}, Batch:{train_batch_size}',
            'Features': f'Features:{num_features}, Scaler:{scaler_name}',
        },
    ]
    
    # Performance metrics table
    metrics_rows = [
        {
            'Model': 'Transformer',
            'Test Accuracy (%)': tfm_result['test_accuracy'],
            'Test F1 Score': tfm_result['test_f1'],
        },
        {
            'Model': 'GRU',
            'Test Accuracy (%)': gru_result['test_accuracy'],
            'Test F1 Score': gru_result['test_f1'],
        },
    ]
    
    return {
        'model_info': pd.DataFrame(model_info_rows),
        'metrics': pd.DataFrame(metrics_rows)
    }


def main():
    results = evaluate_classification_models()
    
    print("\nClassification Model Comparison\n")
    print("Model Information:")
    print(results['model_info'].to_string(index=False))
    
    print("\nModel Performance:")
    print(results['metrics'].to_string(index=False))
    
    # Find the best models
    metrics_df = results['metrics']
    best_acc_idx = metrics_df['Test Accuracy (%)'].idxmax()
    best_f1_idx = metrics_df['Test F1 Score'].idxmax()
    
    print(f"\nBest Accuracy: {metrics_df.loc[best_acc_idx, 'Model']} ({metrics_df.loc[best_acc_idx, 'Test Accuracy (%)']:.2f}%)")
    print(f"Best F1 Score: {metrics_df.loc[best_f1_idx, 'Model']} ({metrics_df.loc[best_f1_idx, 'Test F1 Score']:.4f})")


if __name__ == '__main__':
    main()
