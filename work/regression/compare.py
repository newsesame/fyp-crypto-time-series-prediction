import math
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

try:
    from .main import run as run_transformer
    from .main_gru import run as run_gru
    from .plotting import (
        plot_gru_vs_real, plot_transformer_vs_real, plot_all_models_comparison,
        plot_zoom_in_comparison, generate_all_plots
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from work.regression.main import run as run_transformer
    from work.regression.main_gru import run as run_gru
    from work.regression.plotting import (
        plot_gru_vs_real, plot_transformer_vs_real, plot_all_models_comparison,
        plot_zoom_in_comparison, generate_all_plots
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = min(len(y_true), len(y_pred))
    y_true = np.asarray(y_true[:n], dtype=float)
    y_pred = np.asarray(y_pred[:n], dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if n > 1 else float('nan')
    return {"MAE": mae, "MSE": mse, "CORR": corr}


def evaluate_models(device=None, return_raw_results=False) -> pd.DataFrame:
    """Evaluate both GRU and Transformer models and return comparison results
    
    Args:
        device: Computing device (CPU/GPU)
        return_raw_results: If True, also return raw model results for plotting
    
    Returns:
        dict: Dictionary containing model comparison results and optionally raw results
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device

    tfm = run_transformer(device=device)
    gru = run_gru(device=device)

    tfm_metrics = _metrics(tfm['feature_real'], tfm['feature_pred'])
    gru_metrics = _metrics(gru['feature_real'], gru['feature_pred'])

    # Import config for model parameters
    try:
        from ..config import (
            num_encoder_layers, num_decoder_layers, nhead, dim_feedforward,
            gru_hidden_size, gru_num_layers, gru_dropout,
            epochs, lr, train_batch_size, bptt_src, bptt_tgt, num_features, scaler_name
        )
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from work.config import (
            num_encoder_layers, num_decoder_layers, nhead, dim_feedforward,
            gru_hidden_size, gru_num_layers, gru_dropout,
            epochs, lr, train_batch_size, bptt_src, bptt_tgt, num_features, scaler_name
        )

    # Model information table
    model_info_rows = [
        {
            'Model': 'Transformer',
            'Architecture': f'Enc:{num_encoder_layers}, Dec:{num_decoder_layers}, Heads:{nhead}, FF:{dim_feedforward}',
            'Training': f'Epochs:{epochs}, LR:{lr}, Batch:{train_batch_size}',
            'Sequence': f'Src:{bptt_src}, Tgt:{bptt_tgt}, Features:{num_features}',
            'Scaler': scaler_name,
        },
        {
            'Model': 'GRU',
            'Architecture': f'Hidden:{gru_hidden_size}, Layers:{gru_num_layers}, Dropout:{gru_dropout}',
            'Training': f'Epochs:{epochs}, LR:{lr}, Batch:{train_batch_size}',
            'Sequence': f'Src:{bptt_src}, Tgt:{bptt_tgt}, Features:{num_features}',
            'Scaler': scaler_name,
        },
    ]
    
    # Performance metrics table
    metrics_rows = [
        {
            'Model': 'Transformer',
            'MAE': tfm_metrics['MAE'],
            'MSE': tfm_metrics['MSE'],
            'CORR': tfm_metrics['CORR'],
        },
        {
            'Model': 'GRU',
            'MAE': gru_metrics['MAE'],
            'MSE': gru_metrics['MSE'],
            'CORR': gru_metrics['CORR'],
        },
    ]
    
    result = {
        'model_info': pd.DataFrame(model_info_rows),
        'metrics': pd.DataFrame(metrics_rows)
    }
    
    if return_raw_results:
        result['raw_results'] = {
            'transformer': tfm,
            'gru': gru
        }
    
    return result


def main(generate_plots=True, save_plots=True, output_dir="result_images"):
    """Main function to run model comparison and optionally generate plots
    
    Args:
        generate_plots: Whether to generate comparison plots
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots
    """
    print("=" * 80)
    print("Bitcoin Price Prediction Model Comparison".center(80))
    print("=" * 80)
    
    # Get results with raw data for plotting
    results = evaluate_models(return_raw_results=generate_plots)
    
    print("\n" + "=" * 80)
    print("Model Comparison Results".center(80))
    print("=" * 80)
    
    print("\nModel Configuration:")
    print(results['model_info'].to_string(index=False))
    
    print("\nModel Performance:")
    print(results['metrics'].to_string(index=False))
    
    # Simple best model analysis
    tfm_mae = results['metrics'].iloc[0]['MAE']
    gru_mae = results['metrics'].iloc[1]['MAE']
    tfm_mse = results['metrics'].iloc[0]['MSE']
    gru_mse = results['metrics'].iloc[1]['MSE']
    tfm_corr = results['metrics'].iloc[0]['CORR']
    gru_corr = results['metrics'].iloc[1]['CORR']
    
    print("\nBest Models:")
    print(f"Best MAE: {'Transformer' if tfm_mae < gru_mae else 'GRU'}")
    print(f"Best MSE: {'Transformer' if tfm_mse < gru_mse else 'GRU'}")
    print(f"Best CORR: {'Transformer' if tfm_corr > gru_corr else 'GRU'}")
    
    # Generate plots if requested
    if generate_plots and 'raw_results' in results:
        print("\n" + "=" * 80)
        print("Generating Comparison Plots".center(80))
        print("=" * 80)
        
        raw_results = results['raw_results']
        
        if save_plots:
            # Generate all plots and save them
            generate_all_plots(
                gru_results=raw_results['gru'],
                transformer_results=raw_results['transformer'],
                output_dir=output_dir
            )
        else:
            # Just display plots without saving
            print("Displaying GRU vs Real Values plot...")
            plot_gru_vs_real(raw_results['gru'])
            
            print("Displaying Transformer vs Real Values plot...")
            plot_transformer_vs_real(raw_results['transformer'])
            
            print("Displaying All Models Comparison plot...")
            plot_all_models_comparison(raw_results['gru'], raw_results['transformer'])
            
            print("Displaying Zoomed Comparison plot...")
            plot_zoom_in_comparison(raw_results['gru'], raw_results['transformer'])


def run_with_plots(device=None, output_dir="result_images"):
    """Convenience function to run comparison with plots
    
    Args:
        device: Computing device (CPU/GPU)
        output_dir: Directory to save plots
    """
    main(generate_plots=True, save_plots=True, output_dir=output_dir)


def run_without_plots(device=None):
    """Convenience function to run comparison without plots
    
    Args:
        device: Computing device (CPU/GPU)
    """
    main(generate_plots=False, save_plots=True, output_dir="result_images")


if __name__ == '__main__':
    run_with_plots()


