"""Plotting utilities for regression model results

This module provides functions to generate comparison plots for Bitcoin price
prediction results from GRU and Transformer models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def plot_gru_vs_real(gru_results: dict, save_path: Optional[str] = None) -> None:
    """Plot GRU predictions vs real values
    
    Args:
        gru_results: Dictionary containing GRU model results
        save_path: Optional path to save the plot
    """
    real_values = gru_results['feature_real']
    gru_predictions = gru_results['feature_pred']
    
    # Ensure both arrays have the same length
    min_length = min(len(real_values), len(gru_predictions))
    real_values = real_values[:min_length]
    gru_predictions = gru_predictions[:min_length]
    
    # Create time index for x-axis
    time_index = np.arange(len(real_values))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_index, real_values, label='Real Values', color='blue', linewidth=1.5)
    plt.plot(time_index, gru_predictions, label='GRU Predictions', color='red', linewidth=1.5, alpha=0.8)
    
    plt.title('GRU Price Prediction vs Real Values', fontsize=16, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Bitcoin Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance metrics to the plot
    mae = np.mean(np.abs(real_values - gru_predictions))
    mse = np.mean((real_values - gru_predictions) ** 2)
    corr = np.corrcoef(real_values, gru_predictions)[0, 1]
    
    plt.text(0.02, 0.98, f'MAE: {mae:.4f}\nMSE: {mse:.6f}\nCorrelation: {corr:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GRU vs Real plot saved to: {save_path}")
    
    plt.show()


def plot_transformer_vs_real(transformer_results: dict, save_path: Optional[str] = None) -> None:
    """Plot Transformer predictions vs real values
    
    Args:
        transformer_results: Dictionary containing Transformer model results
        save_path: Optional path to save the plot
    """
    real_values = transformer_results['feature_real']
    transformer_predictions = transformer_results['feature_pred']
    
    # Ensure both arrays have the same length
    min_length = min(len(real_values), len(transformer_predictions))
    real_values = real_values[:min_length]
    transformer_predictions = transformer_predictions[:min_length]
    
    # Create time index for x-axis
    time_index = np.arange(len(real_values))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_index, real_values, label='Real Values', color='blue', linewidth=1.5)
    plt.plot(time_index, transformer_predictions, label='Transformer Predictions', color='green', linewidth=1.5, alpha=0.8)
    
    plt.title('Transformer Price Prediction vs Real Values', fontsize=16, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Bitcoin Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance metrics to the plot
    mae = np.mean(np.abs(real_values - transformer_predictions))
    mse = np.mean((real_values - transformer_predictions) ** 2)
    corr = np.corrcoef(real_values, transformer_predictions)[0, 1]
    
    plt.text(0.02, 0.98, f'MAE: {mae:.4f}\nMSE: {mse:.6f}\nCorrelation: {corr:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Transformer vs Real plot saved to: {save_path}")
    
    plt.show()


def plot_all_models_comparison(gru_results: dict, transformer_results: dict, save_path: Optional[str] = None) -> None:
    """Plot comparison of GRU, Transformer predictions and real values
    
    Args:
        gru_results: Dictionary containing GRU model results
        transformer_results: Dictionary containing Transformer model results
        save_path: Optional path to save the plot
    """
    real_values = gru_results['feature_real']  # Both should have the same real values
    gru_predictions = gru_results['feature_pred']
    transformer_predictions = transformer_results['feature_pred']
    
    # Ensure all arrays have the same length
    min_length = min(len(real_values), len(gru_predictions), len(transformer_predictions))
    real_values = real_values[:min_length]
    gru_predictions = gru_predictions[:min_length]
    transformer_predictions = transformer_predictions[:min_length]
    
    # Create time index for x-axis
    time_index = np.arange(len(real_values))
    
    plt.figure(figsize=(14, 8))
    plt.plot(time_index, real_values, label='Real Values', color='blue', linewidth=2)
    plt.plot(time_index, gru_predictions, label='GRU Predictions', color='red', linewidth=1.5, alpha=0.8)
    plt.plot(time_index, transformer_predictions, label='Transformer Predictions', color='green', linewidth=1.5, alpha=0.8)
    
    plt.title('Bitcoin Price Prediction: GRU vs Transformer vs Real Values', fontsize=16, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Bitcoin Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Calculate and display performance metrics for both models
    gru_mae = np.mean(np.abs(real_values - gru_predictions))
    gru_mse = np.mean((real_values - gru_predictions) ** 2)
    gru_corr = np.corrcoef(real_values, gru_predictions)[0, 1]
    
    tf_mae = np.mean(np.abs(real_values - transformer_predictions))
    tf_mse = np.mean((real_values - transformer_predictions) ** 2)
    tf_corr = np.corrcoef(real_values, transformer_predictions)[0, 1]
    
    metrics_text = f'GRU Metrics:\nMAE: {gru_mae:.4f}\nMSE: {gru_mse:.6f}\nCorr: {gru_corr:.4f}\n\n' \
                  f'Transformer Metrics:\nMAE: {tf_mae:.4f}\nMSE: {tf_mse:.6f}\nCorr: {tf_corr:.4f}'
    
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All models comparison plot saved to: {save_path}")
    
    plt.show()


def plot_zoom_in_comparison(gru_results: dict, transformer_results: dict, 
                           start_idx: int = 0, end_idx: int = 100, 
                           save_path: Optional[str] = None) -> None:
    """Plot zoomed-in comparison of predictions for better detail view
    
    Args:
        gru_results: Dictionary containing GRU model results
        transformer_results: Dictionary containing Transformer model results
        start_idx: Starting index for zoom
        end_idx: Ending index for zoom
        save_path: Optional path to save the plot
    """
    # Ensure we don't exceed array bounds
    real_values = gru_results['feature_real']
    gru_predictions = gru_results['feature_pred']
    transformer_predictions = transformer_results['feature_pred']
    
    min_length = min(len(real_values), len(gru_predictions), len(transformer_predictions))
    end_idx = min(end_idx, min_length)
    start_idx = min(start_idx, end_idx - 1)
    
    real_values = real_values[start_idx:end_idx]
    gru_predictions = gru_predictions[start_idx:end_idx]
    transformer_predictions = transformer_predictions[start_idx:end_idx]
    
    # Create time index for x-axis
    time_index = np.arange(len(real_values))
    
    plt.figure(figsize=(14, 8))
    plt.plot(time_index, real_values, label='Real Values', color='blue', linewidth=2, marker='o', markersize=3)
    plt.plot(time_index, gru_predictions, label='GRU Predictions', color='red', linewidth=1.5, alpha=0.8, marker='s', markersize=2)
    plt.plot(time_index, transformer_predictions, label='Transformer Predictions', color='green', linewidth=1.5, alpha=0.8, marker='^', markersize=2)
    
    plt.title(f'Bitcoin Price Prediction (Zoomed View: Steps {start_idx}-{end_idx})', fontsize=16, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Bitcoin Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Calculate metrics for the zoomed section
    gru_mae = np.mean(np.abs(real_values - gru_predictions))
    tf_mae = np.mean(np.abs(real_values - transformer_predictions))
    
    plt.text(0.02, 0.98, f'Zoomed Section Metrics:\nGRU MAE: {gru_mae:.4f}\nTransformer MAE: {tf_mae:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zoomed comparison plot saved to: {save_path}")
    
    plt.show()


def generate_all_plots(gru_results: dict, transformer_results: dict, 
                      output_dir: str = "result_images") -> None:
    """Generate all comparison plots and save them to the specified directory
    
    Args:
        gru_results: Dictionary containing GRU model results
        transformer_results: Dictionary containing Transformer model results
        output_dir: Directory to save the plots
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating all comparison plots...")
    
    # Generate individual model vs real plots
    plot_gru_vs_real(gru_results, os.path.join(output_dir, "GRU_vs_Real.png"))
    plot_transformer_vs_real(transformer_results, os.path.join(output_dir, "Transformer_vs_Real.png"))
    
    # Generate combined comparison plot
    plot_all_models_comparison(gru_results, transformer_results, 
                              os.path.join(output_dir, "All_Models_Comparison.png"))
    
    # Generate zoomed-in comparison plot
    plot_zoom_in_comparison(gru_results, transformer_results, 
                           start_idx=0, end_idx=100,
                           save_path=os.path.join(output_dir, "Zoomed_Comparison.png"))
    
    print(f"All plots have been generated and saved to: {output_dir}")


__all__ = [
    "plot_gru_vs_real",
    "plot_transformer_vs_real", 
    "plot_all_models_comparison",
    "plot_zoom_in_comparison",
    "generate_all_plots"
]
