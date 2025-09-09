#!/usr/bin/env python3
"""Test script for plotting functionality

This script demonstrates how to use the plotting functions to generate
comparison plots for GRU and Transformer model results.
"""

import numpy as np
import matplotlib.pyplot as plt
from .plotting import (
    plot_gru_vs_real, plot_transformer_vs_real, plot_all_models_comparison,
    plot_zoom_in_comparison, generate_all_plots
)


def create_sample_data():
    """Create sample data for testing plotting functions"""
    # Generate sample time series data
    np.random.seed(42)
    n_points = 200
    
    # Create realistic Bitcoin price-like data
    base_price = 50000
    trend = np.linspace(0, 0.1, n_points)  # 10% upward trend
    noise = np.random.normal(0, 0.02, n_points)  # 2% noise
    real_values = base_price * (1 + trend + noise)
    
    # Create predictions with some error
    gru_predictions = real_values + np.random.normal(0, 1000, n_points)
    transformer_predictions = real_values + np.random.normal(0, 800, n_points)
    
    # Create sample results dictionaries
    gru_results = {
        'feature_real': real_values,
        'feature_pred': gru_predictions,
        'test_loss': 0.001,
        'train_loss': [0.01, 0.005, 0.002],
        'valid_loss': [0.008, 0.004, 0.0015]
    }
    
    transformer_results = {
        'feature_real': real_values,
        'feature_pred': transformer_predictions,
        'test_loss': 0.0008,
        'train_loss': [0.009, 0.004, 0.0018],
        'valid_loss': [0.007, 0.003, 0.0012]
    }
    
    return gru_results, transformer_results


def test_individual_plots():
    """Test individual plotting functions"""
    print("Creating sample data...")
    gru_results, transformer_results = create_sample_data()
    
    print("Testing individual plots...")
    
    # Test GRU vs Real plot
    print("1. Testing GRU vs Real plot...")
    plot_gru_vs_real(gru_results)
    
    # Test Transformer vs Real plot
    print("2. Testing Transformer vs Real plot...")
    plot_transformer_vs_real(transformer_results)
    
    # Test all models comparison
    print("3. Testing All Models Comparison plot...")
    plot_all_models_comparison(gru_results, transformer_results)
    
    # Test zoomed comparison
    print("4. Testing Zoomed Comparison plot...")
    plot_zoom_in_comparison(gru_results, transformer_results, start_idx=50, end_idx=150)


def test_batch_plotting():
    """Test batch plotting function"""
    print("Testing batch plotting...")
    gru_results, transformer_results = create_sample_data()
    
    # Generate all plots and save them
    generate_all_plots(gru_results, transformer_results, output_dir="test_images")


def main():
    """Main test function"""
    print("=" * 60)
    print("Testing Plotting Functions")
    print("=" * 60)
    
    try:
        # Test individual plots
        test_individual_plots()
        
        # Test batch plotting
        test_batch_plotting()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
