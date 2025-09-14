#!/usr/bin/env python3
"""
Demonstrate the new model comparison output format
"""

import pandas as pd

def show_example_output():
    """Show example of the new output format"""
    
    print("=" * 60)
    print("Regression Task Comparison Example")
    print("=" * 60)
    
    # Simulate regression task results
    model_info_regression = pd.DataFrame([
        {
            'Model': 'Transformer',
            'Architecture': 'Enc:6, Dec:6, Heads:8, FF:512',
            'Training': 'Epochs:100, LR:0.001, Batch:32',
            'Sequence': 'Src:60, Tgt:1, Features:34',
            'Scaler': 'standard',
        },
        {
            'Model': 'GRU',
            'Architecture': 'Hidden:256, Layers:2, Dropout:0.1',
            'Training': 'Epochs:100, LR:0.001, Batch:32',
            'Sequence': 'Src:60, Tgt:1, Features:34',
            'Scaler': 'standard',
        },
    ])
    
    metrics_regression = pd.DataFrame([
        {
            'Model': 'Transformer',
            'MAE': 0.0234,
            'MSE': 0.0008,
            'CORR': 0.8567,
        },
        {
            'Model': 'GRU',
            'MAE': 0.0256,
            'MSE': 0.0009,
            'CORR': 0.8234,
        },
    ])
    
    print("\nModel Comparison\n")
    print("Model Information:")
    print(model_info_regression.to_string(index=False))
    
    print("\nModel Performance:")
    print(metrics_regression.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Classification Task Comparison Example")
    print("=" * 60)
    
    # Simulate classification task results
    model_info_classification = pd.DataFrame([
        {
            'Model': 'Transformer',
            'Architecture': 'Enc:6, Heads:8, FF:512, Dropout:0.1',
            'Training': 'Epochs:50, LR:0.001, Batch:32',
            'Features': 'Features:34, Scaler:standard',
        },
        {
            'Model': 'GRU',
            'Architecture': 'Hidden:256, Layers:2, Dropout:0.1',
            'Training': 'Epochs:50, LR:0.001, Batch:32',
            'Features': 'Features:34, Scaler:standard',
        },
    ])
    
    metrics_classification = pd.DataFrame([
        {
            'Model': 'Transformer',
            'Test Accuracy (%)': 67.45,
            'Test F1 Score': 0.6234,
        },
        {
            'Model': 'GRU',
            'Test Accuracy (%)': 65.23,
            'Test F1 Score': 0.5987,
        },
    ])
    
    print("\nClassification Model Comparison\n")
    print("Model Information:")
    print(model_info_classification.to_string(index=False))
    
    print("\nModel Performance:")
    print(metrics_classification.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Advantages of the new format:")
    print("=" * 60)
    print("1. Model information and performance metrics are displayed separately for clarity")
    print("2. Model configuration information is clear at a glance")
    print("3. Performance metrics are displayed independently for easy comparison")
    print("4. Structured output for easy reading and analysis")

if __name__ == '__main__':
    show_example_output()
