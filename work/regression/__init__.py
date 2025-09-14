"""Regression task modules

This package contains modules for Bitcoin price regression prediction.
It implements both Transformer and GRU models to predict continuous
Bitcoin price values for the next time steps.

Modules:
- main: Main execution script for Transformer regression
- main_gru: Main execution script for GRU regression
- train: Training utilities for regression models
- evaluate: Evaluation utilities for regression models
- evaluate_fast: Fast evaluation utilities for quick testing
- compare: Model comparison utilities with plotting support
- plotting: Plotting utilities for visualization of model results
- example_usage: Example usage of comparison and plotting functions
- test_plotting: Test script for plotting functionality

Plotting Features:
- GRU vs Real Values comparison plots
- Transformer vs Real Values comparison plots
- All models comparison plots (GRU, Transformer, Real)
- Zoomed-in detailed comparison plots
- Batch plot generation and saving
"""