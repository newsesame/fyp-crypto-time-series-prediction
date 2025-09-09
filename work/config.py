"""Configuration file for Bitcoin price prediction project

This module contains all hyperparameters and configuration settings
for the Bitcoin price prediction models (Transformer and GRU).
"""

from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "okex_btcusdt_kline_1m.csv"

# Visualization settings
plot_data_process = True  # Whether to plot data processing steps

# Data preprocessing hyperparameters
num_features = 34
scaler_name = 'minmax'  # Normalization method: 'standard' or 'minmax'
train_batch_size = 32
eval_batch_size = 32
epochs = 20
bptt_src = 10
bptt_tgt = 2
overlap = 1

# Model architecture hyperparameters
num_encoder_layers = 4
num_decoder_layers = 4
periodic_features = 10
out_features = 60
nhead = 15
dim_feedforward = 384
dropout = 0.0
activation = 'gelu'

# Training hyperparameters
random_start_point = False
clip_param = 0.75
lr = 0.5
gamma = 0.95
step_size = 1.0

# GRU model specific hyperparameters
gru_hidden_size = 256
gru_num_layers = 2
gru_dropout = 0.1

__all__ = [name for name in dir() if name.isidentifier() and not name.startswith("_")]


