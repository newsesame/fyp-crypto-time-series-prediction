from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "okex_btcusdt_kline_1m.csv"

# Notebook hyper-parameters (kept as defaults)
plot_data_process = True

# data Hyper-Parameters
num_features = 34
scaler_name = 'minmax'  # ['standard','minmax']
train_batch_size = 32
eval_batch_size = 32
epochs = 1
bptt_src = 10
bptt_tgt = 2
overlap = 1

# model Hyper-Parameters
num_encoder_layers = 4
num_decoder_layers = 4
periodic_features = 10
out_features = 60
nhead = 15
dim_feedforward = 384
dropout = 0.0
activation = 'gelu'

# training Hyper-Parameters
random_start_point = False
clip_param = 0.75
lr = 0.5
gamma = 0.95
step_size = 1.0

__all__ = [name for name in dir() if name.isidentifier() and not name.startswith("_")]


