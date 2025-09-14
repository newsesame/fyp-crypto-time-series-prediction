import numpy as np
import torch
import sklearn.preprocessing as pp

# Support two modes:
# 1) Package mode: python -m work.regression.main
# 2) Direct execution: python work/regression/main.py
try:
    from ..config import (
        DATA_PATH, plot_data_process, num_features, scaler_name, train_batch_size, eval_batch_size, epochs,
        bptt_src, bptt_tgt, overlap, num_encoder_layers, num_decoder_layers, periodic_features, out_features,
        nhead, dim_feedforward, dropout, activation, random_start_point, clip_param, lr, gamma, step_size,
    )
    from ..compat import apply_pandas_finta_compatibility
    from ..data_loader import load_and_rename, sort_and_convert_timestamp
    from ..features import add_finta_feature, build_finta_input, drop_until_last_nan_and_reorder, normalize_split_and_batch
    from ..models import BTC_Transformer
    from .train import train_loop
    from .evaluate import test_loss, estimate_BTC
    from .evaluate_fast import estimate_BTC_fast
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from work.config import (
        DATA_PATH, plot_data_process, num_features, scaler_name, train_batch_size, eval_batch_size, epochs,
        bptt_src, bptt_tgt, overlap, num_encoder_layers, num_decoder_layers, periodic_features, out_features,
        nhead, dim_feedforward, dropout, activation, random_start_point, clip_param, lr, gamma, step_size,
    )
    from work.compat import apply_pandas_finta_compatibility
    from work.data_loader import load_and_rename, sort_and_convert_timestamp
    from work.features import add_finta_feature, build_finta_input, drop_until_last_nan_and_reorder, normalize_split_and_batch
    from work.models import BTC_Transformer
    from work.regression.train import train_loop
    from work.regression.evaluate import test_loss, estimate_BTC
    from work.regression.evaluate_fast import estimate_BTC_fast


def run(device=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
    apply_pandas_finta_compatibility()

    # Load, rename, sort and convert timestamp
    data = load_and_rename(str(DATA_PATH))
    data = sort_and_convert_timestamp(data)

    # Add finta technical indicators
    data_min = data.copy()
    data_finta = build_finta_input(data_min)
    extra_features = ['TRIX', 'VWAP', 'MACD', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI', 'ADX', 'STOCHRSI',
                      'MI', 'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP', 'BASP', 'BASPN', 'WTO', 'SQZMI', 'VFI', 'STC']
    both_columns_features = ["DMI", "EBBP", "BASP", "BASPN"]
    data_min = add_finta_feature(data_min, data_finta, extra_features, both_columns_features)

    # Drop until last NaN and reorder columns
    new_columns_order = ['Closing_price', 'Volume_of_transactions', 'Opening_price', 'Highest_price', 'Lowest_price','TRIX', 'VWAP', 'MACD',
                         'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI_1', 'DMI_2', 'ADX', 'STOCHRSI', 'MI', 'CHAIKIN', 
                         'VZO', 'PZO', 'EFI', 'EBBP_1', 'EBBP_2', 'BASP_1', 'BASP_2', 'BASPN_1', 'BASPN_2', 'WTO', 'SQZMI', 'VFI', 'STC']
    data_min, start_index = drop_until_last_nan_and_reorder(data_min, new_columns_order)

    # Split data, normalize and batchify
    train_df, val_df, test_df, train_data, val_data, test_data, scaler = normalize_split_and_batch(
        data_min, num_features, 0.1, 0.1, scaler_name, train_batch_size, eval_batch_size, device
    )

    # Initialize model with hyperparameters
    predicted_feature = train_df.columns.get_loc('Closing_price')
    scaler_impl = pp.StandardScaler() if scaler_name == 'standard' else pp.MinMaxScaler()
    criterion = torch.nn.MSELoss()

    model = BTC_Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        in_features=num_features,
        periodic_features=periodic_features,
        out_features=out_features,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    ).to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(torch.optim.SGD(model.parameters(), lr=lr), step_size, gamma)

    best_model, train_loss_hist, valid_loss_hist = train_loop(
        model, train_data, val_data,
        epochs=epochs, lr=lr, clip_param=clip_param, scheduler=scheduler,
        bptt_src=bptt_src, bptt_tgt=bptt_tgt, overlap=overlap,
        criterion=criterion, predicted_feature=predicted_feature, device=device,
    )

    loss, ppl = test_loss(best_model, test_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device)
    print('-' * 77)
    print(f'Transformer test loss {loss:5.6f}')
    print('-' * 77)
    
    feature_real, feature_pred, pred_start = estimate_BTC(best_model, test_df, num_features, bptt_src, bptt_tgt,
                                                          overlap, predicted_feature, scaler, device, use_real=True, early_stop=1)
    # Use fast version for prediction
    # feature_real, feature_pred, pred_start = estimate_BTC_fast(best_model, test_df, num_features, bptt_src, bptt_tgt,
    #                                                           overlap, predicted_feature, scaler, device, 
    #                                                           use_real=True, early_stop=0.1, max_samples=1000)
    return {
        'train_loss': train_loss_hist,
        'valid_loss': valid_loss_hist,
        'test_loss': loss,
        'feature_real': feature_real,
        'feature_pred': feature_pred,
        'pred_start': pred_start,
    }


if __name__ == '__main__':
    run()


