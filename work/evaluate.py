import math
import numpy as np
import torch

from .train import evaluate as eval_loop, get_batch


def test_loss(best_model, test_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device):
    loss = eval_loop(best_model, test_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device)
    ppl = math.exp(loss)
    return loss, ppl


def greedy_decode(model, src, bptt_src, pred_len, overlap, device):
    model.eval()
    if overlap == 0:
        start_point = src[-1:, :, :]
        src = src[:-1, :, :]
    else:
        start_point = src[-overlap:, :, :]
    bptt_src = src.size(0)
    src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)
    memory = model.encode(src, src_mask)
    targets = start_point
    for i in range(pred_len):
        tgt_mask = model.transformer.generate_square_subsequent_mask(targets.size(0)).to(device)
        prediction = model.decode(targets, memory, tgt_mask)
        prediction = model.generator(prediction)
        targets = torch.cat([start_point, prediction[:i + 1]], dim=0)
    greedy_output = prediction[:pred_len, :, :].detach()
    return greedy_output


def estimate_BTC(best_model, test, num_features, bptt_src, bptt_tgt, overlap, predicted_feature, scaler, device, use_real=True, early_stop=1):
    inference_batch_size = 1
    inference_bptt_src = bptt_src + (overlap == 0)
    pred_len = min(bptt_tgt - overlap, bptt_tgt - 1)
    # batchify like notebook
    def batchify(data, batch_size):
        seq_len = data.size(0) // batch_size
        data = data[:seq_len * batch_size, :]
        data = data.view(batch_size, seq_len, -1)
        data = torch.transpose(data, 0, 1).contiguous()
        return data

    test_data = batchify(
        torch.tensor(scaler.transform(test.iloc[:, :num_features].to_numpy())),
        inference_batch_size,
    ).float()
    num_iter = (test_data.size(0) - bptt_src) // pred_len
    inference_data = test_data[:inference_bptt_src, :, :]
    for i in range(num_iter):
        prediction = greedy_decode(best_model, inference_data, bptt_src, pred_len, overlap, device)
        if use_real:
            inference_data = test_data[i * pred_len: i * pred_len + inference_bptt_src, :, :]
        else:
            inference_data = torch.cat([inference_data, prediction], dim=0)[pred_len:]
        if i == 0:
            predictions = prediction
        else:
            predictions = torch.cat([predictions, prediction], dim=0)
        if i > num_iter * early_stop:
            break

    feature_unnormalized = scaler.inverse_transform(torch.transpose(test_data, 0, 1).reshape(-1, num_features).cpu())[:, predicted_feature]
    feature_prediction_unnormalized = scaler.inverse_transform(torch.transpose(predictions, 0, 1).reshape(-1, num_features).cpu())[:, predicted_feature]
    return feature_unnormalized, feature_prediction_unnormalized, inference_bptt_src


__all__ = ["test_loss", "greedy_decode", "estimate_BTC"]


