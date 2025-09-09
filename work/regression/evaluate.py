"""Evaluation utilities for regression models

This module provides functions for evaluating trained regression models
and generating predictions for Bitcoin price forecasting.
"""

import math
import numpy as np
import torch

from .train import evaluate as eval_loop, get_batch


def test_loss(best_model, test_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device):
    """Calculate test loss and perplexity
    
    Args:
        best_model: Trained model to evaluate
        test_data: Test dataset
        bptt_src: Source sequence length
        bptt_tgt: Target sequence length
        overlap: Overlap between sequences
        criterion: Loss function
        predicted_feature: Index of feature to predict
        device: Computing device
    
    Returns:
        tuple: (loss, perplexity)
    """
    loss = eval_loop(best_model, test_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device)
    ppl = math.exp(loss)
    return loss, ppl


def greedy_decode(model, src, bptt_src, pred_len, overlap, device):
    """Greedy decoding for sequence generation
    
    Args:
        model: Trained model for inference
        src: Source sequence tensor
        bptt_src: Source sequence length
        pred_len: Length of prediction sequence
        overlap: Overlap parameter
        device: Computing device
    
    Returns:
        tensor: Generated target sequence
    """
    model.eval()
    if overlap == 0:
        start_point = src[-1:, :, :]
        src = src[:-1, :, :]
    else:
        start_point = src[-overlap:, :, :]

    # Transformer model path
    if hasattr(model, 'transformer'):
        bptt_src_len = src.size(0)
        src_mask = torch.zeros((bptt_src_len, bptt_src_len), dtype=torch.bool).to(device)
        memory = model.encode(src, src_mask)
        targets = start_point
        for i in range(pred_len):
            tgt_mask = model.transformer.generate_square_subsequent_mask(targets.size(0)).to(device)
            prediction = model.decode(targets, memory, tgt_mask)
            prediction = model.generator(prediction)
            targets = torch.cat([start_point, prediction[:i + 1]], dim=0)
        greedy_output = prediction[:pred_len, :, :].detach()
        return greedy_output

    # GRU model path: step-by-step decoding using internal encoder/decoder
    with torch.no_grad():
        _, h = model.encoder(src)
        targets = start_point
        preds = []
        for _ in range(pred_len):
            out, h = model.decoder(targets[-1:], h)
            y = model.proj(out)  # (1, batch, E)
            preds.append(y)
            targets = torch.cat([targets, y], dim=0)
        prediction = torch.cat(preds, dim=0)
        return prediction.detach()


def estimate_BTC(best_model, test, num_features, bptt_src, bptt_tgt, overlap, predicted_feature, scaler, device, use_real=True, early_stop=1):
    """Generate Bitcoin price predictions using trained model
    
    Args:
        best_model: Trained model for prediction
        test: Test dataset DataFrame
        num_features: Number of input features
        bptt_src: Source sequence length
        bptt_tgt: Target sequence length
        overlap: Overlap between sequences
        predicted_feature: Index of feature to predict
        scaler: Fitted scaler for inverse transformation
        device: Computing device
        use_real: Whether to use real data for next prediction
        early_stop: Early stopping ratio (0.0-1.0)
    
    Returns:
        tuple: (real_feature_values, predicted_feature_values, prediction_start_index)
    """
    inference_batch_size = 1
    inference_bptt_src = bptt_src + (overlap == 0)
    pred_len = min(bptt_tgt - overlap, bptt_tgt - 1)
    # Batchify data similar to notebook implementation
    def batchify(data, batch_size):
        """Convert data to batched format for inference
        
        Args:
            data: Input tensor
            batch_size: Number of sequences per batch
        
        Returns:
            Batched tensor with shape (seq_len, batch_size, features)
        """
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


