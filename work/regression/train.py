"""Training utilities for regression models

This module provides functions for training and evaluating regression models
(Transformer and GRU) for Bitcoin price prediction.
"""

import copy
import time
import numpy as np
import torch


def evaluate(model, data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device):
    """Evaluate model performance on validation/test data
    
    Args:
        model: Model to evaluate
        data: Evaluation dataset
        bptt_src: Source sequence length
        bptt_tgt: Target sequence length
        overlap: Overlap between sequences
        criterion: Loss function
        predicted_feature: Index of feature to predict
        device: Computing device
    
    Returns:
        float: Mean loss value
    """
    model.eval()
    total_loss = 0.0
    src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)
    if hasattr(model, 'transformer'):
        tgt_mask_full = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device)
    else:
        tgt_mask_full = None
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, bptt_src):
            source, targets = get_batch(data, i, bptt_src, bptt_tgt, overlap)
            src_batch_size = source.size(0)
            tgt_batch_size = targets.size(0)
            src_mask_use = src_mask[:src_batch_size, :src_batch_size]
            tgt_mask_use = None if tgt_mask_full is None else tgt_mask_full[:tgt_batch_size, :tgt_batch_size]
            output = model(source, targets, src_mask_use if hasattr(model, 'transformer') else None, tgt_mask_use)
            loss = criterion(output[:-1, :, predicted_feature], targets[1:, :, predicted_feature])
            total_loss += len(source) * loss.item()
    mean_loss = total_loss / (len(data) - 1)
    return mean_loss


def get_batch(data, i, bptt_src, bptt_tgt, overlap):
    """Extract source and target sequences from data
    
    Args:
        data: Input data tensor
        i: Starting index
        bptt_src: Source sequence length
        bptt_tgt: Target sequence length
        overlap: Overlap between source and target
    
    Returns:
        tuple: (source_sequence, target_sequence)
    """
    src_seq_len = min(bptt_src, len(data) - i - 1)
    target_seq_len = min(bptt_tgt, len(data) - i - src_seq_len + overlap)
    source = data[i: i + src_seq_len]
    target = data[i + src_seq_len - overlap: i + src_seq_len + target_seq_len - overlap]
    return source, target


def train_loop(model, train_data, val_data, *, epochs, lr, clip_param, scheduler, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device):
    """Main training loop for regression models
    
    Args:
        model: Model to train
        train_data: Training dataset
        val_data: Validation dataset (can be None)
        epochs: Number of training epochs
        lr: Learning rate
        clip_param: Gradient clipping parameter
        scheduler: Learning rate scheduler
        bptt_src: Source sequence length
        bptt_tgt: Target sequence length
        overlap: Overlap between sequences
        criterion: Loss function
        predicted_feature: Index of feature to predict
        device: Computing device
    
    Returns:
        tuple: (best_model, train_loss_history, validation_loss_history)
    """
    # Initialize optimizer and training state
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_model = None
    train_loss_hist = []
    valid_loss_hist = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        epoch_loss = 0.0
        start_time = time.time()

        # Calculate number of batches and logging interval
        num_batches = (len(train_data)) // bptt_src
        log_interval = max(1, round(num_batches // 3 / 10) * 10)

        # Create attention masks for Transformer models
        src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)
        if hasattr(model, 'transformer'):
            tgt_mask_full = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device)
        else:
            tgt_mask_full = None

        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt_src)):
            source, targets = get_batch(train_data, i, bptt_src, bptt_tgt, overlap)
            src_batch_size = source.size(0)
            tgt_batch_size = targets.size(0)
            src_mask_use = src_mask[:src_batch_size, :src_batch_size]
            tgt_mask_use = None if tgt_mask_full is None else tgt_mask_full[:tgt_batch_size, :tgt_batch_size]
            output = model(source, targets, src_mask_use if hasattr(model, 'transformer') else None, tgt_mask_use)
            loss = criterion(output[:-1, :, predicted_feature], targets[1:, :, predicted_feature])

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_param)
            optimizer.step()

            total_loss += loss.item()
            epoch_loss += len(source) * loss.item()

            if (batch % log_interval == 0) and batch > 0:
                lr_cur = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | lr {lr_cur:02.6f} | ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.6f} ')
                total_loss = 0
                start_time = time.time()

        train_loss_hist.append(epoch_loss / (len(train_data) - 1))

        # Validation step
        if val_data is not None:
            val_loss = evaluate(model, val_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device)
            elapsed = time.time() - epoch_start_time
            print('-' * 77)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.6f} ')
            print('-' * 77)
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
            valid_loss_hist.append(val_loss)

        scheduler.step()

    if val_data is None:
        best_model = copy.deepcopy(model)

    return best_model, train_loss_hist, valid_loss_hist


__all__ = ["train_loop", "evaluate", "get_batch"]


