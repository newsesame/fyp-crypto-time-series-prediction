import torch
import numpy as np
from torch import Tensor
import time


def test_loss(model, test_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device):
    """Calculate test loss for model evaluation
    
    Args:
        model: The trained model to evaluate
        test_data: Test dataset tensor
        bptt_src: Source sequence length for backpropagation through time
        bptt_tgt: Target sequence length for backpropagation through time
        overlap: Overlap between source and target sequences
        criterion: Loss function (e.g., MSELoss)
        predicted_feature: Index of the feature to predict
        device: Computing device (CPU/GPU)
    
    Returns:
        tuple: (average_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, test_data.size(0) - 1, bptt_src):
            source, targets = get_batch(test_data, i, bptt_src, bptt_tgt, overlap)
            if source.size(0) != bptt_src:
                break
            output = model(source, targets)
            loss = criterion(output[:-1, :, predicted_feature], targets[1:, :, predicted_feature])
            total_loss += loss.item()
    return total_loss / (len(test_data) // bptt_src), 0.0


def get_batch(source, i, bptt_src, bptt_tgt, overlap):
    """Extract batch data for training/inference
    
    Args:
        source: Source data tensor
        i: Starting index for the batch
        bptt_src: Source sequence length
        bptt_tgt: Target sequence length
        overlap: Overlap between sequences
    
    Returns:
        tuple: (source_sequence, target_sequence)
    """
    seq_len = min(bptt_src, len(source) - 1 - i)
    src = source[i:i+seq_len]
    tgt_len = min(bptt_tgt, len(source) - 1 - i - seq_len)
    tgt = source[i+seq_len:i+seq_len+tgt_len]
    return src, tgt


def greedy_decode(model, src, bptt_src, pred_len, overlap, device):
    """Greedy decoding for sequence generation - optimized version
    
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
    with torch.no_grad():
        if hasattr(model, 'transformer'):
            # Transformer model
            src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)
            tgt_mask = model.transformer.generate_square_subsequent_mask(pred_len).to(device)
            memory = model.encode(src, src_mask)
            targets = torch.zeros(pred_len, src.size(1), src.size(2)).to(device)
            for i in range(pred_len):
                tgt_mask_i = tgt_mask[:i+1, :i+1]
                out = model.decode(targets[:i+1], memory, tgt_mask_i)
                targets[i] = model.generator(out[-1])
        else:
            # GRU model
            _, h = model.encoder(src)
            targets = torch.zeros(pred_len, src.size(1), src.size(2)).to(device)
            for i in range(pred_len):
                out, h = model.decoder(targets[i:i+1], h)
                targets[i] = model.proj(out[0])
        return targets.detach()


def estimate_BTC_fast(best_model, test, num_features, bptt_src, bptt_tgt, overlap, predicted_feature, scaler, device, 
                     use_real=True, early_stop=0.1, max_samples=1000):
    """
    Fast version of estimate_BTC for quick evaluation
    
    Args:
        best_model: The best trained model
        test: Test dataset DataFrame
        num_features: Number of input features
        bptt_src: Source sequence length
        bptt_tgt: Target sequence length
        overlap: Overlap between sequences
        predicted_feature: Index of feature to predict
        scaler: Fitted scaler for inverse transformation
        device: Computing device
        use_real: Whether to use real data for next prediction
        early_stop: Early stopping ratio (0.1 = process only first 10% of data)
        max_samples: Maximum number of samples to process
    
    Returns:
        tuple: (real_feature_values, predicted_feature_values, prediction_start_index)
    """
    print("Using fast estimation mode...")
    start_time = time.time()
    
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
    
    # Limit the amount of data to process for faster evaluation
    original_size = test_data.size(0)
    max_size = min(max_samples, int(original_size * early_stop)) if early_stop < 1.0 else original_size
    test_data = test_data[:max_size]
    
    num_iter = (test_data.size(0) - bptt_src) // pred_len
    print(f"Processing {num_iter} iterations (reduced from {(original_size - bptt_src) // pred_len})")
    
    inference_data = test_data[:inference_bptt_src, :, :]
    predictions = None
    
    for i in range(num_iter):
        prediction = greedy_decode(best_model, inference_data, bptt_src, pred_len, overlap, device)
        
        if use_real:
            inference_data = test_data[i * pred_len: i * pred_len + inference_bptt_src, :, :]
        else:
            inference_data = torch.cat([inference_data, prediction], dim=0)[pred_len:]
        
        if predictions is None:
            predictions = prediction
        else:
            predictions = torch.cat([predictions, prediction], dim=0)
    
    # Process original data to match prediction length
    feature_unnormalized = scaler.inverse_transform(
        torch.transpose(test_data, 0, 1).reshape(-1, num_features).cpu()
    )[:, predicted_feature]
    
    feature_prediction_unnormalized = scaler.inverse_transform(
        torch.transpose(predictions, 0, 1).reshape(-1, num_features).cpu()
    )[:, predicted_feature]
    
    elapsed = time.time() - start_time
    print(f"Fast estimation completed in {elapsed:.2f}s")
    
    return feature_unnormalized, feature_prediction_unnormalized, inference_bptt_src


def estimate_BTC_sample(best_model, test, num_features, bptt_src, bptt_tgt, overlap, predicted_feature, scaler, device, 
                       sample_size=500):
    """
    Sampling version of estimate_BTC - processes only a subset of data
    
    Args:
        best_model: The best trained model
        test: Test dataset DataFrame
        num_features: Number of input features
        bptt_src: Source sequence length
        bptt_tgt: Target sequence length
        overlap: Overlap between sequences
        predicted_feature: Index of feature to predict
        scaler: Fitted scaler for inverse transformation
        device: Computing device
        sample_size: Number of samples to randomly select
    
    Returns:
        tuple: (real_feature_values, predicted_feature_values, prediction_start_index)
    """
    print(f"Using sampling mode (sample_size={sample_size})...")
    start_time = time.time()
    
    # Random sampling
    total_rows = len(test)
    if total_rows > sample_size:
        sample_indices = np.random.choice(total_rows, sample_size, replace=False)
        sample_indices = np.sort(sample_indices)
        test_sample = test.iloc[sample_indices].reset_index(drop=True)
    else:
        test_sample = test
    
    # Use fast version to process sampled data
    return estimate_BTC_fast(best_model, test_sample, num_features, bptt_src, bptt_tgt, overlap, 
                           predicted_feature, scaler, device, use_real=True, early_stop=1.0, max_samples=sample_size)


__all__ = ["test_loss", "greedy_decode", "estimate_BTC_fast", "estimate_BTC_sample"]
