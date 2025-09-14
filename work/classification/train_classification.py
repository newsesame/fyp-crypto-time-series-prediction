import copy
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def train_classifier(model, train_data, val_data, *, epochs, lr, batch_size, device):
    """Train classification model
    
    Args:
        model: The model to train
        train_data: Training dataset
        val_data: Validation dataset (can be None)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        device: Computing device (CPU/GPU)
    
    Returns:
        tuple: (best_model, train_loss_history, validation_accuracy_history)
    """
    model.to(device)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size) if val_data else None
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    best_val_acc = 0.0
    best_model = None
    train_loss_hist = []
    valid_acc_hist = []
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        train_loss_hist.append(avg_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            val_acc = 100. * val_correct / val_total
            valid_acc_hist.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
            
            elapsed = time.time() - epoch_start_time
            print(f'Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {elapsed:.2f}s')
        else:
            elapsed = time.time() - epoch_start_time
            print(f'Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Time: {elapsed:.2f}s')
        
        scheduler.step()
    
    if val_loader is None:
        best_model = copy.deepcopy(model)
    
    return best_model, train_loss_hist, valid_acc_hist


def evaluate_classifier(model, test_data, device, batch_size=64):
    """Evaluate classification model
    
    Args:
        model: Trained model to evaluate
        test_data: Test dataset
        device: Computing device (CPU/GPU)
        batch_size: Batch size for evaluation
    
    Returns:
        tuple: (accuracy, f1_score, predictions, targets)
    """
    model.eval()
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    correct = 0
    total = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    f1 = f1_score(targets, predictions, average='weighted')
    
    return accuracy, f1, np.array(predictions), np.array(targets)


__all__ = ["train_classifier", "evaluate_classifier"]
