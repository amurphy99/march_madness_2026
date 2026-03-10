"""
Run training or validation epochs.
--------------------------------------------------------------------------------
`src.training.epoch`

"""
import torch.nn.functional as F

from tqdm.auto import tqdm


# ================================================================================
# Separate function for a single epoch
# ================================================================================
def run_epoch(
        model, 
        data_loader, 
        *, 
        device,
        optimizer           = None, 
        progress_bar : tqdm = None, 
        secondary    : bool = False,
):
    # Optimizer only provided if we are in training mode
    if optimizer is not None: model.train()
    else:                     model.eval()
    
    # Tracking Progress
    running_loss  = 0.0
    run_box_loss  = 0.0
    run_win_loss  = 0.0
    total_batches = 0

    # Metrics for win probability predictions
    running_win_correct = 0
    total_win_samples   = 0
    running_win_mse     = 0.0

    for batch in data_loader:
        # --------------------------------------------------------------------------------
        # 1) Prepare the input data
        # --------------------------------------------------------------------------------
        (input_data, box_score_target, win_proba_target) = batch

        # Move to device
        input_data = input_data.to(device)

        box_score_target = box_score_target.to(device)
        win_proba_target = win_proba_target.to(device)
        
        # --------------------------------------------------------------------------------
        # 2) Forward pass (secondary tournament data doesn't have box scores)
        # --------------------------------------------------------------------------------
        box_score_pred, win_proba_pred = model(input_data)
        
        # Calculate and combine both losses
        if not secondary:
            loss_box = F.l1_loss (box_score_pred, box_score_target)
            loss_win = F.mse_loss(win_proba_pred, win_proba_target)
            
            #loss_win = F.binary_cross_entropy(win_proba_pred, win_proba_target)
            
            loss = loss_box + loss_win
        else:
            loss_win = F.mse_loss(win_proba_pred, win_proba_target) # binary_cross_entropy
            #loss_win = F.binary_cross_entropy(win_proba_pred, win_proba_target) # binary_cross_entropy
            loss = loss_win * 5

        # Backpropagation if training
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Progress Tracking
        running_loss  += loss.item()
        run_box_loss  += (loss_box.item() if not secondary else 0)
        run_win_loss  += loss_win.item()
        total_batches += 1
            
        # --------------------------------------------------------------------------------
        # 3) Win probability metrics
        # --------------------------------------------------------------------------------
        # Squeeze to remove extra dimension: expected shape [batch_size]
        win_pred   = win_proba_pred  .squeeze()
        win_target = win_proba_target.squeeze()

        # Accuracy: threshold at 0.5
        pred_labels   = (win_pred >= 0.5).float()
        batch_correct = (pred_labels == win_target).sum().item()
        running_win_correct += batch_correct
        total_win_samples   += win_target.numel()

        # MSE for win probability predictions
        batch_win_mse = F.mse_loss(win_pred, win_target, reduction='mean').item()
        running_win_mse += batch_win_mse * win_target.numel()

        if progress_bar is not None: progress_bar.update(1)
            
    # --------------------------------------------------------------------------------
    # Progress metrics
    # --------------------------------------------------------------------------------     
    metrics = {
        "epoch_loss" : running_loss / total_batches, 
        "box_loss"   : run_box_loss / total_batches,
        "win_loss"   : run_win_loss / total_batches,
        "win_acc"    : running_win_correct / total_win_samples,
        "win_mse"    : running_win_mse     / total_win_samples
    }

    return metrics
