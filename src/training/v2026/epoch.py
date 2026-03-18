"""
Run training or validation epochs.
--------------------------------------------------------------------------------
`src.training.v2026.epoch`

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

# From this project
from ..utils.losses import gaussian_nll_loss, evidential_binary_loss


# --------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------
def _move_batch_to_device(batch: dict, device: str) -> dict:
    """
    Move all tensor values in the batch dict to the target device.
    """
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }

# ================================================================================
# Runs one training or evaluation epoch
# ================================================================================
def run_epoch(
        model,
        data_loader,
        *,
        device,
        optimizer                 = None,
        progress_bar: tqdm | None = None,

        # Loss config
        box_loss_fn               = None,
        win_loss_fn               = None,
        box_loss_weight: float    = 1.0,
        win_loss_weight: float    = 1.0,

        # If False, skip the loss entirely
        use_box_loss      : bool = True,
        use_mean_var_loss : bool = False,

        # Alpha-beta loss
        use_alpha_beta    : bool  = True,
        alpha_beta_weight : float = 0.0,

        # Optional gradient clipping
        grad_clip_norm: float | None = None,
):
    """
    Returns a metrics dict with:  
        epoch_loss, box_loss, win_loss, win_acc, win_mse
    """
    # Default losses
    if box_loss_fn is None: box_loss_fn = F.l1_loss
    if win_loss_fn is None: win_loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer only provided if we are in training mode
    is_training = optimizer is not None
    if is_training: model.train()
    else:           model.eval()

    # Tracking Progress
    running_loss  = 0.0
    run_box_loss  = 0.0
    run_win_loss  = 0.0
    total_batches = 0

    # Metrics for win predictions
    running_win_correct = 0
    total_win_samples   = 0
    running_win_mse     = 0.0

    for batch in data_loader:
        # --------------------------------------------------------------------------------
        # 1) Prepare the input data
        # --------------------------------------------------------------------------------
        batch = _move_batch_to_device(batch, device)

        target_box_score = batch["target_box_score"].float()
        target_win       = batch["target_win"      ].float()

        if is_training: optimizer.zero_grad(set_to_none=True)

        # ================================================================================
        # 2) Forward pass
        # ================================================================================
        with torch.set_grad_enabled(is_training):

            # Unpack differently depending on the model architecture
            if use_alpha_beta:
                (box_mu, box_log_var), win_logit, alpha_beta = model(batch)

            else:
                if use_mean_var_loss: (box_mu, box_log_var), win_logit = model(batch)
                else:                 box_score_pred,        win_logit = model(batch)

            # --------------------------------------------------------------------------------
            # Seed-Weighted Modifiers
            # --------------------------------------------------------------------------------
            # batch["teamA_seed"] is > 0 if they made the tournament, 0 otherwise.
            A_is_tourney = (batch["teamA_seed"] > 0).float()
            B_is_tourney = (batch["teamB_seed"] > 0).float()

            # Base game weight is 1.0 (1 Tourney Team = 2.0 weight; 2 Tourney Teams = 3.0 weight).
            sample_weights = 1.0 + (A_is_tourney * 1.0) + (B_is_tourney * 1.0)

            # Normalize the weights so the batch average remains 1.0 (prevents learning rate from exploding).
            sample_weights = sample_weights / sample_weights.mean()

            # --------------------------------------------------------------------------------
            # a) Win Prediction Loss
            # --------------------------------------------------------------------------------
            # All models now always return the logit only; if we have MSELoss, we do sigmoid here
            win_proba = torch.sigmoid(win_logit)

            # For MSELoss, use `win_proba` instead of the logit            
            #if isinstance(win_loss_fn, nn.MSELoss): loss_win = win_loss_fn(win_proba, target_win)
            #else:                                   loss_win = win_loss_fn(win_logit, target_win)

            # Calculate absolute error
            error = torch.abs(target_win - win_proba)

            # Focal MSE = Error^(2 + gamma)
            # gamma=1.0 means we are cubing the error instead of squaring it
            gamma = 0.5
            loss_win_raw = error ** (2.0 + gamma)

            # Apply the Seed Weights and average across the batch
            loss_win = (loss_win_raw * sample_weights).mean()

            # --------------------------------------------------------------------------------
            # b) Box-Score Loss
            # --------------------------------------------------------------------------------
            
            if use_box_loss: 
                # Gaussian NLL Loss vs. Standard L1/Huber Loss for older models
                if use_mean_var_loss: loss_box = gaussian_nll_loss(box_mu, target_box_score, box_log_var)
                else:                 loss_box = box_loss_fn(box_score_pred, target_box_score)
            else:            
                loss_box = torch.zeros((), device=device)
            
            # --------------------------------------------------------------------------------
            # c) Alpha-Beta Loss
            # --------------------------------------------------------------------------------
            alpha_beta_raw  = evidential_binary_loss(alpha_beta, target_win, gamma=0.5)
            alpha_beta_loss = (alpha_beta_raw * sample_weights).mean()

            # --------------------------------------------------------------------------------
            # d) Combined Loss
            # --------------------------------------------------------------------------------
            loss = (box_loss_weight * loss_box) + (win_loss_weight * loss_win) + (alpha_beta_weight * alpha_beta_loss)

            # Backpropagation if training
            if is_training:
                loss.backward()
                if grad_clip_norm is not None: nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        # --------------------------------------------------------------------------------
        # 3) Progress Tracking
        # --------------------------------------------------------------------------------
        running_loss  += float(loss    .detach().item())
        run_box_loss  += float(loss_box.detach().item()) if use_box_loss else 0.0
        run_win_loss  += float(loss_win.detach().item())
        total_batches += 1

        # --------------------------------------------------------------------------------
        # 4) Win prediction metrics
        # --------------------------------------------------------------------------------
        win_prob   = win_proba
        win_target = target_win

        # Accuracy: threshold at 0.5
        pred_labels   = (win_prob >= 0.5).float()
        batch_correct = (pred_labels == win_target).sum().item()
        running_win_correct += batch_correct
        total_win_samples   += win_target.numel()

        # MSE on probability predictions for logging
        batch_win_mse    = F.mse_loss(win_prob, win_target, reduction="mean").item()
        running_win_mse += batch_win_mse * win_target.numel()

        if progress_bar is not None: progress_bar.update(1)

    # --------------------------------------------------------------------------------
    # Progress metrics
    # --------------------------------------------------------------------------------
    # Guard counts
    metric_batches = max(total_batches,     1)
    metric_samples = max(total_win_samples, 1)

    # Per batch/sample metrics
    metrics = {
        "epoch_loss": running_loss        / metric_batches,
        "box_loss"  : run_box_loss        / metric_batches,
        "win_loss"  : run_win_loss        / metric_batches,
        "win_acc"   : running_win_correct / metric_samples,
        "win_mse"   : running_win_mse     / metric_samples,
    }

    return metrics
