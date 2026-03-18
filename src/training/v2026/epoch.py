"""
Run training or validation epochs.
--------------------------------------------------------------------------------
`src.training.v2026.epoch`

"""
import torch
import torch.nn            as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from tqdm.auto   import tqdm

# From this project
from ..utils.loss.loss_tracker import TournamentLossComputer
from ..utils.average_meter     import AverageMeter

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
        data_loader,                                # Data to run on
        loss_computer: TournamentLossComputer,      # Modularized loss handling
        *,
        optimizer      : Optimizer | None = None,   # Not given for validation runs
        progress_bar   : tqdm      | None = None,   # Overall progress bar to update
        grad_clip_norm : float     | None = None,   # Optional gradient clipping
        device         : str       | None = "cpu",  # Device to use
):
    """
    Returns a metrics dict with: epoch_loss, box_loss, win_loss, win_acc, win_mse
    """
    # Optimizer only provided if we are in training mode
    is_training = optimizer is not None
    if is_training: model.train()
    else:           model.eval()

    # Tracking Progress
    loss_meter     = AverageMeter()
    box_loss_meter = AverageMeter()
    win_loss_meter = AverageMeter()
    win_acc_meter  = AverageMeter()
    win_mse_meter  = AverageMeter()

    # Run each batch through the model
    for batch in data_loader:
        # --------------------------------------------------------------------------------
        # 1) Prepare the input data
        # --------------------------------------------------------------------------------
        batch      = _move_batch_to_device(batch, device)
        target_win = batch["target_win"].float()

        if is_training: optimizer.zero_grad(set_to_none=True)

        # ================================================================================
        # 2) Forward pass
        # ================================================================================
        with torch.set_grad_enabled(is_training):

            # 1) Forward pass
            model_outputs = model(batch)

            # 2) Calculate loss using the handler module
            loss, loss_dict = loss_computer(model_outputs, batch, device)

            # 3) Backpropagation (if training)
            if is_training:
                loss.backward()
                if grad_clip_norm is not None: nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        # ================================================================================
        # 3) Update Metrics
        # ================================================================================
        # a) Update Loss/Batch-Level Meters (n=1 by default)
        loss_meter    .update(loss_dict["total"].item())
        win_loss_meter.update(loss_dict["win"  ].item())
        box_loss_meter.update(loss_dict["box"  ].item())

        # b) Update Sample-Level Meters (n = batch_size)
        win_prob   = loss_dict["win_proba"]
        win_target = target_win.view_as(win_prob) # Make sure target_win has the same shape as win_logit (e.g., [batch_size, 1])
        batch_size = win_target.numel()           # Get number of samples in this batch

        # Accuracy (threshold at 0.5)
        pred_labels   = (win_prob >= 0.5).float()
        batch_correct = (pred_labels == win_target).sum().item()
        batch_acc     = batch_correct / batch_size
        win_acc_meter.update(batch_acc, batch_size)

        # MSE on probability predictions for logging
        batch_win_mse = F.mse_loss(win_prob, win_target, reduction="mean").item()
        win_mse_meter.update(batch_win_mse, batch_size)

        # Update the progress bar
        if progress_bar is not None: progress_bar.update(1)

    # --------------------------------------------------------------------------------
    # Report Metrics
    # --------------------------------------------------------------------------------
    # Return the averages from the helper objects
    metrics = {
        "epoch_loss": loss_meter.avg,
        "box_loss"  : box_loss_meter.avg,
        "win_loss"  : win_loss_meter.avg,
        "win_acc"   : win_acc_meter.avg,
        "win_mse"   : win_mse_meter.avg,
    }

    return metrics
