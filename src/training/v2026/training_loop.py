"""
Training loop function
--------------------------------------------------------------------------------
`src.training.v2026.training_loop`

"""
import torch, os, time

from pathlib   import Path
from tqdm.auto import tqdm

from torch.nn                 import Module
from torch.optim              import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data         import DataLoader

# From this project
from .epoch   import run_epoch
from .metrics import print_epoch_summary, print_best_epoch

# Give it 20 epochs minimum before any scheduling or early stopping kicks in
MIN_EPOCHS = 20
    

# --------------------------------------------------------------------------------
# Run epochs for each of the three DataLoaders
# --------------------------------------------------------------------------------
def run_loader_epochs(
        model     : Module, 
        optimizer : Optimizer,

        # Book-keeping
        cur_epoch  : int,
        num_epochs : int,

        # Epoch Arguments
        rs_tr_args     : dict,
        secondary_args : dict,

        # DataLoaders
        train_loader     : DataLoader, 
        secondary_loader : DataLoader | None = None,
        val_loader       : DataLoader | None = None,
        *,
        # Additional Config
        verbose  : int  = 1,     # Whether or not to print a summary
        training : bool = True,  # False when doing an initial epoch with training off
    
) -> tuple[dict, float]:
    # 1) Train on regular season
    t0 = time.perf_counter()
    train_metrics = run_epoch(model, train_loader, optimizer=(optimizer if training else None), **rs_tr_args)

    # 2) Secondary tournament games (classification-only; no box-scores provided)
    if secondary_loader is not None: secondary_metrics = run_epoch(model, secondary_loader, **secondary_args)
    else:                            secondary_metrics = {}

    # 3) Validation (NCAA tournament)
    if val_loader is not None: val_metrics = run_epoch(model, val_loader, **rs_tr_args)
    else:                      val_metrics = {}

    # Print summary
    if verbose: print_epoch_summary(cur_epoch, num_epochs+1, train_metrics, secondary_metrics, val_metrics)

    # Update history
    epoch_history = {
        "epoch"     : cur_epoch,
        "train"     : train_metrics,
        "secondary" : secondary_metrics,
        "val"       : val_metrics,
        "trained"   : training,
    }

    # Timing
    t1 = time.perf_counter()
    epoch_time = t1 - t0

    return epoch_history, epoch_time


# ================================================================================
# Training loop
# ================================================================================
def train_model_v2(
        model: Module,
        train_loader,
        val_loader       = None,
        secondary_loader = None,
        *,

        # Training config
        history       : list | None = None,
        num_epochs    : int   = 10,
        device        : str   = "cpu",
        learning_rate : float = 1e-4,
        weight_decay  : float = 2e-6,
        first_epoch_no_train: bool = True,

        # Loss config
        box_loss_fn = None,
        win_loss_fn = None,
        box_loss_weight           : float = 1.0,
        win_loss_weight           : float = 1.0,
        secondary_win_loss_weight : float = 5.0,     # Secondary tournament data

        # Extra loss config (for alternate models)
        use_mean_var_loss: bool = False,

        # Alpha-beta loss
        use_alpha_beta    : bool  = True,
        alpha_beta_weight : float = 0.0,

        # Optimizer & Scheduler
        optimizer               : Optimizer   | None = None,
        scheduler               : LRScheduler | None = None,
        early_stopping_patience : int         | None = None, # Early Stopping
        grad_clip_norm          : float       | None = None,

        # Logging
        verbose    : int = 1,
        min_epochs : int = MIN_EPOCHS,

        # Saving models & Monitoring
        save_best    : bool = False,
        save_dir     : str  = "saved_models",
        save_name    : str  = "model_v2_best",
        save_monitor : str  = "win_mse",
        monitor_mode : str  = "min",
):
    """
    Trains the model using pre-built DataLoaders.

    Notes:
    - `train_loader`     => Training (regular season) dataset with 50% chance to flipping team order.
    - `val_loader`       => NCAA tournament games; all games included twice, once with teams in each order.
    - `secondary_loader` => Secondary tournament games; no box score data available
    """
    if history is None: history = []
    total_time = 0

    # --------------------------------------------------------------------------------
    # Prepare for training
    # --------------------------------------------------------------------------------
    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --------------------------------------------------------------------------------
    # Track the best metric for Early Stopping & Saving
    # --------------------------------------------------------------------------------
    if   monitor_mode == "min": best_metric = float( "inf")
    elif monitor_mode == "max": best_metric = float("-inf")
    else:  raise ValueError("monitor_mode must be 'min' or 'max'")

    epochs_no_improve = 0

    if save_best:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        best_model_path = save_path / f"{save_name}.pth"

    # --------------------------------------------------------------------------------
    # Setup progress bar
    # --------------------------------------------------------------------------------
    total_batches = 0

    # Get the total number of batches we will do
    total_batches += num_epochs * len(train_loader)
    if (      val_loader is not None): total_batches += num_epochs * len(      val_loader)
    if (secondary_loader is not None): total_batches += num_epochs * len(secondary_loader)

    progress_bar = tqdm(total=total_batches, desc="Training", leave=True)

    # --------------------------------------------------------------------------------
    # Define some shared epoch parameters
    # --------------------------------------------------------------------------------
    shared = dict(
        device=device, progress_bar=progress_bar, box_loss_fn=box_loss_fn, win_loss_fn=win_loss_fn,
        grad_clip_norm=grad_clip_norm, use_mean_var_loss=use_mean_var_loss,
        use_alpha_beta=use_alpha_beta, alpha_beta_weight=alpha_beta_weight,
    )

    # Regular season and NCAA tournament shared
    rs_tr_args = dict(
        **shared,
        box_loss_weight = box_loss_weight,
        win_loss_weight = win_loss_weight,
        use_box_loss    = True,
    )
    
    # Secondary tournament arguments (no box score available)
    secondary_args = dict(
        **shared,
        box_loss_weight = 0.0,
        win_loss_weight = secondary_win_loss_weight,
        use_box_loss    = False,
    )

    # --------------------------------------------------------------------------------
    # Do the first epoch in evaluation mode
    # --------------------------------------------------------------------------------
    if first_epoch_no_train:
        # Run epochs for all three DataLoaders without training
        epoch_history, _ = run_loader_epochs(
            model, optimizer, 0, num_epochs,            # Current epoch is "0" since this is before training
            rs_tr_args, secondary_args,                 # Epoch Arguments
            train_loader, secondary_loader, val_loader, # DataLoaders
            verbose=verbose, training=False,            # False when doing an initial epoch with training off    
        )
        history.append(epoch_history)

    # --------------------------------------------------------------------------------
    # Training / validation loop
    # --------------------------------------------------------------------------------
    for epoch in range(1, num_epochs+1):

        # Run epochs for all three DataLoaders; training on regular season games only
        epoch_history, epoch_time = run_loader_epochs(
            model, optimizer, epoch, num_epochs,
            rs_tr_args, secondary_args,                 # Epoch Arguments
            train_loader, secondary_loader, val_loader, # DataLoaders
            verbose=verbose, training=True, 
        )
        history.append(epoch_history)

        # ================================================================================
        # Validation Monitoring, Schedulers, & Early Stopping
        # ================================================================================
        if (val_loader is not None):
            val_metrics = epoch_history["val"]

            if (save_monitor not in val_metrics): raise KeyError(f"save_monitor='{save_monitor}' not found in validation metrics")
            current_metric = val_metrics[save_monitor]

            # Scheduler step (Handle Plateau vs Standard)
            if (scheduler is not None) and (epoch > min_epochs): 
                if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(current_metric)
                else:                                        scheduler.step()

            # Check for improvement
            improved = (monitor_mode == "min" and current_metric < best_metric) or (monitor_mode == "max" and current_metric > best_metric)

            # Saving best model
            if improved:
                best_metric = current_metric; epochs_no_improve = 0
                if save_best: torch.save(model.state_dict(), best_model_path)
            else: epochs_no_improve += 1
            
            # Early Stopping Check
            if (early_stopping_patience is not None) and (epochs_no_improve >= early_stopping_patience) and (epoch > min_epochs):
                if verbose:  print(f"\nEarly stopping triggered! No improvement in '{save_monitor}' for {early_stopping_patience} epochs.")
                break
        else:
            # If no validation loader, just step standard schedulers
            if (scheduler is not None) and not isinstance(scheduler, ReduceLROnPlateau): scheduler.step()

   
        # Time check
        total_time += epoch_time
        progress_bar.set_postfix({"sec/epoch": f"{(total_time / epoch):.4}s"})

    # --------------------------------------------------------------------------------
    # Training finished
    # --------------------------------------------------------------------------------
    progress_bar.close()

    # Print metrics from the best epoch
    if verbose: print_best_epoch(history, num_epochs=num_epochs+1)

    if save_best and (val_loader is not None):
        print(f"Best model saved to: {best_model_path}")

    return history
