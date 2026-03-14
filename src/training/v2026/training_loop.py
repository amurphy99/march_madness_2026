"""
Training loop function
--------------------------------------------------------------------------------
`src.training.v2026.training_loop`

"""
import torch, os

from pathlib   import Path
from tqdm.auto import tqdm
from torch.nn                 import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau

# From this project
from .epoch   import run_epoch
from .metrics import print_epoch_summary, print_best_epoch

MIN_EPOCHS = 40

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

        # Optimizer / scheduler
        optimizer = None,
        scheduler = None,
        grad_clip_norm          : float | None = None,
        early_stopping_patience : int   | None = None, # Early Stopping

        # Logging
        verbose: int = 1,

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
        device            = device,
        progress_bar      = progress_bar,
        box_loss_fn       = box_loss_fn,
        win_loss_fn       = win_loss_fn,
        grad_clip_norm    = grad_clip_norm,
        use_mean_var_loss = use_mean_var_loss,
    )

    # Regular season and NCAA tournament shared
    rs_tr_args = dict(
        **shared,
        box_loss_weight = box_loss_weight,
        win_loss_weight = win_loss_weight,
        use_box_loss    = True,
    )
    
    # Secondary tournament arguments (no box score available)
    secondary = dict(
        **shared,
        box_loss_weight = 0.0,
        win_loss_weight = secondary_win_loss_weight,
        use_box_loss    = False,
    )

    # --------------------------------------------------------------------------------
    # Do the first epoch in evaluation mode
    # --------------------------------------------------------------------------------
    if first_epoch_no_train:
        # Train on regular season
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, **rs_tr_args)

        # Secondary tournaments (no box score data)
        if secondary_loader is not None: secondary_metrics = run_epoch(model, secondary_loader, **secondary)
        else:                            secondary_metrics = {}

        # Validation (NCAA tournament)
        if val_loader is not None: val_metrics = run_epoch(model, val_loader, **rs_tr_args)
        else:                      val_metrics = {}

        if verbose: print_epoch_summary(0, num_epochs+1, train_metrics, secondary_metrics, val_metrics)

        # Update history
        history.append({
            "epoch"     : 0,
            "train"     : train_metrics,
            "secondary" : secondary_metrics,
            "val"       : val_metrics,
            "trained"   : False,
        })

    # --------------------------------------------------------------------------------
    # Training / validation loop
    # --------------------------------------------------------------------------------
    for epoch in range(1, num_epochs+1):
        # 1) Train on regular season
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, **rs_tr_args)

        # 2) Train on secondary tournament if provided (classification-only)
        if secondary_loader is not None: secondary_metrics = run_epoch(model, secondary_loader, **secondary)
        else:                            secondary_metrics = {}

        # 3) Validation (NCAA tournament)
        if val_loader is not None: val_metrics = run_epoch(model, val_loader, **rs_tr_args)
        else:                      val_metrics = {}

        # Print summary
        if verbose: print_epoch_summary(epoch, num_epochs+1, train_metrics, secondary_metrics, val_metrics)

        # ================================================================================
        # Validation Monitoring, Schedulers, & Early Stopping
        # ================================================================================
        if val_loader is not None:
            if save_monitor not in val_metrics: raise KeyError(f"save_monitor='{save_monitor}' not found in validation metrics")
            current_metric = val_metrics[save_monitor]

            # Scheduler step (Handle Plateau vs Standard)
            if (scheduler is not None) and (epoch > MIN_EPOCHS): 
                if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(current_metric)
                else:                                        scheduler.step()

            # Check for improvement
            improved = (
                (monitor_mode == "min" and current_metric < best_metric) or
                (monitor_mode == "max" and current_metric > best_metric)
            )

            if improved:
                best_metric       = current_metric
                epochs_no_improve = 0
                if save_best: torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
            
            # Early Stopping Check
            if (early_stopping_patience is not None) and (epochs_no_improve >= early_stopping_patience) and (epoch > MIN_EPOCHS):
                if verbose: 
                    print(f"\nEarly stopping triggered! No improvement in '{save_monitor}' for {early_stopping_patience} epochs.")
                
                # We need to append the final history before breaking so the charts match
                history.append({
                    "epoch"    : epoch,
                    "train"    : train_metrics,
                    "secondary": secondary_metrics,
                    "val"      : val_metrics,
                    "trained"  : True,
                })
                break
        else:
            # If no validation loader, just step standard schedulers
            if (scheduler is not None) and not isinstance(scheduler, ReduceLROnPlateau): scheduler.step()


        # Update history
        history.append({
            "epoch"    : epoch,
            "train"    : train_metrics,
            "secondary": secondary_metrics,
            "val"      : val_metrics,
            "trained"  : True,
        })

    # --------------------------------------------------------------------------------
    # Training finished
    # --------------------------------------------------------------------------------
    progress_bar.close()

    # Print metrics from the best epoch
    if verbose: print_best_epoch(history, num_epochs=num_epochs+1)

    if save_best and (val_loader is not None):
        print(f"Best model saved to: {best_model_path}")

    return history
