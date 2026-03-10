"""
Training loop function.
--------------------------------------------------------------------------------
`src.training.training_loop`

"""
import torch, os
from torch.utils.data import DataLoader
from torch.nn         import Module

from tqdm.auto import tqdm

# From this project
from ..processing.build_dataset import BasketballDataset
from .epoch                     import run_epoch
from .metrics                   import print_metrics_header, print_train_metrics, print_train_val_metrics


# ================================================================================
# Training loop
# ================================================================================ 
def train_model_v1(
        model: Module,          # Model instance
        train_data,             # Training data (list/array accepted by BasketballDataset)
        val_data       = None,  # DataLoader for validation set (optional, can be None)
        secondary_data = None, 
        *,

        # Training config
        history       : list  = [],
        num_epochs    : int   = 10,
        batch_size    : int   = 32, 
        device        : str   = "cpu", # "mps" | "cpu" 
        learning_rate : float = 1e-4, 

        # Other config
        tourney              : str  = "W",    # M | W
        verbose              : int  = 1, 
        first_epoch_no_train : bool = True,   # Skip training on the first epoch to get a baseline

        # Saving models
        save_best : bool = False,          # True | False
        save_dir  : str  = "saved_models", # Directory to save to -- "M" or "W" will be directories under these
        save_name : str  = "test_v01",
):
    # --------------------------------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------------------------------
    train_loader = DataLoader(BasketballDataset(train_data), batch_size=batch_size, shuffle=False)
    
    if secondary_data is not None: sec_loader = DataLoader(BasketballDataset(secondary_data), batch_size=batch_size, shuffle=False)
    if       val_data is not None: val_loader = DataLoader(BasketballDataset(      val_data), batch_size=batch_size, shuffle=False)
    
    # --------------------------------------------------------------------------------
    # Prepare for training
    # --------------------------------------------------------------------------------
    # Move model to device
    model.to(device)
    
    # Create optimizer
    #optimizer = torch.optim.Adam (model.parameters(), lr=learning_rate, weight_decay=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-6)
    
    # Create scheduler
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Total number of batches for progress bar updates
    prog_bar_range = num_epochs * (len(train_data) // batch_size) 
    if secondary_data is not None: prog_bar_range += num_epochs * (len(secondary_data) // batch_size)
    
    progress_bar = tqdm(range(prog_bar_range), desc="Training", leave=True)
    
    # Print header for the metrics
    print_metrics_header()
    
    # --------------------------------------------------------------------------------
    # Track the best model
    # --------------------------------------------------------------------------------
    if save_best:
        best_val_mse = float('inf'); best_epoch = -1
        save_path    = f"{save_dir}/{tourney}"
        if not os.path.exists(save_path): os.makedirs(save_path)
        best_model_path = f"{save_path}/{save_name}.pth"
        
    # --------------------------------------------------------------------------------
    # Do the first epoch in evaluation mode
    # --------------------------------------------------------------------------------
    if first_epoch_no_train:
        # Run a training epoch using regular season data
        train_metrics = run_epoch(model, train_loader, device=device, progress_bar=progress_bar)

        # Now do the secondary tournament data
        if secondary_data is not None: secondary_metrics = run_epoch(model, sec_loader, device=device, progress_bar=progress_bar, secondary=True)
        else:                          secondary_metrics = {}

        # Validation step if provided (no optimizer passed, so evaluation mode)
        if val_data is not None:
            # Run a validation epoch and print the progress for both
            val_metrics = run_epoch(model, val_loader, device=device)
            print_train_val_metrics(0, num_epochs, history, train_metrics, secondary_metrics, val_metrics, verbose=verbose)

        else:
            # Otherwise just print the training progress on its own
            print_train_metrics(epoch, num_epochs, train_metrics)    
        
    # --------------------------------------------------------------------------------
    # Training / validation loop
    # --------------------------------------------------------------------------------
    for epoch in range(1, num_epochs):
        # Run a training epoch using regular season data
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer, progress_bar=progress_bar)
        
        # Now do the secondary tournament data
        if secondary_data is not None: secondary_metrics = run_epoch(model, sec_loader, device=device, optimizer=optimizer, progress_bar=progress_bar, secondary=True)
        else:                          secondary_metrics = {}
        
        # Validation step if provided (no optimizer passed, so evaluation mode)
        if val_data is not None:
            # Run a validation epoch and print the progress for both
            val_metrics = run_epoch(model, val_loader, device=device)
            print_train_val_metrics(epoch, num_epochs, history, train_metrics, secondary_metrics, val_metrics, verbose=verbose)
            
            # Save best model
            if save_best:
                current_val_mse = val_metrics["win_mse"]
                if current_val_mse < best_val_mse:
                    best_val_mse = current_val_mse
                    best_epoch   = epoch
                    
                    # Save the model's state_dict
                    torch.save(model.state_dict(), best_model_path)
        else:
            # Otherwise just print the training progress on its own
            print_train_metrics(epoch, num_epochs, train_metrics)
    
        # Step the scheduler after each epoch.
        #scheduler.step()
    
    progress_bar.close()
    print("Training complete!")

