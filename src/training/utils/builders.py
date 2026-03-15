"""
Builders for training objects (Optimizers, Schedulers, DataLoaders, etc.)
--------------------------------------------------------------------------------
`src.training.utils.builders`

"""
import torch

# For type highlighting
from torch.nn                 import Module
from torch.optim              import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data         import DataLoader, Dataset

# Actual optimizer & scheduler types used
from torch.optim              import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# From this project
from ...config import DEFAULT_TRAINING_CONFIG as CF


# ================================================================================
# Optimizers
# ================================================================================
def get_optimizer(
        optimizer_type : str, 
        model          : Module, 
        *, 
        lr           : float = CF.LEARNING_RATE,
        weight_decay : float = CF.WEIGHT_DECAY,
) -> Optimizer:
    # Parse the given string
    optimizer_type = optimizer_type.lower()

    # AdamW
    if optimizer_type == "adamw": 
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Always default to Adam
    else: 
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  
    
# ================================================================================
# Schedulers
# ================================================================================
def get_scheduler(scheduler_type: str, optimizer: Optimizer) -> LRScheduler | None:
    scheduler_type = scheduler_type.lower()

    # --------------------------------------------------------------------------------
    # Reduce on Plateau
    # --------------------------------------------------------------------------------
    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode     = "min",        # Depends on the metric we want to go down (usually `win_mse` here)
            factor   = 0.50,         # Cut learning rate in half when plateaued
            patience = 5,            # Wait for 5 epochs of no improvement before dropping learning rate
            min_lr   = CF.ETA_MIN,   # Minimum learning rate the scheduler will lower us to
        )
        
    # --------------------------------------------------------------------------------
    # Cosine Annealing
    # --------------------------------------------------------------------------------
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max    = CF.EPOCHS,    # Should match num_epochs
            eta_min  = CF.ETA_MIN,   # Minimum learning rate to decay to
        )
        
    return None


# ================================================================================
# DataLoaders (will probably only be called at the notebook level)
# ================================================================================
# Requiring arguments to be named to avoid any possible confusion between NCAA games and secondary tournament games
def get_dataloaders(*, rs_ds: Dataset, tr_ds: Dataset, st_ds: Dataset):
    rs_loader = DataLoader(rs_ds, batch_size=CF.BATCH_SIZE, shuffle=True )
    st_loader = DataLoader(st_ds, batch_size=CF.BATCH_SIZE, shuffle=False)
    tr_loader = DataLoader(tr_ds, batch_size=CF.BATCH_SIZE, shuffle=False)
    return rs_loader, st_loader, tr_loader
