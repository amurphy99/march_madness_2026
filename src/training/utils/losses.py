"""
Custom training loss functions.
--------------------------------------------------------------------------------
`src.training.utils.losses`

"""
import torch


# --------------------------------------------------------------------------------
# Custom Negative Log-Likelihood loss for a Gaussian distribution 
# --------------------------------------------------------------------------------
def gaussian_nll_loss(mu: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Model predicts mean and variance.
    """
    squared_error = (target - mu) ** 2
    
    # Calculate the element-wise NLL
    loss = 0.5 * (log_var + squared_error * torch.exp(-log_var))
    
    # Return the mean loss across the batch
    return loss.mean()


