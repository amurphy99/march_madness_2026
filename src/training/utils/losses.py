"""
Custom training loss functions.
--------------------------------------------------------------------------------
`src.training.utils.losses`

"""
import torch


# --------------------------------------------------------------------------------
# Custom Negative Log-Likelihood loss for a Gaussian distribution.
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


# --------------------------------------------------------------------------------
# Evidential MSE Loss for binary classification using a Beta distribution.
# --------------------------------------------------------------------------------
def evidential_binary_loss(alpha_beta: torch.Tensor, target_win: torch.Tensor) -> torch.Tensor:
    """
    alpha_beta: Shape (Batch, 2) containing [alpha, beta]
    target_win: Shape (Batch,) containing 1.0 or 0.0
    """
    # 1) Convert the 1D target into a 2D one-hot target: [Win, Loss]
    # If target_win is 1.0 -> y = [1.0, 0.0]; If target_win is 0.0 -> y = [0.0, 1.0]
    y = torch.stack([target_win, 1.0 - target_win], dim=-1)
    
    # 2) Calculate Evidence and Total Strength (S)
    S = torch.sum(alpha_beta, dim=-1, keepdim=True)
    
    # 3) Expected Probabilities & Standard MSE (squared error)
    p   = alpha_beta / S               # (B, 2)
    mse = (y - p) ** 2                 # (B, 2)   
    
    # 4) Beta Distribution Variance Penalty
    # This penalizes predicting high uncertainty when the network is wrong
    variance = (p * (1 - p)) / (S + 1.0)
    
    # Sum the errors across the two classes, then take the mean of the batch
    loss = torch.sum(mse + variance, dim=-1)
    return loss.mean()

