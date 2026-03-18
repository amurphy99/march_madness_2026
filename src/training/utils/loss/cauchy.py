"""
Cauchy Loss (or "Lorentzian Loss")
--------------------------------------------------------------------------------
`src.training.utils.loss.cauchy`

More stability to outliers (e.g., Purdue losing in the first round as a 1 seed).

"""
import torch
import torch.nn as nn


# ================================================================================
# Cauchy Loss as a PyTorch Module
# ================================================================================
class CauchyLoss(nn.Module):
    def __init__(self, c: float = 1.0, reduction: str = "mean"):
        """
        c: The scale parameter. Larger c makes it behave more like MSE. 
           Smaller c makes it more robust to outliers.
        """
        super().__init__()
        self.c         = c
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate the raw error
        error = pred - target
        
        # Apply the Cauchy / Lorentzian formula
        loss = (self.c ** 2) * torch.log(1.0 + (error / self.c) ** 2)
        
        # Apply reductions
        if   self.reduction == "mean" : return loss.mean()
        elif self.reduction == "sum"  : return loss.sum ()
        
        return loss # 'none' (returns sample-wise losses)


# ================================================================================
# Cauchy Negative Log-Likelihood (for uncertainty/variance prediction modes)
# ================================================================================
def cauchy_nll_loss(
        target : torch.Tensor,       # The actual ground truth
        pred   : torch.Tensor,       # The predicted mean (e.g., predicted margin)
        var    : torch.Tensor,       # The predicted variance (must be > 0)
        *,
        epsilon     : float = 1e-6,  # Prevent divide by 0 errors
        return_mean : bool  = True,  # Don't return the mean yet if I need to weigh the losses for each sample
) -> torch.Tensor:
    """
    Calculates the Negative Log-Likelihood of a Cauchy distribution.
    Should be better for extreme outliers (e.g., massive unpredictable blowouts).
    """
    # Clamp variance to prevent division by zero or log(0)
    var = torch.clamp(var, min=epsilon)
    
    # Calculate squared error
    error_sq = (pred - target) ** 2
    
    # Calculate the Cauchy NLL
    # ln(1 + E^2 / V) + 0.5 * ln(V)
    loss = torch.log(1.0 + (error_sq / var)) + (0.5 * torch.log(var))
    
    # Return the batch mean or just the loss vector if I want to weigh it later
    if return_mean: return loss.mean()
    else:           return loss

