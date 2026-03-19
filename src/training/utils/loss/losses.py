"""
Custom training loss functions.
--------------------------------------------------------------------------------
`src.training.utils.losses`

TODO: Move into the 'loss' subdirectory and update imports

"""
import torch


# ================================================================================
# Custom Negative Log-Likelihood loss for a Gaussian distribution.
# ================================================================================
def gaussian_nll_loss(mu: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Model predicts mean and variance.
    """
    squared_error = (target - mu) ** 2
    
    # Calculate the element-wise NLL
    loss = 0.5 * (log_var + squared_error * torch.exp(-log_var))
    
    # Return the mean loss across the batch
    return loss.mean()

# ================================================================================
# Evidential MSE Loss for binary classification using a Beta distribution.
# ================================================================================
def evidential_binary_loss_v1(alpha_beta: torch.Tensor, target_win: torch.Tensor, gamma=-0.5) -> torch.Tensor:
    """
    [FOCAL]
    alpha_beta: Shape (Batch, 2) containing [alpha, beta]
    target_win: Shape (Batch,) containing 1.0 or 0.0
    """
    # Convert the 1D target into a 2D one-hot target: [Win, Loss]
    # If target_win is 1.0 -> y = [1.0, 0.0]; If target_win is 0.0 -> y = [0.0, 1.0]
    y = torch.stack([target_win, 1.0 - target_win], dim=-1)
    
    # Calculate probability from evidence and total strength (S)
    S = torch.sum(alpha_beta, dim=-1, keepdim=True)
    p = alpha_beta / S                 # (B, 2)

    # Variance penalty with (penalize high uncertainty when wrong)
    variance = (p * (1 - p)) / (S + 1.0)             # (B, 2)
    
    # Focal error term 
    error      = torch.abs(y - p)                    # (B, 2)
    focal_loss = error ** (2.0 + gamma)              # (B, 2)
    
    # Combine and sum across the two classes 
    loss = torch.sum(focal_loss + variance, dim=-1)  # (B,)
 
    # Win probability for logging / monitoring
    win_proba = p[:, 0]                              # (B,)
    
    return loss, win_proba

import torch





# ================================================================================
# KL divergence from Beta(alpha, beta) to the uniform prior Beta(1, 1)
# ================================================================================
def kl_beta_to_uniform(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    KL( Beta(alpha, beta) || Beta(1, 1) )

    Used to penalize wrong-class evidence — measures how far the model's
    wrong-side Beta distribution has drifted from a flat (no-evidence) prior.
    """
    S = alpha + beta
    return (
          torch.lgamma(S)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))
        + (beta  - 1.0) * (torch.digamma(beta)  - torch.digamma(S))
    )


# ================================================================================
# Evidential Loss for binary classification using a Beta distribution.
# ================================================================================
def evidential_binary_loss(
        alpha_beta  : torch.Tensor,  # Shape (B, 2) — [alpha, beta]
        target_win  : torch.Tensor,  # Shape (B,)   — 1.0 or 0.0
        gamma       : float = 0.0,   # Focal exponent: loss ∝ error^(2+gamma)
        kl_weight   : float = 5e-5,  # Weight on the wrong-class KL penalty
        anneal_coef : float = 1.0,   # Scale the KL term (1.0 = fully on)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (per_sample_loss, win_proba) — shapes (B,) and (B,).

    Two terms:
      1) Data-fit  — focal error^(2+gamma) + variance penalty (same as v1)
      2) KL penalty — penalizes evidence that accumulated on the WRONG side by
                      measuring KL( Beta(wrong_evidence) || Beta(1,1) )

    The KL term is what fixes the confidence cap: the model is now explicitly
    punished for being confidently wrong, not just inaccurate on average.
    """
    # --------------------------------------------------------------------------------
    # Unpack and compute expected probabilities
    # --------------------------------------------------------------------------------
    alpha = alpha_beta[:, 0]    # (B,)
    beta  = alpha_beta[:, 1]    # (B,)

    S     = alpha + beta        # (B,)
    p_win = alpha / S           # (B,)

    # Stack into (B, 2) for the vector loss terms
    p = torch.stack([p_win, 1.0 - p_win], dim=-1)              # (B, 2)
    y = torch.stack([target_win, 1.0 - target_win], dim=-1)    # (B, 2)

    # --------------------------------------------------------------------------------
    # Data-fit term (focal error + variance penalty)
    # --------------------------------------------------------------------------------
    error    = torch.abs(y - p)                                 # (B, 2)
    focal    = error ** (2.0 + gamma)                           # (B, 2)
    variance = (p * (1.0 - p)) / (S.unsqueeze(-1) + 1.0)       # (B, 2)

    data_fit = torch.sum(focal + variance, dim=-1)              # (B,)

    # --------------------------------------------------------------------------------
    # Wrong-class KL penalty
    # --------------------------------------------------------------------------------
    # Zero out correct-class evidence; keep only the evidence that accumulated on
    # the wrong side. Penalize how far that wrong-side Beta is from the flat prior.
    #   y=1 (win)  → alpha_tilde=1, beta_tilde=beta  (penalize beta)
    #   y=0 (loss) → alpha_tilde=alpha, beta_tilde=1  (penalize alpha)
    alpha_tilde = target_win       + (1.0 - target_win) * alpha  # (B,)
    beta_tilde  = (1.0 - target_win) + target_win        * beta  # (B,)

    kl_penalty = kl_beta_to_uniform(alpha_tilde, beta_tilde)     # (B,)

    # --------------------------------------------------------------------------------
    # Combined loss
    # --------------------------------------------------------------------------------
    loss = data_fit + (anneal_coef * kl_weight * kl_penalty)    # (B,)

    return loss, p_win

