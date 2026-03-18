"""
Modulerized loss calculations
--------------------------------------------------------------------------------
`src.training.utils.loss.loss_tracker`

Don't need to touch "run_epoch()" at all anymore when I go to test out different
loss setups.

TODO: Could also add specific "target_A_points" and "target_B_points" to the loss

TODO: I REALLY want to rename this to "LossHandler" everywhere...

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# From this project
from    .losses                 import gaussian_nll_loss, evidential_binary_loss
from    .cauchy                 import CauchyLoss, cauchy_nll_loss  
from ....models.utils.parse_box import extract_points


# ================================================================================
# Model Definition
# ================================================================================
class TournamentLossComputer(nn.Module):
    def __init__(
        self,

        # Specific loss functions to use (not really used anymore due to more customized calculations)
        box_loss_fn = None,
        win_loss_fn = None,

        # If False, skip the loss entirely
        use_box_loss      : bool  = True,
        use_mean_var_loss : bool  = True,
        use_alpha_beta    : bool  = True,
        use_margin_loss   : bool  = False,

        # Weights for each loss function (or task)
        box_loss_weight    : float = 1.0,
        win_loss_weight    : float = 1.0,
        alpha_beta_weight  : float = 1.0,
        margin_loss_weight : float = 1.0,

        # Cauchy Loss Configuration 
        use_cauchy_margin  : bool = True,   # Toggle Cauchy for the margin
        use_cauchy_nll     : bool = False,  # Standard or NLL version 

        # Gamma value adjusts the "focal" modifier strength on the loss
        gamma : float = 0.5,  # gamma=1.0 means we are cubing the error instead of squaring it
    ):
        super().__init__()

        # Default loss functions (if not provided)
        self.box_loss_fn = box_loss_fn if box_loss_fn is not None else F.l1_loss
        self.win_loss_fn = win_loss_fn if win_loss_fn is not None else nn.BCEWithLogitsLoss()
        
        # If False, skip the loss entirely
        self.use_box_loss      = use_box_loss
        self.use_mean_var_loss = use_mean_var_loss
        self.use_alpha_beta    = use_alpha_beta
        self.use_margin_loss   = use_margin_loss

        # Weights for each loss function (or task)
        self.box_loss_weight    = box_loss_weight
        self.win_loss_weight    = win_loss_weight
        self.alpha_beta_weight  = alpha_beta_weight
        self.margin_loss_weight = margin_loss_weight

        # Cauchy loss configuration
        self.use_cauchy_margin = use_cauchy_margin
        self.use_cauchy_nll    = use_cauchy_nll
        self.cauchy_loss_fn    = CauchyLoss()       # Initialize the torch module version

        # Gamma value adjusts the "focal" modifier strength on the loss
        self.gamma = gamma

    # ================================================================================
    # Loss Calculations
    # ================================================================================
    def forward(self, model_outputs, batch, device) -> tuple[torch.tensor, dict[str, torch.tensor]]:
        """
        Computes the combined tournament loss.
        Returns: total_loss (Tensor), loss_dict (dict for logging)
        """
        # --------------------------------------------------------------------------------
        # 1) Unpack model outputs based on architecture
        # --------------------------------------------------------------------------------
        # The "box_mu" output from the mean-variance heads renamed to "box_pred"
        if   self.use_alpha_beta    : (box_pred, box_log_var), win_logit, alpha_beta = model_outputs
        elif self.use_mean_var_loss : (box_pred, box_log_var), win_logit             = model_outputs
        else:                          box_pred,               win_logit             = model_outputs

        # Get target values from the batch (make sure target_win has the same shape as win_logit (e.g., [batch_size, 1]))
        target_box_score = batch["target_box_score"].float()
        target_win       = batch["target_win"      ].float().view_as(win_logit) 
        target_margin    = batch["target_margin"   ].float().view_as(win_logit)

        # --------------------------------------------------------------------------------
        # 2) Calculate Sample Weights (Seed Modifiers)
        # --------------------------------------------------------------------------------
        # batch["teamA_seed"] is > 0 if they made the tournament, 0 otherwise
        A_is_tourney = (batch["teamA_seed"] > 0).float()
        B_is_tourney = (batch["teamB_seed"] > 0).float()

        # Base game weight is 1.0 (1 Tourney Team = 2.0 weight; 2 Tourney Teams = 3.0 weight)
        sample_weights = 1.0 + A_is_tourney + B_is_tourney

        # Normalize the weights so the batch average remains 1.0 (prevents learning rate from exploding)
        sample_weights = sample_weights / sample_weights.mean() 

        # --------------------------------------------------------------------------------
        # 3) Win Prediction Loss (Focal MSE)
        # --------------------------------------------------------------------------------
        # All models now always return the logit only; do sigmoid here for MSELoss
        win_proba = torch.sigmoid(win_logit)

        # Calculate absolute error
        error = torch.abs(target_win - win_proba)

        # Focal MSE = Error^(2 + gamma)
        loss_win_raw = error ** (2.0 + self.gamma)

        # Apply the Seed Weights and average across the batch
        loss_win = (loss_win_raw * sample_weights).mean()

        # --------------------------------------------------------------------------------
        # 4) Box-Score Loss
        # --------------------------------------------------------------------------------
        loss_box = torch.zeros((), device=device)
        if self.use_box_loss:
            if self.use_mean_var_loss: loss_box = gaussian_nll_loss(box_pred, target_box_score, box_log_var)
            else:                      loss_box = self.box_loss_fn (box_pred, target_box_score)

        # --------------------------------------------------------------------------------
        # 5) Alpha-Beta Loss
        # --------------------------------------------------------------------------------
        loss_alpha_beta = torch.zeros((), device=device)
        if self.use_alpha_beta:
            alpha_beta_raw = evidential_binary_loss(alpha_beta, target_win, gamma=self.gamma)
            loss_alpha_beta = (alpha_beta_raw * sample_weights).mean()

        # --------------------------------------------------------------------------------
        # 6) Points Margin Loss (Cauchy / NLL)
        # --------------------------------------------------------------------------------
        # TODO: Going to start with typical cauchy loss for now, not the mean-var version
        loss_margin = torch.zeros((), device=device)
        if self.use_margin_loss:
            # a) Get predictions and variance
            margin_pred, margin_std, _, _ = extract_points(box_pred, box_log_var)
            margin_var = margin_std ** 2 
            
            # b) Calculate Loss
            if self.use_cauchy_margin:
                # Use Cauchy Negative Log-Likelihood
                if self.use_cauchy_nll: 
                     margin_loss_raw = cauchy_nll_loss(
                        target      = target_margin, 
                        pred        = margin_pred, 
                        var         = margin_var, 
                        return_mean = False # Keep sample dimensions
                    )
                # Use standard Cauchy loss
                else: margin_loss_raw = self.cauchy_loss_fn(margin_pred, target_margin)
            
            # Get normal MSE loss of Cauchy is disabled
            else: margin_loss_raw = F.mse_loss(margin_pred, target_margin, reduction="none")
            
            # Apply the seed weights and average across the batch
            loss_margin = (margin_loss_raw * sample_weights).mean()

        # --------------------------------------------------------------------------------
        # 7) Combined Loss
        # --------------------------------------------------------------------------------
        total_loss = (
            (self.box_loss_weight    * loss_box       ) +
            (self.win_loss_weight    * loss_win       ) +
            (self.alpha_beta_weight  * loss_alpha_beta) +
            (self.margin_loss_weight * loss_margin    )
        )

        # Pack individual components into a dictionary for easy logging in run_epoch
        loss_dict = {
            # Primary loss components
            "total"       : total_loss,
            "box"         : loss_box,
            "win"         : loss_win,
            "margin"      : loss_margin, 

            # Passed back so run_epoch can calculate metrics
            "win_proba"   : win_proba,

            # Passed back for logging convenience
            "margin_pred" : margin_pred, 
            "margin_var"  : margin_var,  
        }

        return total_loss, loss_dict
    
