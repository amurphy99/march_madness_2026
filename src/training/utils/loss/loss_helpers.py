"""
Separate helper functions that automatically get loss for a prediction head.
--------------------------------------------------------------------------------
`src.training.utils.loss.loss_helpers`

For example, the box-score loss stats can have multiple outputs and expected
loss formats (normal vs. mean-variance loss). Helper methods here should be able
to get the correct loss for the `run_epoch` function automatically.

TODO: Finish the box-score loss helper

"""
import torch



# ================================================================================
# TODO: Get box score loss according to parameters
# ================================================================================
# More of a flexible helper for the `run_epoch` function
def box_score_loss(
        target_box_score: torch.tensor,
        
        box_score_pred: torch.tensor,
        box_mu      : torch.tensor,
        box_log_var: torch.tensor,

        box_loss_fn,
        device,

        use_box_loss: bool = True,
        use_mean_var_loss: bool = True,
        
):
    if not use_box_loss: torch.zeros((), device=device)
    pass











