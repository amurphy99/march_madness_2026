"""
Parse the box-score prediction head for specific statistics.
--------------------------------------------------------------------------------
`src.models.utils.parse_box`

Here we will use the box-score prediction head to extract a predicted score for
each team to use as the points/margin. 

TODO: Probably kind of a stupid way to organize/place these functions, but whatever...
TODO: Also should probably just get one like points getter method that these reuse?

"""
import numpy as np
import torch

# From this project
from ...config import POINT_COL_IDX as BOX


# ================================================================================
# Extract the predicted points and confidence from the existing box score heads
# ================================================================================
def extract_points(box_mu: torch.tensor, box_var: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    # Points = (2FGM * 2) + (3FGM * 3) + (FTM * 1)
    A_points = (box_mu[:, BOX["W2FGM"]] * 2.0) + (box_mu[:, BOX["W3FGM"]] * 3.0) + (box_mu[:, BOX["WFTM"]] * 1.0)
    B_points = (box_mu[:, BOX["L2FGM"]] * 2.0) + (box_mu[:, BOX["L3FGM"]] * 3.0) + (box_mu[:, BOX["LFTM"]] * 1.0)

    # Variance of (aX + bY + cZ) = a^2*Var(X) + b^2*Var(Y) + c^2*Var(Z)
    A_var    = (box_var[:, BOX["W2FGM"]] * 4.0) + (box_var[:, BOX["W3FGM"]] * 9.0) + (box_var[:, BOX["WFTM"]] * 1.0)
    B_var    = (box_var[:, BOX["L2FGM"]] * 4.0) + (box_var[:, BOX["L3FGM"]] * 9.0) + (box_var[:, BOX["LFTM"]] * 1.0)

    # Points margin
    margin_pred = A_points - B_points

    # Variance of (A - B) is Var(A) + Var(B) (assuming the heads predict independently)
    margin_var = A_var + B_var
    margin_std = torch.sqrt(margin_var)

    return margin_pred, margin_std, A_points, B_points


# --------------------------------------------------------------------------------
# Helper that parses a regular box score array to return a margin value
# --------------------------------------------------------------------------------
# Assuming that the "Score" column is not included
def calculate_margin(box_score: np.array) -> tuple[np.array, np.array, np.array]:
    """
    We get the indices of where the TeamA and TeamB column values would be using 
    the 'W' and 'L' indices from the config.
    """
    A_points = (box_score[BOX["W2FGM"]] * 2.0) + (box_score[BOX["W3FGM"]] * 3.0) + (box_score[BOX["WFTM"]] * 1.0)
    B_points = (box_score[BOX["L2FGM"]] * 2.0) + (box_score[BOX["L3FGM"]] * 3.0) + (box_score[BOX["LFTM"]] * 1.0)

    # Points margin
    margin = A_points - B_points

    # Return all three
    return margin, A_points, B_points

