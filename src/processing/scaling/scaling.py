"""
Scaling utilities for the project.
--------------------------------------------------------------------------------
`src.processing.features.scaling`

Load in the correct scaler according to the data being used.

"""
import pandas as pd
import joblib, os

from sklearn.preprocessing import StandardScaler

# From this project
from ...config import W_TEAM_STAT_COLS, L_TEAM_STAT_COLS, TEAM_BOX_SCORE_COLUMNS


# --------------------------------------------------------------------------------
# Import saved scalers (not optimal, but they are small so it's fine)
# --------------------------------------------------------------------------------
# Saved scaler objects are just stored right in this directory
SCALERS_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/saved_scalers"

# Default scaler to load if no scaler exists for the given parameters
DEFAULT_SCALER_PATH = f"{SCALERS_PATH}/M_07y_box_scaler.pkl"

# --------------------------------------------------------------------------------
# Get the correct scaler to use
# --------------------------------------------------------------------------------
def _get_scaler(tourney: str, years: int) -> StandardScaler:
    # Get the path to the scaler using the params
    scaler_path = f"{SCALERS_PATH}/{tourney}_{years:02d}y_box_scaler.pkl"

    # Make sure it exists
    if os.path.exists(scaler_path): return joblib.load(scaler_path)
    else:                           return joblib.load(DEFAULT_SCALER_PATH)

# ================================================================================
# Apply the scaler to a box score DataFrame 
# ================================================================================
def scale_box_scores(df: pd.DataFrame, tourney: str, years: int):
    # 1. Split the DataFrame into winning/losing team stats
    W_df = df[W_TEAM_STAT_COLS].copy()
    L_df = df[L_TEAM_STAT_COLS].copy()

    # Rename the columns to neutral names 
    W_df.columns = TEAM_BOX_SCORE_COLUMNS
    L_df.columns = TEAM_BOX_SCORE_COLUMNS

    # Load & apply the scaler
    scaler = _get_scaler(tourney, years)  # (get the correct scaler for the given sex/historical years)
    W_scaled = scaler.transform(W_df)
    L_scaled = scaler.transform(L_df)

    # Put them back into the original DataFrame & return
    df = df.copy()
    df[W_TEAM_STAT_COLS] = W_scaled
    df[L_TEAM_STAT_COLS] = L_scaled
    return df
