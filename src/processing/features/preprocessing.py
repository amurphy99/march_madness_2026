"""
Initial data processing (no statistical stuff, just formatting)
--------------------------------------------------------------------------------
`src.processing.features.preprocessing`

This covers the year-team IDs to separate teams for each year, and any initial 
adjustments to the box score stats.

TODO: Is there a value in counting total attemps vs. misses?
      (this logic is currently in `general` still...)

"""

import numpy  as np
import pandas as pd


# --------------------------------------------------------------------------------
# Add the year-team ID to a DataFrame
# --------------------------------------------------------------------------------
def apply_year_team_IDs(df: pd.DataFrame) -> pd.DataFrame:
    # Working copy of the DataFrame (input must have "Season" column)
    df = df.copy()

    # There are variations of the "TeamID" column seen across data
    teamID_prefixes = ["", "W", "L"]
    for prefix in teamID_prefixes:

        # Create updated ID columns for each variation found
        if f"{prefix}TeamID" in df.columns:
            df[f"{prefix}YearTeamID"] = df["Season"].astype(str) + "_" + df[f"{prefix}TeamID"].astype(str)

    # Return the updated DataFrame
    return df

# --------------------------------------------------------------------------------
# Separate 2s and 3s in the main "Field Goals" stat
# --------------------------------------------------------------------------------
def _handle_field_goals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Change the FG column so it just counts 2s, not 2s AND 3s (is there value in total attempts vs. misses?)
    for team in ["W", "L"]:
        df[f"{team}2FGA"] = df[f"{team}FGA"] - df[f"{team}FGA3"]
        df[f"{team}2FGM"] = df[f"{team}FGM"] - df[f"{team}FGM3"]

    # Rename a few columns for clarity (so the last character is always the make/miss indicator)
    rename_dict = {}
    for team in ["W", "L"]:
        rename_dict[f"{team}FGA3"] = f"{team}3FGA"
        rename_dict[f"{team}FGM3"] = f"{team}3FGM"
    df = df.rename(columns=rename_dict)

    return df

# ================================================================================
# Apply all preprocessing methods
# ================================================================================
def apply_box_score_preprocessing(df: pd.DataFrame, do_box=True) -> pd.DataFrame:
    df = df.copy()

    # 1) Set the Year-Team IDs
    df = apply_year_team_IDs(df)

    # 2) Separate 2s and 3s (secondary tournament results have no box score)
    if do_box: df = _handle_field_goals(df)

    # 3) Sort by "Season" and "DayNum" (oldest games are first)
    df = df.sort_values(by=["Season", "DayNum"], ascending=True)

    # Return with changes
    return df

