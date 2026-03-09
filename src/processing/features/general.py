"""
General data processing utilities.
--------------------------------------------------------------------------------
`src.processing.features.general`

This is based off of my code from 2025. 

Going to store things in this file while I adapt old code and create places for
different things.

TODO: Need to explicitly map out what the IDs are

"""
import numpy  as np
import pandas as pd

# From this project
from ...config import TEAM_BOX_SCORE_COLUMNS


# ================================================================================
# Add a seed column to the game data using the seed data
# ================================================================================
def prep_seeds_df(seeds_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    # Trim off the end pieces that are sometimes on the seeds
    seeds_df["Seed"] = seeds_df["Seed"].astype(str).str[:3]

    # Map seeds to int
    unique_seeds = pd.unique(seeds_df["Seed"])
    seed_to_int  = {seed_str: (idx+1) for idx, seed_str in enumerate(unique_seeds)}
    seeds_df["Seed"] = seeds_df["Seed"].map(seed_to_int)

    # Add the year ID to the seeds DataFrame & create a dictionary to map seeds to teams with
    seeds_df["year_ID_str"] = seeds_df["Season"].astype(str) + "_" + seeds_df["TeamID"].astype(str)
    seed_for_team = dict(zip(seeds_df["year_ID_str"], seeds_df["Seed"]))
    
    # Return three things
    return seeds_df, unique_seeds, seed_for_team



# ================================================================================
# Map the IDs and Seeds to ints
# ================================================================================
def prep_for_embeddings(
        df             : pd.DataFrame, 
        seed_for_team, 
        team_ID_to_int : dict[str, int] = None, 
        do_stats       : bool           = True, 
        verbose        : bool           = False,
):
    """
    `team_ID_to_int` => Generated when we call this for the training data, and then we provide it, 
                        pre-created, for the validation data. These are for the embeddings. 
    """
    # Start by getting all of the team IDs (separated by year)
    df["W_year_ID_str"] = df["Season"].astype(str) + "_" + df["WTeamID"].astype(str)
    df["L_year_ID_str"] = df["Season"].astype(str) + "_" + df["LTeamID"].astype(str)
    
    # Use the year ID strings to map seeds to this dataframe
    df["W_Seed"] = df["W_year_ID_str"].map(lambda team_ID: seed_for_team.get(team_ID, 0))
    df["L_Seed"] = df["L_year_ID_str"].map(lambda team_ID: seed_for_team.get(team_ID, 0))
        
    # Now map the team IDs to ints for the embeddings
    if team_ID_to_int is None:
        # Concatenate the two new columns into a single Series and get the uniques for a list of all teams
        unique_teams_str = pd.unique(pd.concat([df["W_year_ID_str"], df["L_year_ID_str"]]))
        team_ID_to_int   = {team_str: idx+1 for idx, team_str in enumerate(unique_teams_str)}

    # Convert the DataFrame columns
    df["W_year_ID"] = df["W_year_ID_str"].map(lambda team_ID: team_ID_to_int.get(team_ID, 0))
    df["L_year_ID"] = df["L_year_ID_str"].map(lambda team_ID: team_ID_to_int.get(team_ID, 0))

    # Concatenate the two new columns into a single Series and get the uniques for a list of all teams
    unique_teams = pd.unique(pd.concat([df["W_year_ID"], df["L_year_ID"]]))
    print(f"Number of teams: {len(unique_teams):,}")
    
    # --------------------------------------------------------------------------------
    # Prepare box score stats
    # --------------------------------------------------------------------------------
    if do_stats:
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

        # Get the stat columns that will be used
        stat_columns = ["OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

        # Do the field goals and free throws in a loop
        shot_stats = []
        for shot_type in ["2FG", "3FG", "FT"]:
            for make in ["M", "A"]:
                shot_stats.append(f"{shot_type}{make}")

        # Combine them into one list
        team_box_score_columns = shot_stats + stat_columns
        if verbose:
            print(f"\n# of stats per team: {len(team_box_score_columns)}")
            print(team_box_score_columns)
    
    # Done, return the processed DataFrame and team info
    return df, unique_teams, team_ID_to_int




# ================================================================================
# Prepare the data
# ================================================================================
def prepare_data(df: pd.DataFrame, *, do_stats: bool = True):
    # Store the formatted data
    data = []
    
    # Iterate through all games
    for idx, game in df.iterrows():
        # --------------------------------------------------------------------------------
        # Get everything from the data that I need
        # --------------------------------------------------------------------------------
        # Get the team IDs
        W_ID, L_ID = game["W_year_ID"], game["L_year_ID"]
        
        # Get the team seeds
        W_seed, L_seed = game["W_Seed"], game["L_Seed"]
    
        # Now just get the game stats
        if do_stats:
            W_stats = get_team_game_stats(game, "W").to_list()
            L_stats = get_team_game_stats(game, "L").to_list()
        
        # Sometimes we won't actually worry about box score predictions and get just blank values
        else:
            W_stats = [0]*13
            L_stats = [0]*13
    
        # --------------------------------------------------------------------------------
        # Now format it and add it to the data (each game goes both ways)
        # --------------------------------------------------------------------------------
        data.append({"Input": [W_ID, W_seed, L_ID, L_seed], "Output": [W_stats + L_stats, 1.0]})
        data.append({"Input": [L_ID, L_seed, W_ID, W_seed], "Output": [L_stats + W_stats, 0.0]})
        
    return data

# Just makes it easier to get the columns
# TODO: I don't know if `team` has an underscore or not
def get_team_game_stats(game: pd.Series, team: str) -> pd.Series:
    """
    `team` is the W or L value preceding the stats.
    """
    columns = [f"{team}{stat}" for stat in TEAM_BOX_SCORE_COLUMNS]
    return game[columns]
