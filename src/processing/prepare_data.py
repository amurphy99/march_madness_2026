"""
Prepare training, validation, and/or testing data/
--------------------------------------------------------------------------------
`src.processing.prepare_data`

"""
import pandas as pd

# From this project
from .features.general import prep_seeds_df, prep_for_embeddings, prepare_data
from ..utils.logging   import RESET, BOLD, UNBOLD, BLUE

# ================================================================================
# Load in the game data and the seed data
# ================================================================================
def load_data(DATA_PATH: str, TOURNEY: str, *, num_past_years: int = 5, verbose : int = 1):
    """
    TOURNEY refers to which tournament's data we should load: either "M" or "W"
    """
    # --------------------------------------------------------------------------------
    # 1) Load the raw Kaggle data
    # --------------------------------------------------------------------------------
    rs_df = pd.read_csv(f"{DATA_PATH}/{TOURNEY}RegularSeasonDetailedResults.csv")
    st_df = pd.read_csv(f"{DATA_PATH}/{TOURNEY}SecondaryTourneyCompactResults.csv")
    tr_df = pd.read_csv(f"{DATA_PATH}/{TOURNEY}NCAATourneyDetailedResults.csv")
    seeds = pd.read_csv(f"{DATA_PATH}/{TOURNEY}NCAATourneySeeds.csv")

    # Convert the Season column to ints (if they aren't already), then get unique values and sort them.
    unique_seasons = sorted(tr_df["Season"].astype(int).unique())

    # Get the last X seasons (the highest values)
    last_five_seasons = unique_seasons[-num_past_years:]

    # Filter the DataFrame to only rows where the Season is in the last 5 seasons.
    tr_df = tr_df[tr_df["Season"].astype(int).isin(last_five_seasons)]

    # Print some info about the data
    if verbose: 
        print(
            f"Loaded data for {BOLD}{BLUE}{'MENS' if (TOURNEY=='M') else 'WOMENS'}{RESET} tournament. \n"
            f"Seasons included (last {BOLD}{BLUE}{num_past_years}{RESET} years): {BLUE}{last_five_seasons}{RESET}"
        )

    # --------------------------------------------------------------------------------
    # 2) Apply data preparation code
    # --------------------------------------------------------------------------------
    # Add a seed column to the game data using the seed data
    seeds, unique_seeds, seed_for_team = prep_seeds_df(seeds)

    # Prepare each DataFrame for embeddings (give teams integer IDs)
    rs_df, rs_teams, team_ID_to_int = prep_for_embeddings(rs_df, seed_for_team)
    st_df, st_teams, _ = prep_for_embeddings(st_df, seed_for_team, team_ID_to_int=team_ID_to_int, do_stats=False)
    tr_df, tr_teams, _ = prep_for_embeddings(tr_df, seed_for_team, team_ID_to_int=team_ID_to_int)

    # Call the function
    rs_data = prepare_data(rs_df, verbose=verbose)
    st_data = prepare_data(st_df, verbose=verbose, do_stats=False)
    tr_data = prepare_data(tr_df, verbose=verbose)

    # Print preview the finished data
    if verbose: 
        print(
            "\n"
            f"Regular Season Games:       {len(rs_data):7,} \n"
            f"Secondary Tournament Games: {len(st_data):7,} \n"
            f"NCAA Tournament Games:      {len(tr_data):7,}"
        )

    # Return **everything**
    return (
        rs_df, rs_teams, rs_data, 
        st_df, st_teams, st_data,
        tr_df, tr_teams, tr_data,

        seeds, unique_seeds, seed_for_team,
    )








