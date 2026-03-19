"""
Prepare training, validation, and/or testing data.
--------------------------------------------------------------------------------
`src.processing.prepare_data`

TOURNEY refers to which tournament's data we should load: either "M" or "W".

"""
import pandas as pd

# From this project
from  .features.general       import prep_seeds_df, prep_for_embeddings, prepare_data
from  .features.preprocessing import apply_year_team_IDs, apply_box_score_preprocessing
from  .features.preprocessing import get_teamID_to_int_dict, convert_teamIDs_to_int
from ..utils.logging          import RESET, BOLD, UNBOLD, BLUE
from ..config                 import DEFAULT_PAST_YEARS


# ================================================================================
# Load the raw Kaggle data
# ================================================================================
def load_data(
        DATA_PATH: str, 
        TOURNEY  : str, 
        *, 
        scale_data          : bool = True, # If we should apply a pre-trained StandardScaler to the box scores
        convert_IDs_to_ints : bool = True, # If we should already convert the IDs
        verbose             : int  = 1,

        # Which years to get data from
        num_past_years      : int              = DEFAULT_PAST_YEARS, # Number of years ago to get (e.g., the last 5)
        do_past_years       : list[int] | None = None                # Direct list of years to train on (e.g., [2018, 2019, ...])
):
    # --------------------------------------------------------------------------------
    # 1) Load the raw Kaggle data
    # --------------------------------------------------------------------------------
    # Game box score results
    rs_df = pd.read_csv(f"{DATA_PATH}/{TOURNEY}RegularSeasonDetailedResults.csv")
    st_df = pd.read_csv(f"{DATA_PATH}/{TOURNEY}SecondaryTourneyCompactResults.csv")
    tr_df = pd.read_csv(f"{DATA_PATH}/{TOURNEY}NCAATourneyDetailedResults.csv")

    # Other data to use (Kaggle-provided seeds, etc.)
    seeds = pd.read_csv(f"{DATA_PATH}/{TOURNEY}NCAATourneySeeds.csv")

    # --------------------------------------------------------------------------------
    # 2) Apply additional processing methods (team-year ID creation, etc.)
    # --------------------------------------------------------------------------------
    # Convert the Season column to ints (if they aren't already), then get unique values and sort them
    #unique_seasons  = sorted(tr_df["Season"].astype(int).unique())
    unique_seasons  = sorted(rs_df["Season"].astype(int).unique())
    include_seasons = unique_seasons[-num_past_years:]

    # If given specific past years, use them; otherwise use data from the past X seasons
    if do_past_years: include_seasons = do_past_years
    
    # Filter the DataFrame to only rows for the seasons we want to train on
    rs_df = rs_df[rs_df["Season"].astype(int).isin(include_seasons)]
    st_df = st_df[st_df["Season"].astype(int).isin(include_seasons)]
    tr_df = tr_df[tr_df["Season"].astype(int).isin(include_seasons)]

    # Scaling args
    scale_args = dict(scale_data=scale_data, tourney=TOURNEY, years=num_past_years)

    # Year-Team IDs + other preprocessing for the box score data
    rs_df = apply_box_score_preprocessing(rs_df, **scale_args)
    tr_df = apply_box_score_preprocessing(tr_df, **scale_args)
    st_df = apply_box_score_preprocessing(st_df, do_box=False, scale_data=False)
    
    seeds = apply_year_team_IDs(seeds)

    # Print some info about the data
    if verbose: 
        print(
            f"Loaded data for {BOLD}{BLUE}{'MENS' if (TOURNEY=='M') else 'WOMENS'}{RESET} tournament. \n"
            f"Seasons included (last {BLUE}{num_past_years}{RESET} years): {BLUE}{include_seasons}{RESET} \n"
        )

    # --------------------------------------------------------------------------------
    # 3) Convert string team IDs to integer IDs
    # --------------------------------------------------------------------------------
    team_ID_to_int = get_teamID_to_int_dict(rs_df)

    if convert_IDs_to_ints:
        rs_df = convert_teamIDs_to_int(rs_df, team_ID_to_int)
        st_df = convert_teamIDs_to_int(st_df, team_ID_to_int)
        tr_df = convert_teamIDs_to_int(tr_df, team_ID_to_int)

    # --------------------------------------------------------------------------------
    # 4) Add a seed column to the game data using the seed data
    # --------------------------------------------------------------------------------
    # Seeds are already in int/ID format after this function (ready for embeddings)
    seeds, unique_seeds, seed_for_team = prep_seeds_df(seeds)

    # Use the year ID strings to map seeds to this dataframe
    for df in [rs_df, st_df, tr_df]:
        df["W_Seed"] = df["WYearTeamID"].map(lambda team_ID: seed_for_team.get(team_ID, 0))
        df["L_Seed"] = df["LYearTeamID"].map(lambda team_ID: seed_for_team.get(team_ID, 0))

    # Return all data
    return (
        rs_df, st_df, tr_df, seeds, team_ID_to_int,
        unique_seeds, seed_for_team,
    )



# ================================================================================
# Load in the game data and the seed data
# ================================================================================
def load_training_data_v1(DATA_PATH: str, TOURNEY: str, *, num_past_years: int = 5, verbose : int = 1):
    # 1) Load the raw Kaggle data
    rs_df, st_df, tr_df, seeds, _ = load_data(
        DATA_PATH, TOURNEY, num_past_years=num_past_years, verbose=verbose, convert_IDs_to_ints=False
    )

    # --------------------------------------------------------------------------------
    # 2) Apply data preparation code
    # --------------------------------------------------------------------------------
    # Add a seed column to the game data using the seed data
    seeds, unique_seeds, seed_for_team = prep_seeds_df(seeds)

    # Prepare each DataFrame for embeddings (give teams integer IDs)
    rs_df, rs_teams, team_ID_to_int = prep_for_embeddings(rs_df, seed_for_team, verbose=verbose)
    st_df, st_teams, _              = prep_for_embeddings(st_df, seed_for_team, verbose=verbose, team_ID_to_int=team_ID_to_int, do_stats=False)
    tr_df, tr_teams, _              = prep_for_embeddings(tr_df, seed_for_team, verbose=verbose, team_ID_to_int=team_ID_to_int)

    # Call the function
    rs_data = prepare_data(rs_df, verbose=verbose)
    st_data = prepare_data(st_df, verbose=verbose, do_stats=False)
    tr_data = prepare_data(tr_df, verbose=verbose)

    # Print preview the finished data
    if verbose: 
        print(
            "\n"
            f"Regular Season Games:       {BLUE}{len(rs_data):7,}{RESET} \n"
            f"Secondary Tournament Games: {BLUE}{len(st_data):7,}{RESET} \n"
            f"NCAA Tournament Games:      {BLUE}{len(tr_data):7,}{RESET}"
        )

    # Return **everything**
    return (
        rs_df, rs_teams, rs_data, 
        st_df, st_teams, st_data,
        tr_df, tr_teams, tr_data,

        seeds, unique_seeds, seed_for_team,
    )








