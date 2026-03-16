"""
Builds model examples chronologically with team game histories. 
--------------------------------------------------------------------------------
`src.processing.datasets.generator`

We will create a Dataset object from the result of this here that randomly flips
the team order for us at training time.

"""
import numpy  as np
import pandas as pd

from collections import defaultdict, deque
from copy        import deepcopy
from tqdm.auto   import tqdm
from typing      import Any

# From this project
from ...config              import W_TEAM_STAT_COLS, L_TEAM_STAT_COLS, BOX_SCORE_COLS
from ...config              import DEFAULT_HISTORY_LEN
from  ..features.elo_rating import get_new_elos
from   .history             import make_team_history_entry, history_to_arrays


# --------------------------------------------------------------------------------
# Build current-game targets from Team A perspective
# --------------------------------------------------------------------------------
def make_current_game_targets(row: pd.Series, teamA_ID, teamB_ID) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """
    Returns the two box scores separately so we can still flip them if needed.
    """
    # To match the two teams with the winner and loser
    W_team = row["W_year_ID"]
    L_team = row["L_year_ID"]

    # Get the stats (different order for which team won)
    if (teamA_ID == W_team) and (teamB_ID == L_team):
        teamA_stats = row[W_TEAM_STAT_COLS].to_numpy(dtype=np.float32)
        teamB_stats = row[L_TEAM_STAT_COLS].to_numpy(dtype=np.float32)
        target_win  = 1.0
    elif (teamA_ID == L_team) and (teamB_ID == W_team):
        teamA_stats = row[L_TEAM_STAT_COLS].to_numpy(dtype=np.float32)
        teamB_stats = row[W_TEAM_STAT_COLS].to_numpy(dtype=np.float32)
        target_win  = 0.0
    else: raise ValueError("teamA/teamB do not match current row")

    # Return the box score targets and the win flag
    return teamA_stats, teamB_stats, np.float32(target_win)

# ================================================================================
# Build all model examples chronologically without future leakage
# ================================================================================
def build_examples(
        df          : pd.DataFrame,                          # Box score results/data for games
        history_len : int = DEFAULT_HISTORY_LEN,             # How many games to keep in the history
        *,
        team_histories : dict[str, deque] | None = None,  # Can pass the end of regular season history to tournaments
        team_elos      : dict[int, float] | None = None,  # Elo tracker
        
        update_hist : bool = True,                           # Whether or not to update the team histories
        has_box     : bool = True,                           # Secondary tournament games have no box score data

) -> tuple[list[dict[str, Any]], dict[str, deque], dict[int, float]]:
    """
    Reads history first, then updates history with the current game. Each game is 
    stored once in the canonical orientation (winning team first).

    We will create a Dataset object from the result of this here that randomly flips
    the team order for us at training time.
    """
    # Sort again just in case
    df = df.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    # --------------------------------------------------------------------------------
    # Initialize Histories (histories from the end of the regular season will be provided for tournament games)
    # --------------------------------------------------------------------------------
    # If not provided with starting historical data, create a fresh dictionary
    if team_histories is None: team_histories = defaultdict(lambda: deque(maxlen=history_len))
    else:                      team_histories = deepcopy(team_histories)

    # Initialize Elos (Standard starting Elo is 1500)
    if team_elos is None: team_elos = defaultdict(lambda: 1500.0)
    else:                 team_elos = deepcopy(team_elos)

    # Storage for the final output examples
    examples = []

    # --------------------------------------------------------------------------------
    # Loop through each game in chronological order
    # --------------------------------------------------------------------------------
    pbar = tqdm(df.iterrows(), desc="Building team histories & training examples", total=len(df), leave=True)
    for row_idx, row in pbar:
        # School IDs for the two teams 
        W_team_school = row["WTeamID"]
        L_team_school = row["LTeamID"]

        # Year IDs
        W_team = row["W_year_ID"]
        L_team = row["L_year_ID"]

        # --------------------------------------------------------------------------------
        # Winning team first (winner, loser)
        # --------------------------------------------------------------------------------
        teamA_id = W_team
        teamB_id = L_team

        # Get current Elos
        teamA_elo = team_elos[W_team_school]
        teamB_elo = team_elos[L_team_school]

        # Individual team historic data
        teamA_hist_numeric, teamA_hist_opp_ids, teamA_hist_mask = history_to_arrays(team_histories[teamA_id], history_len=history_len)
        teamB_hist_numeric, teamB_hist_opp_ids, teamB_hist_mask = history_to_arrays(team_histories[teamB_id], history_len=history_len)

        # Create targets for this game (or fill with 0s if no data)
        if has_box:
            teamA_box_score, teamB_box_score, target_win = make_current_game_targets(row, teamA_id, teamB_id)
        else:
            num_team_boxscore_features = len(W_TEAM_STAT_COLS)
            teamA_box_score = np.zeros((num_team_boxscore_features,), dtype=np.float32)
            teamB_box_score = np.zeros((num_team_boxscore_features,), dtype=np.float32)
            target_win      = 1.0  # I just know this is 1.0 here

        # Put the full example together
        examples.append({
            # Additional info
            "season"  : int(row["Season"]),
            "daynum"  : int(row["DayNum"]),
            "row_idx" : int(row_idx),

            # Team IDs
            "teamA_id" : int(teamA_id),
            "teamB_id" : int(teamB_id),

            # Seed IDs
            "teamA_seed": int(row["W_Seed"]),
            "teamB_seed": int(row["L_Seed"]),

            # Team Elo ratings
            "teamA_elo": float(teamA_elo),
            "teamB_elo": float(teamB_elo),

            # Team A historic stats
            "teamA_hist_numeric" : teamA_hist_numeric,
            "teamA_hist_opp_ids" : teamA_hist_opp_ids,
            "teamA_hist_mask"    : teamA_hist_mask,

            # Team B historic stats
            "teamB_hist_numeric" : teamB_hist_numeric,
            "teamB_hist_opp_ids" : teamB_hist_opp_ids,
            "teamB_hist_mask"    : teamB_hist_mask,

            # Targets
            "teamA_target_box_score" : teamA_box_score,  # Team A's box score stats
            "teamB_target_box_score" : teamB_box_score,  # Team B's box score stats
            "target_win"             : target_win,       # From Team A's perspective
        })

        # --------------------------------------------------------------------------------
        # Update team histories & Elo ratings
        # --------------------------------------------------------------------------------
        # We don't always update the history (secondary tournament games have no box scores)
        if update_hist:
            if has_box:
                # Generate a history entry for this game and save it
                W_entry = make_team_history_entry(row, team_id=W_team, opp_id=L_team, is_winner=True )
                L_entry = make_team_history_entry(row, team_id=L_team, opp_id=W_team, is_winner=False)
                team_histories[W_team].append(W_entry)
                team_histories[L_team].append(L_entry)

            # Update Elos post-game 
            new_A_elo, new_B_elo = get_new_elos(teamA_elo, teamB_elo)
            team_elos[W_team_school] = new_A_elo
            team_elos[L_team_school] = new_B_elo

    return examples, team_histories, team_elos

