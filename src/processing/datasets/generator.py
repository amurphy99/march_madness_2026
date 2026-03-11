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
from ...config  import W_TEAM_STAT_COLS, L_TEAM_STAT_COLS, BOX_SCORE_COLS
from ...config  import DEFAULT_HISTORY_LEN
from   .history import make_team_history_entry, history_to_arrays


# --------------------------------------------------------------------------------
# Build current-game targets from Team A perspective
# --------------------------------------------------------------------------------
def make_current_game_targets(row: pd.Series, teamA_ID, teamB_ID) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """
    Returns the two box scores separately so we can still flip them if needed.
    """
    # To match the two teams with the winner and loser
    W_team = row["WTeamID"]
    L_team = row["LTeamID"]

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
        team_histories    : dict[str, deque] | None = None,  # Can pass the end of regular season history to tournaments
        update_history    : bool = True,                     # Whether or not to update the team histories
        has_full_boxscore : bool = True,                     # Secondary tournament games have no box score data

) -> tuple[list[dict[str, Any]], dict[str, deque]]:
    """
    Reads history first, then updates history with the current game. Each game is 
    stored once in the canonical orientation (winning team first).

    We will create a Dataset object from the result of this here that randomly flips
    the team order for us at training time.
    """
    # Sort again just in case
    df = df.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    # If not provided with starting historical data, create a fresh dictionary
    # (histories from the end of the regular season will be provided for tournament games)
    if team_histories is None: team_histories = defaultdict(lambda: deque(maxlen=history_len))
    else:                      team_histories = deepcopy(team_histories)

    # Storage for the final output examples
    examples = []

    # Loop through each game in chronological order
    pbar = tqdm(df.iterrows(), desc="Building team histories & training examples", total=len(df), leave=True)
    for row_idx, row in pbar:
        # IDs for the two teams 
        W_team = row["WTeamID"]
        L_team = row["LTeamID"]

        # --------------------------------------------------------------------------------
        # Winning team first (winner, loser)
        # --------------------------------------------------------------------------------
        teamA_id = W_team
        teamB_id = L_team

        # Individual team historic data
        teamA_hist_numeric, teamA_hist_opp_ids, teamA_hist_mask = history_to_arrays(team_histories[teamA_id], history_len=history_len)
        teamB_hist_numeric, teamB_hist_opp_ids, teamB_hist_mask = history_to_arrays(team_histories[teamB_id], history_len=history_len)

        # Create targets for this game (or fill with 0s if no data)
        if has_full_boxscore:
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
        # Update team histories
        # --------------------------------------------------------------------------------
        # We don't always update the history (secondary tournament games have no box scores)
        if update_history:
            if not has_full_boxscore: raise ValueError("Cannot update history when has_full_boxscore=False")

            # Generate a history entry for this game and save it
            W_entry = make_team_history_entry(row, team_id=W_team, opp_id=L_team, is_winner=True )
            L_entry = make_team_history_entry(row, team_id=L_team, opp_id=W_team, is_winner=False)
            team_histories[W_team].append(W_entry)
            team_histories[L_team].append(L_entry)

    return examples, team_histories

