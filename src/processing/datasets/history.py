"""
Build game histories for each team, from their own perspective. 
--------------------------------------------------------------------------------
`src.processing.datasets.history`

Includes box score stats and the opposing teams ID. 

"""
import numpy  as np
import pandas as pd

from collections import defaultdict, deque
from tqdm.auto   import tqdm
from typing      import Any

# From this project
from ...config import TEAM_BOX_SCORE_COLUMNS, W_TEAM_STAT_COLS, L_TEAM_STAT_COLS
from ...config import DEFAULT_HISTORY_LEN


# --------------------------------------------------------------------------------
# Get the location for the given team's perspective
# --------------------------------------------------------------------------------
def _normalize_loc_for_team(wloc: str, is_winner: bool) -> int:
    """
    Returns location from the given team's perspective:
    1 = home, 0 = neutral, -1 = away
    """
    # Neutral
    if pd.isna(wloc): return 0
    if (wloc == "N"): return 0
    
    # Home depends on who won
    if is_winner: return  1 if wloc == "H" else -1
    else:         return -1 if wloc == "H" else  1

# --------------------------------------------------------------------------------
# Create a team history record based on a row of box score data
# --------------------------------------------------------------------------------
def make_team_history_entry(row: pd.Series, team_id: int, opp_id: int, is_winner: bool) -> dict[str, Any]:
    """
    At this point, we should have already mapped the string team IDs to integers.
    """
    # Box score stats (from the perspective of the given team)
    if is_winner:
        team_stats = row[W_TEAM_STAT_COLS].to_numpy(dtype=np.float32)
        opp_stats  = row[L_TEAM_STAT_COLS].to_numpy(dtype=np.float32)
        
    else:
        team_stats = row[L_TEAM_STAT_COLS].to_numpy(dtype=np.float32)
        opp_stats  = row[W_TEAM_STAT_COLS].to_numpy(dtype=np.float32)

    # Additional Stats
    win_flag = float(is_winner)
    margin   = float(team_stats[0] - opp_stats[0])   # Score diff
    loc      = float(_normalize_loc_for_team(row["WLoc"], is_winner))
    num_ot   = float(row.get("NumOT", 0))

    # Data entry
    entry = {
        "team_id"    : team_id,
        "opp_id"     :  opp_id,
        "season"     : int(row["Season"]),
        "daynum"     : int(row["DayNum"]),
        "team_stats" : team_stats,     # shape (14,)
        "opp_stats"  :  opp_stats,     # shape (14,)
        "margin"     : margin,
        "win"        : win_flag,
        "loc"        : loc,
        "num_ot"     : num_ot,
    }
    return entry

# --------------------------------------------------------------------------------
# Convert a team's prior-game deque into fixed-size arrays
# --------------------------------------------------------------------------------
def history_to_arrays(history_deque: deque, history_len: int = DEFAULT_HISTORY_LEN):
    """
    Output order is oldest -> newest
    """
    entries = list(history_deque)

    # Feature dimensions
    team_dim  = len(TEAM_BOX_SCORE_COLUMNS)
    extra_dim = 3  # margin, loc, num_ot
    feat_dim  = team_dim + team_dim + extra_dim

    # History elements (numeric stats, opponent IDs, mask for padding)
    hist_numeric = np.zeros((history_len, feat_dim), dtype=np.float32)
    hist_opp_ids = np.zeros((history_len,         ), dtype=np.int64  )
    hist_mask    = np.zeros((history_len,         ), dtype=np.float32)

    # Keep most recent `history_len` games, but preserve chronological order
    entries = entries[-history_len:]
    start   = history_len - len(entries)

    # Fill the arrays
    for i, entry in enumerate(entries):
        j = start + i

        # Numeric box score stats for both teams & any extra stats to include (margin, loc, num_ot)
        hist_numeric[j, :] = np.concatenate([
            entry["team_stats"], entry[ "opp_stats"],
            np.array([entry["margin"], entry["loc"], entry["num_ot"]], dtype=np.float32)
        ])

        # IDs and mask (mask will remain 0 for any missing teams)
        hist_opp_ids[j] = int(entry["opp_id"])
        hist_mask   [j] = 1.0

    return hist_numeric, hist_opp_ids, hist_mask



# ================================================================================
# Build chronological per-team game history
# ================================================================================
def build_team_histories(df: pd.DataFrame, *, history_len: int = DEFAULT_HISTORY_LEN, verbose: int = 1) -> dict[str, deque]:
    """
    DEPRECATED:
    Team histories are now built in `generator.py` and used to create the training data.
    This can still be used to debug, but it won't be called during the final training process.
    """
    # Sort again just in case
    df = df.sort_values(["Season", "DayNum"]).reset_index(drop=True)
    
    # History storage 
    team_histories = defaultdict(lambda: deque(maxlen=history_len))

    # Loop through each game in chronological order
    pbar = tqdm(df.iterrows(), desc="Building team histories", total=len(df), leave=True)
    for _, row in pbar:
        W_team = row["WTeamID"]
        L_team = row["LTeamID"]

        W_entry = make_team_history_entry(row, team_id=W_team, opp_id=L_team, is_winner=True )
        L_entry = make_team_history_entry(row, team_id=L_team, opp_id=W_team, is_winner=False)

        team_histories[W_team].append(W_entry)
        team_histories[L_team].append(L_entry)

    return team_histories

