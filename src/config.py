"""
Configuration parameters / constants for the project.
--------------------------------------------------------------------------------
`src.config`

"""

# --------------------------------------------------------------------------------
# Data Columns
# --------------------------------------------------------------------------------
TEAM_BOX_SCORE_COLUMNS = ["Score","2FGM", "2FGA", "3FGM", "3FGA", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

W_TEAM_STAT_COLS = [f"W{stat}" for stat in TEAM_BOX_SCORE_COLUMNS]
L_TEAM_STAT_COLS = [f"W{stat}" for stat in TEAM_BOX_SCORE_COLUMNS]

BOX_SCORE_COLS = W_TEAM_STAT_COLS + L_TEAM_STAT_COLS


# --------------------------------------------------------------------------------
# Default Config
# --------------------------------------------------------------------------------
# Number of past tournament years to evaluate with (still uses all regular season data)
DEFAULT_PAST_YEARS = 7

# Number of previous games to store in each team's history
DEFAULT_HISTORY_LEN = 10

