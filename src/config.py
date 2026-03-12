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

# Combined box score
BOX_SCORE_COLS = W_TEAM_STAT_COLS + L_TEAM_STAT_COLS
BOX_SCORE_DIM  = len(BOX_SCORE_COLS)

# Current numeric historical data dimensions (both team box scores + three extra stats: margin, loc, num_ot)
HIST_NUMERIC_DIM = len(BOX_SCORE_COLS) + 3

# --------------------------------------------------------------------------------
# Default Config
# --------------------------------------------------------------------------------
# Number of past tournament years to evaluate with (still uses all regular season data)
DEFAULT_PAST_YEARS = 7

# Number of previous games to store in each team's history
DEFAULT_HISTORY_LEN = 10

# --------------------------------------------------------------------------------
# Default Config
# --------------------------------------------------------------------------------
# Maximum possible Elo change from a single game
ELO_K = 100.0
