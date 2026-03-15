"""
Configuration parameters / constants for the project.
--------------------------------------------------------------------------------
`src.config`

Putting some values into their own config classes to help group things up.

TODO: Edit VSCode settings to make it so that some of this stuff is still blue, 
      even though it is inside a class. It just helps so much with readability...

"""

# --------------------------------------------------------------------------------
# Data Columns
# --------------------------------------------------------------------------------
TEAM_BOX_SCORE_COLUMNS = ["Score", "2FGM", "2FGA", "3FGM", "3FGA", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

W_TEAM_STAT_COLS = [f"W{stat}" for stat in TEAM_BOX_SCORE_COLUMNS]
L_TEAM_STAT_COLS = [f"W{stat}" for stat in TEAM_BOX_SCORE_COLUMNS]

# Combined box score
BOX_SCORE_COLS = W_TEAM_STAT_COLS + L_TEAM_STAT_COLS
BOX_SCORE_DIM  = len(BOX_SCORE_COLS)

# Current numeric historical data dimensions (both team box scores + three extra stats: margin, loc, num_ot)
HIST_NUMERIC_DIM = len(BOX_SCORE_COLS) + 3

# --------------------------------------------------------------------------------
# Default Pre-processing Config
# --------------------------------------------------------------------------------
# Number of past tournament years to evaluate with (still uses all regular season data)
DEFAULT_PAST_YEARS = 10

# Number of previous games to store in each team's history
DEFAULT_HISTORY_LEN = 10

# Maximum possible Elo change from a single game
ELO_K = 50.0

# --------------------------------------------------------------------------------
# Default Training Config
# --------------------------------------------------------------------------------
class DEFAULT_TRAINING_CONFIG:
    DEVICE        = "mps" # mps | cpu
    BATCH_SIZE    = 128   # Was faster than 64 last time I tested
    EPOCHS        =  30   # Default epcohs (also get's used by CosineAnnealingLR)

    LEARNING_RATE = 1e-4  # Initial learning rate 
    ETA_MIN       = 1e-5  # For learning rate schedulers
    WEIGHT_DECAY  = 1e-6  # Regulization for some optimizers

    # Default Setup
    OPTIMIZER_TYPE = "adam"
    SCHEDULER_TYPE = "none"


