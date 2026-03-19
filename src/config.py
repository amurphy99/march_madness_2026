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
# Taking the "Score" column out for now -- it should be able to figure that out on its own...
TEAM_BOX_SCORE_COLUMNS = ["2FGM", "2FGA", "3FGM", "3FGA", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

W_TEAM_STAT_COLS = [f"W{stat}" for stat in TEAM_BOX_SCORE_COLUMNS]
L_TEAM_STAT_COLS = [f"L{stat}" for stat in TEAM_BOX_SCORE_COLUMNS]

# Combined box score
BOX_SCORE_COLS = W_TEAM_STAT_COLS + L_TEAM_STAT_COLS
BOX_SCORE_DIM  = len(BOX_SCORE_COLS)

# Current numeric historical data dimensions (both team box scores + three extra stats: margin, loc, num_ot)
HIST_NUMERIC_DIM = len(BOX_SCORE_COLS) + 3

# Point column locations within the box score output
POINT_COL_IDX = {
    "W2FGM" : BOX_SCORE_COLS.index("W2FGM"),
    "W3FGM" : BOX_SCORE_COLS.index("W3FGM"),
    "WFTM"  : BOX_SCORE_COLS.index("WFTM" ),

    "L2FGM" : BOX_SCORE_COLS.index("L2FGM"),
    "L3FGM" : BOX_SCORE_COLS.index("L3FGM"),
    "LFTM"  : BOX_SCORE_COLS.index("LFTM" ),
}


# --------------------------------------------------------------------------------
# Default Pre-processing Config
# --------------------------------------------------------------------------------
# Number of past tournament years to evaluate with (still uses all regular season data)
DEFAULT_PAST_YEARS = 10

# Number of previous games to store in each team's history
DEFAULT_HISTORY_LEN = 20 # 10


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
    PATIENCE      = 5     # Wait for X epochs of no improvement before dropping learning rate

    # Default Setup
    OPTIMIZER_TYPE = "adam"
    SCHEDULER_TYPE = "none"


