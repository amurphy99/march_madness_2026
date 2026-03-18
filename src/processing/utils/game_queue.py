"""
Help manage the game history queue for build_examples()
--------------------------------------------------------------------------------
`src.processing.utils.game_queue`

The original version used deque objects, removing oldest games first. In this
one, I want to try prioritizing keeping games against seeded opponents. 

The game history should be reset each year when using this method.

TODO: Maybe track some sort of "conference Elo" in the future and also keep games
      based on that.

"""
from collections import deque

# From this project
from ...config import DEFAULT_HISTORY_LEN


# --------------------------------------------------------------------------------
# Sort game history entries according to opponent SeedIDs
# --------------------------------------------------------------------------------
def _seed_worse_key(entry: dict) -> tuple[int, int, int]:
    """
    Bigger key = worse entry.
    Lower seed number is better (1 is best).
    For ties, keep newer games and discard older ones.
    """
    seed = int(entry["opp_seed"])
    normalized_seed = 17 if (seed == 0) else seed # Seed 0 is treated as worse than 16.

    return (
        normalized_seed,
        -int(entry["season"]),
        -int(entry["daynum"]),
    )

# --------------------------------------------------------------------------------
# Manually handle the length of the game hitory deque
# --------------------------------------------------------------------------------
def append_seed_pruned(history: deque, entry: dict, history_len: int = DEFAULT_HISTORY_LEN) -> None:
    """
    Since we no longer use a 'maxlen' in our deque, we control the length here.
    """
    # Add the new entry
    history.append(entry)

    # If we aren't over the max length, we are fine
    if len(history) <= history_len: return

    # Remove the "worst" entry according to the seeds (e.g., remove unseeded teams first)
    worst_entry = max(history, key=_seed_worse_key)
    history.remove(worst_entry)
