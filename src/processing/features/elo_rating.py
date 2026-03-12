"""
Track Elo ratings for teams over historical data
--------------------------------------------------------------------------------
`src.processing.features.elo_rating`

"""
from ...config import ELO_K

# --------------------------------------------------------------------------------
# Elo Rating System
# --------------------------------------------------------------------------------
def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Calculates the expected probability of Team A beating Team B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

# --------------------------------------------------------------------------------
# Updates Elo ratings for a completed game
# --------------------------------------------------------------------------------
def get_new_elos(rating_w: float, rating_l: float, k: float = ELO_K) -> tuple[float, float]:
    """
    k: The maximum possible Elo change from a single game
    """
    expected_w = calculate_expected_score(rating_w, rating_l)
    expected_l = calculate_expected_score(rating_l, rating_w)

    # Winner actual score = 1.0, Loser actual score = 0.0
    new_rating_w = rating_w + k * (1.0 - expected_w)
    new_rating_l = rating_l + k * (0.0 - expected_l)

    return new_rating_w, new_rating_l

