"""
Track Elo ratings for teams over historical data
--------------------------------------------------------------------------------
`src.processing.features.elo_rating`

I'm actually going to keep all Elo config in this file

"""
import math

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
STARTING_ELO = 1_500.0  # Standard starting Elo is 1,500
ELO_WIDTH    =   400.0
ELO_K        =    50.0  # Maximum possible Elo change from a single game



# --------------------------------------------------------------------------------
# Elo Rating System
# --------------------------------------------------------------------------------
def calculate_expected_score(rating_a: float, rating_b: float, is_home_a: bool = False, is_home_b: bool = False) -> float:
    """
    Calculates the expected probability of Team A beating Team B.
    Adds a temporary 100-point Home Court Advantage (HCA) if applicable.
    """
    hca_a = 100.0 if is_home_a else 0.0
    hca_b = 100.0 if is_home_b else 0.0
    
    adj_rating_a = rating_a + hca_a
    adj_rating_b = rating_b + hca_b

    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

# --------------------------------------------------------------------------------
# Mean Reversion (reset all Elos marginally for next season)
# --------------------------------------------------------------------------------
def apply_mean_reversion(elos: dict) -> dict:
    """
    Revert all Elos 25% back to 1500 at the start of a new season.
    """
    for team in elos:
        elos[team] = (0.75 * elos[team]) + (0.25 * STARTING_ELO)
    return elos


# --------------------------------------------------------------------------------
# [Version 1] Updates Elo ratings for a completed game
# --------------------------------------------------------------------------------
def get_new_elos_v0(rating_w: float, rating_l: float, k: float = ELO_K) -> tuple[float, float]:
    """
    k: The maximum possible Elo change from a single game
    """
    expected_w = calculate_expected_score(rating_w, rating_l)
    expected_l = calculate_expected_score(rating_l, rating_w)

    # Winner actual score = 1.0, Loser actual score = 0.0
    new_rating_w = rating_w + k * (1.0 - expected_w)
    new_rating_l = rating_l + k * (0.0 - expected_l)

    return new_rating_w, new_rating_l

# --------------------------------------------------------------------------------
# Dynamic Updates for Elos
# --------------------------------------------------------------------------------
def get_dynamic_k(rating: float, base_k: float = ELO_K) -> float:
    """
    Reduces the K-factor for elite teams to prevent Elo inflation,
    and for terrible teams to prevent Elo deflation.
    """
    if   rating > 1750.0: return base_k * 0.5  # Elite teams move half as fast
    elif rating < 1250.0: return base_k * 0.5  # Bottom-tier teams are protected from free-falling
    else:                 return base_k        # Average teams move at normal speed

def get_continuous_k(rating: float, base_k: float =ELO_K) -> float:
    """
    Continuously smoothly decays the K-factor as a team gets further away from 1500.
    """
    # Calculate how far they are from average
    diff_from_avg = abs(rating - STARTING_ELO)
    
    # As the difference approaches 500+ points, the multiplier smoothly drops toward 0.5
    # When the difference is 0, the multiplier is exactly 1.0
    decay_multiplier = 1.0 - (0.5 * (diff_from_avg / 1000.0))
    
    # Put a hard floor so K never drops below half the base rate
    decay_multiplier = max(0.5, decay_multiplier)
    
    return base_k * decay_multiplier

# ================================================================================
# [Version 2] Updates Elo ratings for a completed game
# ================================================================================
def get_new_elos(
    rating_w  : float, 
    rating_l  : float, 
    margin    : float, 
    is_home_w : bool = False, 
    is_home_l : bool = False,
    base_k    : float = ELO_K  # Base K-factor
) -> tuple[float, float]:
    """
    Updates Elos using the Margin of Victory (MoV) multiplier.
    margin: The absolute difference in points (Winner Points - Loser Points)
    """
    expected_w = calculate_expected_score(rating_w, rating_l, is_home_a=is_home_w, is_home_b=is_home_l)
    expected_l = 1.0 - expected_w  # Symmetrical

    # 1) Calculate the Margin of Victory Multiplier
    # FiveThirtyEight's specific formula for basketball:
    # MoV_Mult = ln(|Margin| + 1) * (2.2 / ((Elo_Winner - Elo_Loser) * 0.001 + 2.2))
    elo_diff = rating_w - rating_l
    
    # Failsafe for 0-point margins (which shouldn't happen)
    if margin > 0: mov_multiplier = math.log(margin + 1.0) * (2.2 / ((elo_diff * 0.001) + 2.2))
    else:          mov_multiplier = 1.0 

    # 2) Calculate individual K-factors based on their current ratings
    k_w = get_continuous_k(rating_w, base_k)
    k_l = get_continuous_k(rating_l, base_k)

    # 3) Apply the unique shifts
    shift_w = k_w * mov_multiplier * (1.0 - expected_w)
    shift_l = k_l * mov_multiplier * (0.0 - expected_l)  # This will be negative
    
    new_rating_w = rating_w + shift_w
    new_rating_l = rating_l + shift_l 

    return new_rating_w, new_rating_l

