"""
Set all of the random seeds
--------------------------------------------------------------------------------
`src.utils.random_seed`

Can call this function whenever needed to reset the random seeds to this.

"""
import numpy as np
import torch, random

# Default seed
SEED = 0

# Set seeds for all libraries
def set_seeds(seed: int = SEED):
    random      .seed(SEED)
    np.random   .seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

