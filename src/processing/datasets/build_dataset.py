"""
PyTorch Dataset for the MarchMadnessPackedData
--------------------------------------------------------------------------------
`src.processing.datasets.build_dataset`

Uses augmentation method that flips the team order for 50% of game samples.

"""
import numpy as np
import torch, random

from torch.utils.data import Dataset

# From this project
from .build_dataclass import MarchMadnessPackedData


# ================================================================================
# Build Torch Dataset from our Dataclass
# ================================================================================
class MarchMadnessHistoryDataset(Dataset):
    def __init__(
        self,
        data    : MarchMadnessPackedData,
        indices : np.ndarray | list[int] | None = None,
        *,
        training  : bool  = True,
        flip_prob : float = 0.5,
    ):
        # Data & config
        self.data      = data
        self.training  = training
        self.flip_prob = flip_prob

        # Data indices
        if indices is None: self.indices = np.arange (data.num_examples, dtype=np.int64)
        else:               self.indices = np.asarray(indices,           dtype=np.int64)

    # Double the dataset size during validation for deterministic flipping
    def __len__(self) -> int:
        if self.training: return len(self.indices)
        else:             return len(self.indices) * 2

    # ================================================================================
    # Sampling
    # ================================================================================
    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        # --------------------------------------------------------------------------------
        # Determine actual index and whether to flip
        # --------------------------------------------------------------------------------
        if self.training:
            idx     = int(self.indices[i])
            do_flip = random.random() < self.flip_prob
        else:
            # Map index to data, and deterministically flip every odd index
            idx = int(self.indices[i // 2])
            do_flip = (i % 2 == 1)

        # --------------------------------------------------------------------------------
        # Pull numpy data
        # --------------------------------------------------------------------------------
        # Team IDs
        teamA_id = self.data.teamA_id[idx]
        teamB_id = self.data.teamB_id[idx]

        # Team A historic stats
        teamA_hist_numeric = self.data.teamA_hist_numeric[idx].copy()
        teamA_hist_opp_ids = self.data.teamA_hist_opp_ids[idx].copy()
        teamA_hist_mask    = self.data.teamA_hist_mask   [idx].copy()

        # Team B historic stats
        teamB_hist_numeric = self.data.teamB_hist_numeric[idx].copy()
        teamB_hist_opp_ids = self.data.teamB_hist_opp_ids[idx].copy()
        teamB_hist_mask    = self.data.teamB_hist_mask   [idx].copy()

        # Targets
        teamA_target_box_score = self.data.teamA_target_box_score[idx].copy()
        teamB_target_box_score = self.data.teamB_target_box_score[idx].copy()
        target_win             = np.float32(self.data.target_win [idx])

        # --------------------------------------------------------------------------------
        # Flip data if triggered
        # --------------------------------------------------------------------------------
        flipped = False
        if do_flip:
            flipped = True

            # Flip IDs
            teamA_id, teamB_id = teamB_id, teamA_id

            # Flip historical data
            teamA_hist_numeric, teamB_hist_numeric = teamB_hist_numeric, teamA_hist_numeric
            teamA_hist_opp_ids, teamB_hist_opp_ids = teamB_hist_opp_ids, teamA_hist_opp_ids
            teamA_hist_mask,    teamB_hist_mask    = teamB_hist_mask,    teamA_hist_mask

            # Flip targets
            teamA_target_box_score, teamB_target_box_score = teamB_target_box_score, teamA_target_box_score
            target_win = np.float32(1.0 - target_win)

        # Concatenate the box score outcome for the game
        target_box_score = np.concatenate([teamA_target_box_score, teamB_target_box_score], axis=0).astype(np.float32)

        return {
            # Additional info
            "season"  : torch.tensor(self.data.season [idx], dtype=torch.long),
            "daynum"  : torch.tensor(self.data.daynum [idx], dtype=torch.long),
            "row_idx" : torch.tensor(self.data.row_idx[idx], dtype=torch.long),

            # Team IDs
            "teamA_id" : torch.tensor(teamA_id, dtype=torch.long),
            "teamB_id" : torch.tensor(teamB_id, dtype=torch.long),

            "teamA_hist_numeric" : torch.tensor(teamA_hist_numeric, dtype=torch.float32),
            "teamA_hist_opp_ids" : torch.tensor(teamA_hist_opp_ids, dtype=torch.long   ),
            "teamA_hist_mask"    : torch.tensor(teamA_hist_mask,    dtype=torch.float32),

            # Team B historic stats
            "teamB_hist_numeric" : torch.tensor(teamB_hist_numeric, dtype=torch.float32),
            "teamB_hist_opp_ids" : torch.tensor(teamB_hist_opp_ids, dtype=torch.long   ),
            "teamB_hist_mask"    : torch.tensor(teamB_hist_mask,    dtype=torch.float32),

            # Targets
            "target_box_score" : torch.tensor(target_box_score, dtype=torch.float32),
            "target_win"       : torch.tensor(target_win,       dtype=torch.float32),

            # Flag for if the teams were flipped
            "flipped": torch.tensor(flipped, dtype=torch.bool),
        }
