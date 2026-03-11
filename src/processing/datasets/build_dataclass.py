"""
Packed dataclass for history-based March Madness examples
--------------------------------------------------------------------------------
`src.processing.datasets.build_dataclass`

"""
import numpy as np
import torch

from dataclasses import dataclass, asdict
from pathlib     import Path


# ================================================================================
# Dataclass for embedding + history based predictions
# ================================================================================
@dataclass
class MarchMadnessPackedData:
    # Additional info
    season  : np.ndarray
    daynum  : np.ndarray
    row_idx : np.ndarray

    # Team IDs
    teamA_id : np.ndarray
    teamB_id : np.ndarray

    # Team A historic stats
    teamA_hist_numeric : np.ndarray
    teamA_hist_opp_ids : np.ndarray
    teamA_hist_mask    : np.ndarray

    # Team B historic stats
    teamB_hist_numeric : np.ndarray
    teamB_hist_opp_ids : np.ndarray
    teamB_hist_mask    : np.ndarray

    # Targets
    teamA_target_box_score : np.ndarray
    teamB_target_box_score : np.ndarray
    target_win             : np.ndarray

    @property
    def num_examples(self) -> int:
        return len(self.teamA_id)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "MarchMadnessPackedData":
        obj = torch.load(Path(path), weights_only=False)
        return cls(**obj)
    

# --------------------------------------------------------------------------------
# Packs list-of-dict examples into stacked numpy arrays
# -------------------------------------------------------------------------------- 
def pack_examples(examples: list[dict]) -> MarchMadnessPackedData:
    return MarchMadnessPackedData(
        # Additional info
        season  = np.asarray([ex["season" ] for ex in examples], dtype=np.int64),
        daynum  = np.asarray([ex["daynum" ] for ex in examples], dtype=np.int64),
        row_idx = np.asarray([ex["row_idx"] for ex in examples], dtype=np.int64),

        # Team IDs
        teamA_id = np.asarray([ex["teamA_id"] for ex in examples], dtype=np.int64),
        teamB_id = np.asarray([ex["teamB_id"] for ex in examples], dtype=np.int64),

        # Team A historic stats
        teamA_hist_numeric = np.stack([ex["teamA_hist_numeric"] for ex in examples]).astype(np.float32),
        teamA_hist_opp_ids = np.stack([ex["teamA_hist_opp_ids"] for ex in examples]).astype(np.int64  ),
        teamA_hist_mask    = np.stack([ex["teamA_hist_mask"   ] for ex in examples]).astype(np.float32),

        # Team B historic stats
        teamB_hist_numeric = np.stack([ex["teamB_hist_numeric"] for ex in examples]).astype(np.float32),
        teamB_hist_opp_ids = np.stack([ex["teamB_hist_opp_ids"] for ex in examples]).astype(np.int64),
        teamB_hist_mask    = np.stack([ex["teamB_hist_mask"   ] for ex in examples]).astype(np.float32),

        # Targets
        teamA_target_box_score = np.stack  ([ex["teamA_target_box_score"] for ex in examples]).astype(np.float32),
        teamB_target_box_score = np.stack  ([ex["teamB_target_box_score"] for ex in examples]).astype(np.float32),
        target_win             = np.asarray([ex["target_win"            ] for ex in examples], dtype=np.float32),
    )
