"""
Torch Dataset for processed game data.
--------------------------------------------------------------------------------
`src.processing.build_dataset`

"""
import torch

from torch.utils.data import Dataset


# --------------------------------------------------------------------------------
# Load game data during training
# --------------------------------------------------------------------------------
class BasketballDataset(Dataset):
    
    def __init__(self, input_data):
        self.data = input_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Extract the record
        record = self.data[idx]
        
        # Inputs
        input_data = torch.tensor(record["Input"], dtype=torch.long)
        
        # Outputs
        box_score_target = torch.tensor( record["Output"][0],  dtype=torch.float)
        win_proba_target = torch.tensor([record["Output"][1]], dtype=torch.float)
        
        # Return everything
        return (input_data, box_score_target, win_proba_target)

