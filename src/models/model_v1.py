"""
Model architecture from last year.
--------------------------------------------------------------------------------
`src.models.model_v1`

"""
import torch
import torch.nn            as nn
import torch.nn.functional as F

# From this project
from ..config import BOX_SCORE_DIM

# ================================================================================
# Model Definition
# ================================================================================
class MarchMadnessModel_v1(nn.Module):

    def __init__(self, num_teams, num_seeds, team_embed_dim, seed_embed_dim, middle_dim: int = 256, dropout: float = 0.25):
        super(MarchMadnessModel_v1, self).__init__()
        self.dropout = dropout
        
        # Embeddings
        self.team_embedding = nn.Embedding(num_teams, team_embed_dim)
        self.seed_embedding = nn.Embedding(num_seeds, seed_embed_dim)

        in_dim = (team_embed_dim + seed_embed_dim) * 2

        # MLP backbone with batch norm
        self.linear_1 = nn.Linear(in_dim, middle_dim)
        self.bn_1     = nn.BatchNorm1d(middle_dim)

        self.linear_2 = nn.Linear(middle_dim, 64)
        self.bn_2     = nn.BatchNorm1d(64)

        # Outputs
        self.box_score_out = nn.Linear(64, BOX_SCORE_DIM)
        self.win_out       = nn.Linear(64,  1)
        
    # ================================================================================
    # Forward pass 
    # ================================================================================
    def forward(self, batch):
        # Get the team IDs
        teamA_id = batch["teamA_id"]
        teamB_id = batch["teamB_id"]
    
        # Get the seed IDs
        teamA_seed = batch["teamA_seed"]
        teamB_seed = batch["teamB_seed"]

        # 1) Embedding lookup for the teams
        team_A_emb = self.team_embedding(teamA_id)
        team_B_emb = self.team_embedding(teamB_id)

        # 2) Embedding lookup for the seeds
        team_A_seed_emb = self.seed_embedding(teamA_seed)
        team_B_seed_emb = self.seed_embedding(teamB_seed)

        # 3) Concatenate them together
        x = torch.cat([team_A_emb, team_A_seed_emb, team_B_emb, team_B_seed_emb], dim=-1)

        # 4) Dense stack
        x = self.linear_1(x)
        x = self.bn_1    (x)
        x = F.relu       (x)
        x = F.dropout    (x, self.dropout, training=self.training)

        x = self.linear_2(x)
        x = self.bn_2    (x)
        x = F.relu       (x)

        # 5) Heads
        box_score_pred = self.box_score_out(x) 
        win_logit      = self.win_out      (x).squeeze(-1)

        return box_score_pred, win_logit

