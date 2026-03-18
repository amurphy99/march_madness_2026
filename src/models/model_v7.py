"""
Model architecture from last year.
--------------------------------------------------------------------------------
`src.models.model_v7`

Copy of model v1, with probabilistic outputs.

"""
import torch
import torch.nn            as nn
import torch.nn.functional as F

# From this project
from ..config import BOX_SCORE_DIM

# ================================================================================
# Model Definition
# ================================================================================
class MarchMadnessModel_v7(nn.Module):

    def __init__(self, 
        num_teams      : int, 
        num_seeds      : int, 
        team_embed_dim : int, 
        seed_embed_dim : int, 
        middle_dim     : int   = 256,
        dropout        : float = 0.25,
    ):
        super(MarchMadnessModel_v7, self).__init__()
        self.dropout = dropout
        
        # --------------------------------------------------------------------------------
        # 1) Embeddings
        # --------------------------------------------------------------------------------
        self.team_embedding = nn.Embedding(num_teams, team_embed_dim)
        self.seed_embedding = nn.Embedding(num_seeds, seed_embed_dim)

        in_dim = (team_embed_dim + seed_embed_dim) * 2

        # --------------------------------------------------------------------------------
        # 2) MLP backbone with batch norm
        # --------------------------------------------------------------------------------
        self.linear_1 = nn.Linear(in_dim, middle_dim)
        self.bn_1     = nn.BatchNorm1d(middle_dim)

        self.linear_2 = nn.Linear(middle_dim, 64)
        self.bn_2     = nn.BatchNorm1d(64)

        # --------------------------------------------------------------------------------
        # 3) Heads
        # --------------------------------------------------------------------------------
        # Box-Score Heads (mean + variance)
        self.box_score_mu      = nn.Linear(64, BOX_SCORE_DIM)
        self.box_score_log_var = nn.Linear(64, BOX_SCORE_DIM)
        
        # Win Probability Head
        self.win_logit   = nn.Linear(64, 1)

        # Evidential Head
        self.win_evidence = nn.Linear(64, 2)

        
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

        # --------------------------------------------------------------------------------
        # 5) Prediction Heads
        # --------------------------------------------------------------------------------
        # Box-Score (mean + variance)
        box_mu      = self.box_score_mu     (x)
        box_log_var = self.box_score_log_var(x)

        # Win Probability 
        win_logit   = self.win_logit(x).squeeze(-1)

        # Calculate alpha and beta
        raw_evidence = self.win_evidence(x)
        alpha_beta   = F.softplus(raw_evidence) + 1.0

        return (box_mu, box_log_var), win_logit, alpha_beta
