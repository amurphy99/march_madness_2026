"""
Model architecture from last year.
--------------------------------------------------------------------------------
`src.models.model_v1`

"""
import torch
import torch.nn            as nn
import torch.nn.functional as F


class MarchMadnessModel_v1(nn.Module):

    def __init__(self, num_teams, num_seeds, team_embed_dim, seed_embed_dim, dropout):
        super(MarchMadnessModel_v1, self).__init__()
        self.dropout = dropout
        
        # Embeddings
        self.team_embedding = nn.Embedding(num_teams, team_embed_dim)
        self.seed_embedding = nn.Embedding(num_seeds, seed_embed_dim)

        in_dim = (team_embed_dim + seed_embed_dim) * 2

        # MLP backbone with batch norm
        self.linear_1 = nn.Linear(in_dim, 256)
        self.bn_1     = nn.BatchNorm1d(256)

        self.linear_2 = nn.Linear(256, 64)
        self.bn_2     = nn.BatchNorm1d(64)

        # Outputs
        self.box_score_out = nn.Linear(64, 26)
        self.win_proba_out = nn.Linear(64,  1)
        
    def forward(self, input_data):
        # 1) Embedding lookup for the teams
        team_A_emb = self.team_embedding(input_data[:, 0])
        team_B_emb = self.team_embedding(input_data[:, 2])

        # 2) Embedding lookup for the seeds
        team_A_seed_emb = self.seed_embedding(input_data[:, 1])
        team_B_seed_emb = self.seed_embedding(input_data[:, 3])

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
        box_score_pred =               self.box_score_out(x)   # Regression
        win_proba_pred = torch.sigmoid(self.win_proba_out(x))  # Classification

        return box_score_pred, win_proba_pred

