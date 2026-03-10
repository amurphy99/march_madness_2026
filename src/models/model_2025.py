"""
Model architecture from last year.
--------------------------------------------------------------------------------
`src.models.2025_model`

"""
import torch
import torch.nn            as nn
import torch.nn.functional as F


class MarchMadnessModel2025(nn.Module):

    def __init__(self, num_teams, num_seeds, team_embed_dim, seed_embed_dim, dropout):
        super(MarchMadnessModel2025, self).__init__()
        self.dropout = dropout
        
        # Team ID & Seed Embeddings
        self.team_embedding = nn.Embedding(num_teams, team_embed_dim)
        self.seed_embedding = nn.Embedding(num_seeds, seed_embed_dim)
        
        # Hidden Linear Layers
        self.linear_1 = nn.Linear((team_embed_dim+seed_embed_dim)*2, 256)
        self.linear_2 = nn.Linear(256, 64)
        #self.linear_3 = nn.Linear(64, 64)
        
        # Output Layers
        self.box_score_out = nn.Linear(64, 26)
        self.win_proba_out = nn.Linear(64,  1)
        
    def forward(self, input_data):
        #print(input_data)
        # 1) Embedding lookup for the teams
        team_A_emb = self.team_embedding(input_data[:, 0])
        team_B_emb = self.team_embedding(input_data[:, 2])
        
        # 2) Embedding lookup for the seeds
        team_A_seed_emb = self.seed_embedding(input_data[:, 1])
        team_B_seed_emb = self.seed_embedding(input_data[:, 3])

        # 3) Concatenate all of them
        x = torch.cat([team_A_emb, team_A_seed_emb, team_B_emb, team_B_seed_emb], dim=-1)
        
        # 4) Pass the data through the linear layers
        x = F.tanh(self.linear_1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.tanh(self.linear_2(x))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.tanh(self.linear_3(x))
        
        # 5) Outputs
        box_score_pred =               self.box_score_out(x)  # Regression
        win_proba_pred = torch.sigmoid(self.win_proba_out(x)) # Classification
        
        return box_score_pred, win_proba_pred

