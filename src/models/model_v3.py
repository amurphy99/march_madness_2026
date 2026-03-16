"""
Uses team embeddings as well as historical data (results + opponent IDs)
--------------------------------------------------------------------------------
`src.models.model_v3`

Integrated with Masked GAP and Residual Connections.

"""
import torch
import torch.nn            as nn
import torch.nn.functional as F

# From this project
from ..config import DEFAULT_HISTORY_LEN, BOX_SCORE_DIM, HIST_NUMERIC_DIM

# ================================================================================
# Model Definition
# ================================================================================
class MarchMadnessModel_v3(nn.Module):
    def __init__(
        self,
        num_teams,
        team_embed_dim,
        *,
        hist_numeric_dim = HIST_NUMERIC_DIM,
        history_len      = DEFAULT_HISTORY_LEN,
        hist_hidden_dim  =  64,
        hist_out_dim     =  64,
        middle_dim       = 128, # was 256
        dropout          = 0.3,
        box_score_dim    = BOX_SCORE_DIM,
    ):
        super().__init__()
        self.dropout     = dropout
        self.history_len = history_len

        # Shared team embedding table (0th index is reserved for padding)
        self.team_embedding = nn.Embedding(num_teams, team_embed_dim, padding_idx=0)

        # Per-timestep history feature projection
        self.hist_input_proj = nn.Linear(hist_numeric_dim + team_embed_dim, hist_hidden_dim)

        # Small CNN over last-N games
        self.hist_conv_1 = nn.Conv1d(hist_hidden_dim, hist_hidden_dim, kernel_size=3, padding=1)
        self.hist_bn_1   = nn.BatchNorm1d(hist_hidden_dim)

        self.hist_conv_2 = nn.Conv1d(hist_hidden_dim, hist_hidden_dim, kernel_size=3, padding=1)
        self.hist_bn_2   = nn.BatchNorm1d(hist_hidden_dim)

        # Projection after Global Average Pooling
        self.hist_fc    = nn.Linear(hist_hidden_dim, hist_out_dim)
        self.hist_fc_bn = nn.BatchNorm1d(hist_out_dim)

        # Final fusion MLP (+3 for Team A Elo, Team B Elo, and Elo Diff)
        fusion_dim = (team_embed_dim * 2) + (hist_out_dim * 2) + 3 # + team_embed_dim + hist_out_dim

        self.linear_1 = nn.Linear(fusion_dim, middle_dim)
        self.bn_1     = nn.BatchNorm1d(middle_dim)
        
        # New Residual Block in MLP
        self.linear_res = nn.Linear(middle_dim, middle_dim)
        self.bn_res     = nn.BatchNorm1d(middle_dim)

        self.linear_2 = nn.Linear(middle_dim, 64)
        self.bn_2     = nn.BatchNorm1d(64)

        # Heads
        self.box_score_out = nn.Linear(64, box_score_dim)
        self.win_out       = nn.Linear(64, 1)

    # ================================================================================
    # Encode the historical data with CNNs + GAP
    # ================================================================================
    def encode_history(self, hist_numeric, hist_opp_ids, hist_mask):
        """
        hist_numeric: (B, T, F)
        hist_opp_ids: (B, T)
        hist_mask:    (B, T)
        """
        # 1) Get historical opponent embeddings
        opp_emb = self.team_embedding(hist_opp_ids)    # (B, T, E)

        # Zero out padded timesteps at the input level
        mask         = hist_mask.unsqueeze(-1)         # (B, T, 1)
        opp_emb      = opp_emb      * mask
        hist_numeric = hist_numeric * mask

        # Concatenate numeric features + opponent embeddings
        x = torch.cat([hist_numeric, opp_emb], dim=-1) # (B, T, F+E)

        # Project each timestep
        x = self.hist_input_proj(x)                    # (B, T, H)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # 2) Convolutional block
        x = x.transpose(1, 2)                          # (B, H, T)

        # Conv #1
        x = self.hist_conv_1(x)
        x = self.hist_bn_1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Conv #2 with Residual Connection
        residual = x
        x = self.hist_conv_2(x)
        x = self.hist_bn_2(x)
        x = F.relu(x + residual)                       # Skip connection addition
        x = F.dropout(x, self.dropout, training=self.training)

        # 3) Masked Global Average Pooling
        # Zero out padded outputs before pooling
        conv_mask = hist_mask.unsqueeze(1)             # (B, 1, T)
        x = x * conv_mask
        
        # Sum over time, divide by the actual number of unpadded games
        true_lens = conv_mask.sum(dim=-1).clamp(min=1.0) # (B, 1)
        x = x.sum(dim=-1) / true_lens                    # (B, H)

        # Final history projection
        x = self.hist_fc(x)
        x = self.hist_fc_bn(x)
        x = F.relu(x)

        return x

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

        # 1) Get and normalize Elo ratings
        # Normalizing to keep values roughly between -2.0 and 2.0
        teamA_elo = (batch["teamA_elo"].unsqueeze(-1) - 1500.0) / 400.0  # (B, 1)
        teamB_elo = (batch["teamB_elo"].unsqueeze(-1) - 1500.0) / 400.0  # (B, 1)
        elo_diff  = teamA_elo - teamB_elo

        # 2) Get current team embeddings
        teamA_emb = self.team_embedding(teamA_id)
        teamB_emb = self.team_embedding(teamB_id)

        # 3) Process histories
        teamA_hist = self.encode_history(
            batch["teamA_hist_numeric"],
            batch["teamA_hist_opp_ids"],
            batch["teamA_hist_mask"],
        )

        teamB_hist = self.encode_history(
            batch["teamB_hist_numeric"],
            batch["teamB_hist_opp_ids"],
            batch["teamB_hist_mask"],
        )

        # 4) Calculate differences
        #current_diff = teamA_emb  - teamB_emb
        #hist_diff    = teamA_hist - teamB_hist

        # 5) Fuse everything together
        x = torch.cat([
            teamA_emb,  teamB_emb, 
            teamA_hist, teamB_hist, 
            #current_diff, hist_diff, 
            teamA_elo,  teamB_elo,  elo_diff,
        ], dim=-1)

        # --------------------------------------------------------------------------------
        # MLP with Residual
        # --------------------------------------------------------------------------------
        # Block 1
        x = self.linear_1(x)
        x = self.bn_1    (x)
        x = F.relu       (x)
        x = F.dropout    (x, self.dropout, training=self.training)

        # Block 2 (Residual)
        res = x
        x = self.linear_res(x)
        x = self.bn_res    (x)
        x = F.relu         (x + res)  # Skip connection residual
        x = F.dropout(x, self.dropout, training=self.training)

        # Block 3
        x = self.linear_2(x)
        x = self.bn_2    (x)
        x = F.relu       (x)

        # Prediction Heads
        box_score_pred = self.box_score_out(x)
        win_logit      = self.win_out      (x).squeeze(-1)

        return box_score_pred, win_logit 
    