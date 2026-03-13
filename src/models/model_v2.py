"""
Uses team embeddings as well as historical data (results + opponent IDs)
--------------------------------------------------------------------------------
`src.models.model_v2`

This means we can tune the team embeddings not only when they play games
themselves, but also when somebody who they played against recently plays a new
game. So if TeamA beats TeamB, and TeamC beats TeamA next, we might learn some 
info about how TeamC compares to TeamB through the game history.

"""
import torch
import torch.nn            as nn
import torch.nn.functional as F

# From this project
from ..config import DEFAULT_HISTORY_LEN, BOX_SCORE_DIM, HIST_NUMERIC_DIM

# ================================================================================
# Model Definition
# ================================================================================
class MarchMadnessModel_v2(nn.Module):
    def __init__(
        self,
        num_teams,
        team_embed_dim,
        *,
        hist_numeric_dim = HIST_NUMERIC_DIM,
        history_len      = DEFAULT_HISTORY_LEN,
        hist_hidden_dim  = 64,
        hist_out_dim     = 64,
        middle_dim       = 128, # was 256
        dropout          = 0.3,
        box_score_dim    = BOX_SCORE_DIM,
    ):
        super().__init__()
        self.dropout     = dropout
        self.history_len = history_len

        # Shared team embedding table (0th index is reserved for padding historical opponent IDs)
        self.team_embedding = nn.Embedding(num_teams, team_embed_dim, padding_idx=0)

        # Per-timestep history feature projection (game stats + opponent embedding)
        self.hist_input_proj = nn.Linear(hist_numeric_dim + team_embed_dim, hist_hidden_dim)

        # Small CNN over last-N games
        self.hist_conv_1 = nn.Conv1d(hist_hidden_dim, hist_hidden_dim, kernel_size=3, padding=1)
        self.hist_bn_1   = nn.BatchNorm1d(hist_hidden_dim)

        self.hist_conv_2 = nn.Conv1d(hist_hidden_dim, hist_hidden_dim, kernel_size=3, padding=1)
        self.hist_bn_2   = nn.BatchNorm1d(hist_hidden_dim)

        # Compress flattened sequence into one history vector
        self.hist_fc    = nn.Linear(hist_hidden_dim * history_len, hist_out_dim)
        self.hist_fc_bn = nn.BatchNorm1d(hist_out_dim)

        # Final fusion MLP 
        # (current A, current B, hist A, hist B, current diff, hist diff)
        # (+3 for Team A Elo, Team B Elo, and Elo Diff)
        fusion_dim = (team_embed_dim * 2) + (hist_out_dim * 2) + 3 #+ team_embed_dim + hist_out_dim

        self.linear_1 = nn.Linear(fusion_dim, middle_dim)
        self.bn_1     = nn.BatchNorm1d(middle_dim)

        self.linear_2 = nn.Linear(middle_dim, 64)
        self.bn_2     = nn.BatchNorm1d(64)

        # Heads
        self.box_score_out = nn.Linear(64, box_score_dim)
        self.win_out       = nn.Linear(64, 1)

    # ================================================================================
    # Encode the historical data with CNNs
    # ================================================================================
    def encode_history(self, hist_numeric, hist_opp_ids, hist_mask):
        """
        hist_numeric: (B, T, F)
        hist_opp_ids: (B, T)
        hist_mask:    (B, T)
        """
        # --------------------------------------------------------------------------------
        # Get historical opponent embeddings
        # --------------------------------------------------------------------------------
        opp_emb = self.team_embedding(hist_opp_ids)    # (B, T, E)

        # Zero out padded timesteps
        mask         = hist_mask.unsqueeze(-1)         # (B, T, 1)
        opp_emb      = opp_emb      * mask
        hist_numeric = hist_numeric * mask

        # Concatenate numeric features + opponent embeddings
        x = torch.cat([hist_numeric, opp_emb], dim=-1) # (B, T, F+E)

        # Project each timestep
        x = self.hist_input_proj(x)                  # (B, T, H)
        x = F.silu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # --------------------------------------------------------------------------------
        # Convolutional block
        # --------------------------------------------------------------------------------
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)                          # (B, H, T)

        # Conv #1 (with residual)
        residual = x
        x = self.hist_conv_1(x)
        x = self.hist_bn_1  (x)
        x = F.silu          (x + residual)
        x = F.dropout       (x, self.dropout, training=self.training)

        # Conv #2 (with residual)
        residual = x
        x = self.hist_conv_2(x)
        x = self.hist_bn_2  (x)
        x = F.silu          (x + residual)
        x = F.dropout       (x, self.dropout, training=self.training)

        # --------------------------------------------------------------------------------
        # Flatten on the way out (don't do average-pooling)
        # --------------------------------------------------------------------------------
        # Flattening preserves position / recency information across the history slots
        x = x.flatten(start_dim=1)                     # (B, H*T)

        # Final projection
        x = self.hist_fc   (x)
        x = self.hist_fc_bn(x)
        x = F.silu         (x)

        return x

    # ================================================================================
    # Forward pass 
    # ================================================================================
    def forward(self, batch):
        # Current matchup team IDs
        teamA_id = batch["teamA_id"]
        teamB_id = batch["teamB_id"]

        # 1) Get and normalize Elo ratings
        # Normalizing to keep values roughly between -2.0 and 2.0
        teamA_elo = (batch["teamA_elo"].unsqueeze(-1) - 1500.0) / 400.0  # (B, 1)
        teamB_elo = (batch["teamB_elo"].unsqueeze(-1) - 1500.0) / 400.0  # (B, 1)
        elo_diff  = teamA_elo - teamB_elo

        # 2) Get current team embeddings
        teamA_emb = self.team_embedding(teamA_id)       # (B, E)
        teamB_emb = self.team_embedding(teamB_id)       # (B, E)

        # 3) Process histories
        teamA_hist = self.encode_history(
            batch["teamA_hist_numeric"],
            batch["teamA_hist_opp_ids"],
            batch["teamA_hist_mask"   ],
        )

        teamB_hist = self.encode_history(
            batch["teamB_hist_numeric"],
            batch["teamB_hist_opp_ids"],
            batch["teamB_hist_mask"   ],
        )

        # 4) Calculate differences
        #current_diff = teamA_emb  - teamB_emb
        #hist_diff    = teamA_hist - teamB_hist

        # Fuse everything
        x = torch.cat([
            teamA_emb, teamB_emb, 
            teamA_hist, teamB_hist, 
            #current_diff, hist_diff
            teamA_elo,  teamB_elo,  elo_diff,
        ], dim=-1)

        # --------------------------------------------------------------------------------
        # MLP
        # --------------------------------------------------------------------------------
        x = self.linear_1(x)
        x = self.bn_1    (x)
        x = F.silu       (x)
        x = F.dropout    (x, self.dropout, training=self.training)

        x = self.linear_2(x)
        x = self.bn_2    (x)
        x = F.silu       (x)

        # Prediction Heads
        box_score_pred = self.box_score_out(x)
        win_logit      = self.win_out      (x).squeeze(-1)
        #win_prob       = torch.sigmoid(win_logit)

        return box_score_pred, win_logit # win_logit | win_prob
