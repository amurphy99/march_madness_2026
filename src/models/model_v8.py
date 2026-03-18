"""
Multi-head attention on game histories.
--------------------------------------------------------------------------------
`src.models.model_v8`

Copy of model_v5 with edits.

TODO: Maybe I should concatenate the team_embedding, seed_embedding, and Elo together
TODO: Increase the size of the history
TODO: Do something fancier when building the values ?

TODO: Change the output to dictionaries

"""
import torch
import torch.nn            as nn
import torch.nn.functional as F

# From this project
from ..config                         import DEFAULT_HISTORY_LEN, BOX_SCORE_DIM, HIST_NUMERIC_DIM
from ..processing.features.elo_rating import STARTING_ELO, ELO_WIDTH


# ================================================================================
# Model Definition
# ================================================================================
class MarchMadnessModel_v8(nn.Module):
    def __init__(
        self,
        num_teams: int, # How many entries the team embedding layer needs to prepare
        num_seeds: int, # How many entries the seed embedding layer needs to prepare
        *,

        # Embedding dimensions
        team_embed_dim: int = 96,                # Size of each team embedding vector
        seed_embed_dim: int = 32,                # Size of each seed embedding vector

        # Config Defaults
        hist_numeric_dim = HIST_NUMERIC_DIM,     # Size of one game in the numeric game history
        history_len      = DEFAULT_HISTORY_LEN,  # Number of games stored in history
        box_score_dim    = BOX_SCORE_DIM,        # Box-score prediction output dim
        
        # Attention Config
        dim_hist_hidden : int = 64,              # Size of attention values / outputs
        dim_hist_out    : int = 64,              # Size of post-attention project layer
        num_heads       : int = 4,               # Number of attention heads
        
        # MLP Layer Dimensions
        middle_dim : int = 128,                  # 1st linear layer of the MLP
        dim_outer  : int =  64,                  # 2nd linear layer of the MLP

        # Misc. Config
        dropout   : float = 0.25,                # Training dropout 
    ):
        super().__init__()
        self.dropout     = dropout
        self.history_len = history_len

        # --------------------------------------------------------------------------------
        # Embeddings
        # --------------------------------------------------------------------------------
        # Shared team & seed embedding tables
        self.team_embedding = nn.Embedding(num_teams, team_embed_dim, padding_idx=0)
        self.seed_embedding = nn.Embedding(num_seeds, seed_embed_dim, padding_idx=0)

        # Position embedding for game history
        self.pos_embedding = nn.Parameter(torch.randn(1, history_len, dim_hist_hidden)) 

        # --------------------------------------------------------------------------------
        # Attention Components
        # --------------------------------------------------------------------------------
        # Project history elements (Keys and Values)
        self.hist_kv_proj = nn.Linear(hist_numeric_dim + team_embed_dim, dim_hist_hidden)

        # Multi-Head Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim   = team_embed_dim, 
            num_heads   = num_heads, 
            vdim        = dim_hist_hidden,
            dropout     = dropout, 
            batch_first = True
        )

        # Final projection after attention
        self.hist_fc    = nn.Linear(team_embed_dim, dim_hist_out)
        self.hist_fc_bn = nn.BatchNorm1d           (dim_hist_out)

        # --------------------------------------------------------------------------------
        # Final fusion MLP 
        # --------------------------------------------------------------------------------
        # 2 attention results + 2 seed embeddings + 3 values for the Elo ratings
        fusion_dim = (dim_hist_out * 2) + (seed_embed_dim * 2) + 3 

        # Linear pass #1
        self.linear_1 = nn.Linear(fusion_dim, middle_dim)
        self.bn_1     = nn.BatchNorm1d       (middle_dim)

        # Linear pass #2
        self.linear_2 = nn.Linear(middle_dim, dim_outer)
        self.bn_2     = nn.BatchNorm1d       (dim_outer)

        # --------------------------------------------------------------------------------
        # Prediction Heads
        # --------------------------------------------------------------------------------
        # Box-Score Heads (mean + variance)
        self.box_score_mu      = nn.Linear(dim_outer, box_score_dim)
        self.box_score_log_var = nn.Linear(dim_outer, box_score_dim)

        # Win Probability Head
        self.win_logit         = nn.Linear(dim_outer, 1)

        # Evidential Head
        self.win_evidence      = nn.Linear(dim_outer, 2)

    # ================================================================================
    # Encode the historical data with Cross-Attention
    # ================================================================================
    def attend_history(self, query_emb, hist_numeric, hist_opp_ids, hist_mask):
        """
        query_emb:      (B, E) - The embedding we use to search the history
        hist_numeric:   (B, T, F)
        hist_opp_ids:   (B, T)
        hist_mask:      (B, T) - 1.0 for valid, 0.0 for padding
        """
        # --------------------------------------------------------------------------------
        # Build values from history
        # --------------------------------------------------------------------------------
        opp_emb = self.team_embedding(hist_opp_ids)    # (B, T, E)

        # Zero out padded timesteps
        mask         = hist_mask.unsqueeze(-1)         # (B, T, 1)
        opp_emb      = opp_emb      * mask
        hist_numeric = hist_numeric * mask

        x = torch.cat([hist_numeric, opp_emb], dim=-1) # (B, T, F+E)

        # Project Keys/Values
        kv = self.hist_kv_proj(x)                      # (B, T, H)
        kv = F.silu(kv)
        kv = F.dropout(kv, self.dropout, training=self.training)

        # Positional encoding
        kv = kv + self.pos_embedding

        # --------------------------------------------------------------------------------
        # Attention
        # --------------------------------------------------------------------------------
        # Invert the padding mask so it gives "True" for elements to ignore
        padding_mask = (hist_mask == 0.0).bool()       # (B, T)

        # Query is just the raw team embedding, unsqueezed to 3D
        query_3d = query_emb.unsqueeze(1)      # (B, 1, team_embed_dim)

        # Pass the exact tensors to their distinct roles
        attn_out, _ = self.attention(
            query            = query_3d,               # Pure Embedding (team_embed_dim)
            key              = opp_emb,                # Pure Embedding (team_embed_dim)
            value            = kv,                     # Projected Stats (dim_hist_hidden)
            key_padding_mask = padding_mask,
            need_weights     = False
        ) # Output is automatically projected back to (B, 1, team_embed_dim)

        # Remove the sequence dimension
        attn_out = attn_out.squeeze(1)                 # (B, team_embed_dim)

        # Final projection down to dim_hist_out 
        out = self.hist_fc(attn_out)
        out = self.hist_fc_bn(out)
        out = F.silu(out)

        return out

    # ================================================================================
    # Forward Pass 
    # ================================================================================
    def forward(self, batch):
        # 1) Embedding lookup for the teams
        teamA_id = batch["teamA_id"]
        teamB_id = batch["teamB_id"]
        teamA_emb = self.team_embedding(teamA_id)       # (B, E)
        teamB_emb = self.team_embedding(teamB_id)       # (B, E)

        # 2) Embedding lookup for the seeds
        teamA_seed = batch["teamA_seed"]
        teamB_seed = batch["teamB_seed"]
        team_A_seed_emb = self.seed_embedding(teamA_seed)
        team_B_seed_emb = self.seed_embedding(teamB_seed)

        # 2) Get and normalize Elo ratings
        teamA_elo = (batch["teamA_elo"].unsqueeze(-1) - STARTING_ELO) / ELO_WIDTH  # (B, 1)
        teamB_elo = (batch["teamB_elo"].unsqueeze(-1) - STARTING_ELO) / ELO_WIDTH  # (B, 1)
        elo_diff  = teamA_elo - teamB_elo

        # --------------------------------------------------------------------------------
        # 3) Cross-Attention
        # --------------------------------------------------------------------------------
        # Team A searches its history using Team B's embedding as the Query
        teamB_vs_teamA_hist = self.attend_history(
            query_emb    = teamB_emb,
            hist_numeric = batch["teamA_hist_numeric"],
            hist_opp_ids = batch["teamA_hist_opp_ids"],
            hist_mask    = batch["teamA_hist_mask"   ],
        )
        # Team B searches its history using Team A's embedding as the Query
        teamA_vs_teamB_hist = self.attend_history(
            query_emb    = teamA_emb,
            hist_numeric = batch["teamB_hist_numeric"],
            hist_opp_ids = batch["teamB_hist_opp_ids"],
            hist_mask    = batch["teamB_hist_mask"   ],
        )

        # --------------------------------------------------------------------------------
        # 4) Final fusion MLP 
        # --------------------------------------------------------------------------------
        # Fuse data together
        x = torch.cat([
            # Team embeddings (commented out)
            #teamA_emb,  teamB_emb, 

            # Seed embeddings
            team_A_seed_emb, team_B_seed_emb,

            # Cross-attention outputs
            teamB_vs_teamA_hist, teamA_vs_teamB_hist,

            # Elo ratings
            teamA_elo,  teamB_elo,  elo_diff,
        ], dim=-1)

        # Linear pass #1
        x = self.linear_1(x)
        x = self.bn_1    (x)
        x = F.silu       (x)
        x = F.dropout    (x, self.dropout, training=self.training)

        # Linear pass #2
        x = self.linear_2(x)
        x = self.bn_2    (x)
        x = F.silu       (x)
        x = F.dropout    (x, self.dropout, training=self.training)

        # --------------------------------------------------------------------------------
        # 6) Prediction Heads
        # --------------------------------------------------------------------------------
        # Box-score (mean + variance)
        box_mu      = self.box_score_mu     (x)
        box_log_var = self.box_score_log_var(x)

        # Win Probability 
        win_logit   = self.win_logit(x).squeeze(-1)

        # Calculate alpha and beta
        raw_evidence = self.win_evidence(x)
        alpha_beta   = F.softplus(raw_evidence) + 1.0

        return (box_mu, box_log_var), win_logit, alpha_beta
