"""
Transformer lookup using embeddings and game histories.
--------------------------------------------------------------------------------
`src.models.model_v5`

Uses Cross-Attention to dynamically query game histories based on the current 
matchup.

Same box-score mean + variance head as in model_v4. 

"""
import torch
import torch.nn            as nn
import torch.nn.functional as F

# From this project
from ..config import DEFAULT_HISTORY_LEN, BOX_SCORE_DIM, HIST_NUMERIC_DIM

# ================================================================================
# Model Definition
# ================================================================================
class MarchMadnessModel_v5(nn.Module):
    def __init__(
        self,
        num_teams,
        num_seeds,
        team_embed_dim: int = 96,
        seed_embed_dim: int = 32,
        *,
        hist_numeric_dim = HIST_NUMERIC_DIM,
        history_len      = DEFAULT_HISTORY_LEN,
        hist_hidden_dim  = 64,
        hist_out_dim     = 64,
        middle_dim       = 128, 
        dropout          = 0.25,
        box_score_dim    = BOX_SCORE_DIM,  # Box-score prediction output dim 
        num_heads        = 4,              # Number of attention heads
    ):
        super().__init__()
        self.dropout     = dropout
        self.history_len = history_len

        # Shared team embedding table
        self.team_embedding = nn.Embedding(num_teams, team_embed_dim, padding_idx=0)
        self.seed_embedding = nn.Embedding(num_seeds, seed_embed_dim)

        # --------------------------------------------------------------------------------
        # Attention Components
        # --------------------------------------------------------------------------------
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, history_len, hist_hidden_dim)) 

        # 1) Project history elements (Keys and Values)
        self.hist_kv_proj = nn.Linear(hist_numeric_dim + team_embed_dim, hist_hidden_dim)
        
        # 2) Project current opponent embedding (Query)
        self.query_proj = nn.Linear(team_embed_dim, hist_hidden_dim)

        # 3) Multi-Head Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim   = hist_hidden_dim, 
            num_heads   = num_heads, 
            dropout     = dropout, 
            batch_first = True
        )
        self.attn_layer_norm = nn.LayerNorm(hist_hidden_dim)

        # Final projection after attention
        self.hist_fc    = nn.Linear(hist_hidden_dim, hist_out_dim)
        self.hist_fc_bn = nn.BatchNorm1d(hist_out_dim)

        # --------------------------------------------------------------------------------
        # Final fusion MLP 
        # --------------------------------------------------------------------------------
        #fusion_dim = (team_embed_dim * 2) + (hist_out_dim * 2) + 3 
        fusion_dim = (hist_out_dim * 4) + (seed_embed_dim * 2) + 3 

        self.linear_1 = nn.Linear(fusion_dim, middle_dim)
        self.bn_1     = nn.BatchNorm1d(middle_dim)

        self.linear_2 = nn.Linear(middle_dim, 64)
        self.bn_2     = nn.BatchNorm1d(64)

        # Heads (mean+variance box score heads & simple win/loss head)
        self.box_score_mu      = nn.Linear(64, box_score_dim)
        self.box_score_log_var = nn.Linear(64, box_score_dim)
        self.win_logit         = nn.Linear(64, 1)

        # Evidential Head
        self.win_evidence      = nn.Linear(64, 2)

    # ================================================================================
    # Encode the historical data with Cross-Attention
    # ================================================================================
    def attend_history(self, current_query_emb, hist_numeric, hist_opp_ids, hist_mask):
        """
        current_query_emb: (B, E) - The embedding we use to search the history
        hist_numeric:      (B, T, F)
        hist_opp_ids:      (B, T)
        hist_mask:         (B, T) - 1.0 for valid, 0.0 for padding
        """
        # --------------------------------------------------------------------------------
        # Build Keys & Values from history
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
        # Build Query from current matchup
        # --------------------------------------------------------------------------------
        # Project and add a sequence dimension of length 1 -> (B, 1, H)
        q = self.query_proj(current_query_emb).unsqueeze(1) 

        # --------------------------------------------------------------------------------
        # Attention
        # --------------------------------------------------------------------------------
        # PyTorch needs 'True' for elements to IGNORE. 
        # Since your mask is 1.0 for valid games and 0.0 for padding, we invert it.
        padding_mask = (hist_mask == 0.0).bool()       # (B, T)

        # Run MultiHead Attention
        attn_out, _ = self.attention(
            query            = q,
            key              = kv,
            value            = kv,
            key_padding_mask = padding_mask,
            need_weights     = False
        ) # Output is (B, 1, H)

        # Remove the sequence dimension
        attn_out = attn_out.squeeze(1)                 # (B, H)
        
        # Add residual-style LayerNorm stabilization
        attn_out = self.attn_layer_norm(attn_out + q.squeeze(1)) 

        # Final projection
        out = self.hist_fc(attn_out)
        out = self.hist_fc_bn(out)
        out = F.silu(out)

        return out

    # ================================================================================
    # Forward Pass 
    # ================================================================================
    def forward(self, batch):
        # 1) Get current team embeddings
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
        teamA_elo = (batch["teamA_elo"].unsqueeze(-1) - 1500.0) / 400.0  # (B, 1)
        teamB_elo = (batch["teamB_elo"].unsqueeze(-1) - 1500.0) / 400.0  # (B, 1)
        elo_diff  = teamA_elo - teamB_elo

        # --------------------------------------------------------------------------------
        # 3) CROSS-ATTENTION
        # --------------------------------------------------------------------------------
        # a) Team A searches its history using Team B's embedding as the Query
        teamB_vs_teamA_hist = self.attend_history(
            current_query_emb = teamB_emb,  # Query
            hist_numeric      = batch["teamA_hist_numeric"],
            hist_opp_ids      = batch["teamA_hist_opp_ids"],
            hist_mask         = batch["teamA_hist_mask"   ],
        )

        # b) Team A searches its history using Team A's embedding as the Query
        teamA_vs_teamA_hist = self.attend_history(
            current_query_emb = teamA_emb,  # Query
            hist_numeric      = batch["teamA_hist_numeric"],
            hist_opp_ids      = batch["teamA_hist_opp_ids"],
            hist_mask         = batch["teamA_hist_mask"   ],
        )

        # c) Team B searches its history using Team A's embedding as the Query
        teamA_vs_teamB_hist = self.attend_history(
            current_query_emb = teamA_emb,  # Query
            hist_numeric      = batch["teamB_hist_numeric"],
            hist_opp_ids      = batch["teamB_hist_opp_ids"],
            hist_mask         = batch["teamB_hist_mask"  ],
        )

        # d) Team B searches its history using Team B's embedding as the Query
        teamB_vs_teamB_hist = self.attend_history(
            current_query_emb = teamB_emb,  # Query
            hist_numeric      = batch["teamB_hist_numeric"],
            hist_opp_ids      = batch["teamB_hist_opp_ids"],
            hist_mask         = batch["teamB_hist_mask"  ],
        )


        # Fuse everything
        x = torch.cat([
            # Team embeddings
            #teamA_emb,  teamB_emb, 

            # Seed embeddings
            team_A_seed_emb, team_B_seed_emb,

            # Cross-attention outputs
            teamB_vs_teamA_hist, teamA_vs_teamA_hist, 
            teamA_vs_teamB_hist, teamB_vs_teamB_hist,

            # Elo ratings
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

        # --------------------------------------------------------------------------------
        # Prediction Heads
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
