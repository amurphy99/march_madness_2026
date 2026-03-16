"""
Model v6: Siamese Architecture with 4-Way Cross-Attention
--------------------------------------------------------------------------------
`src.models.model_v6`

Lean version:
1. Uses 4-Way Cross-Attention (Team A queries A, A queries B, etc.)
2. Uses Positional Encoding for temporal awareness
3. Evaluates both teams fairly through a shared Siamese Tower
4. Stripped of all MPS safety wrappers for maximum readability and speed

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# From this project
from ..config import DEFAULT_HISTORY_LEN, BOX_SCORE_DIM, HIST_NUMERIC_DIM

# ================================================================================
# Helper: Residual Block
# ================================================================================
class ResidualBlock(nn.Module):
    """Simple Residual Block to prevent vanishing gradients in deep MLPs."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.linear  = nn.Linear(dim, dim)
        self.bn      = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.bn(out)
        out = F.silu(out)
        out = self.dropout(out)
        return out + identity

# ================================================================================
# Model Definition
# ================================================================================
class MarchMadnessModel_v6(nn.Module):
    def __init__(
        self,
        num_teams,
        num_seeds,
        team_embed_dim  = 96,
        seed_embed_dim  = 32,
        *,
        hist_numeric_dim = HIST_NUMERIC_DIM,
        history_len      = DEFAULT_HISTORY_LEN,
        hist_hidden_dim  =  64,
        middle_dim       = 128,
        num_heads        =   4,
        dropout          = 0.25,
        box_score_dim    = BOX_SCORE_DIM,
    ):
        super().__init__()
        self.dropout         = dropout
        self.history_len     = history_len
        self.hist_hidden_dim = hist_hidden_dim

        # --------------------------------------------------------------------------------
        # 1) Embeddings & Positional Encoding
        # --------------------------------------------------------------------------------
        self.team_embedding = nn.Embedding(num_teams, team_embed_dim, padding_idx=0)
        self.seed_embedding = nn.Embedding(num_seeds, seed_embed_dim)
        
        self.pos_embedding  = nn.Parameter(torch.randn(1, history_len, hist_hidden_dim)) 

        # --------------------------------------------------------------------------------
        # 2) Attention Components (Keys, Values, Queries)
        # --------------------------------------------------------------------------------
        self.hist_conv = nn.Conv1d(
            in_channels  = hist_numeric_dim + team_embed_dim,
            out_channels = hist_hidden_dim,
            kernel_size  = 3,
            padding      = 1
        )
        self.hist_conv_bn = nn.BatchNorm1d(hist_hidden_dim)

        self.hist_input_proj = nn.Linear(hist_numeric_dim + team_embed_dim, hist_hidden_dim)
        self.query_proj      = nn.Linear(team_embed_dim, hist_hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = hist_hidden_dim,
            num_heads   = num_heads,
            dropout     = dropout, 
            batch_first = True
        )
        self.attn_layer_norm = nn.LayerNorm(hist_hidden_dim)

        # --------------------------------------------------------------------------------
        # 3) Siamese Tower
        # --------------------------------------------------------------------------------
        # Input: Team Emb + Seed Emb + Elo (1) + (Self-History Attn) + (Opp-History Attn)
        tower_in_dim = team_embed_dim + seed_embed_dim + 1 + (hist_hidden_dim * 2)

        self.siamese_tower = nn.Sequential(
            nn.Linear(tower_in_dim, middle_dim),
            nn.BatchNorm1d(middle_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # --------------------------------------------------------------------------------
        # 4) Residual MLP Backbone & Prediction Heads
        # --------------------------------------------------------------------------------
        merged_dim = middle_dim * 2

        self.res_block1 = ResidualBlock(merged_dim, dropout)
        self.res_block2 = ResidualBlock(merged_dim, dropout)

        self.box_mu      = nn.Linear(merged_dim, box_score_dim)
        self.box_log_var = nn.Linear(merged_dim, box_score_dim)
        self.win_logit   = nn.Linear(merged_dim, 1)

    # ================================================================================
    # 4-Way Cross-Attention
    # ================================================================================
    def attend_history(self, current_query_emb, hist_numeric, hist_opp_ids, hist_mask):
        """
        Uses an embedding (Query) to dynamically search a game history (Keys/Values).
        """
        # --------------------------------------------------------------------------------
        # 1) Build Keys & Values
        # --------------------------------------------------------------------------------
        opp_embs = self.team_embedding(hist_opp_ids)
        mask     = hist_mask.unsqueeze(-1)
        
        # Concat numeric stats and opponent embeddings
        x = torch.cat([hist_numeric * mask, opp_embs * mask], dim=-1)

        # --------------------------------------------------------------------------------
        # 2) Residual CNN Block
        # -------------------------------------------------------------------------------- 
        # a) Project to hidden dim
        res = self.hist_input_proj(x)
        res = F.silu(res)
        
        # b) Apply the CNN (Conv1d expects Shape: Batch, Channels, Length)
        x = x.transpose(1, 2)
        x = self.hist_conv(x)
        x = self.hist_conv_bn(x)
        x = x.transpose(1, 2)  # Flip back to (Batch, Length, Channels)

        # c) Add residual and apply final activation/dropout
        kv = F.silu(x + res)
        kv = F.dropout(kv, self.dropout, training=self.training)
        
        # --------------------------------------------------------------------------------
        # 3) Build Query 
        # --------------------------------------------------------------------------------
        q = self.query_proj(current_query_emb).unsqueeze(1)  # (B, 1, H)
        
        # --------------------------------------------------------------------------------
        # 4) Cross-Attention
        # --------------------------------------------------------------------------------
        padding_mask = (hist_mask == 0.0).bool()

        attn_out, _ = self.cross_attn(
            query            = q,
            key              = kv,
            value            = kv,
            key_padding_mask = padding_mask,
            need_weights     = False
        )
        
        attn_out = attn_out.squeeze(1)  # (B, H)
        
        # Add residual-style LayerNorm stabilization
        return self.attn_layer_norm(attn_out + q.squeeze(1))

    # ================================================================================
    # Forward pass
    # ================================================================================
    def forward(self, batch):
        # 1) Embedding Lookups
        teamA_emb = self.team_embedding(batch["teamA_id"])
        teamB_emb = self.team_embedding(batch["teamB_id"])
        
        seedA_emb = self.seed_embedding(batch["teamA_seed"])
        seedB_emb = self.seed_embedding(batch["teamB_seed"])
        
        # Normalize Elos
        teamA_elo = (batch["teamA_elo"].unsqueeze(-1) - 1500.0) / 400.0
        teamB_elo = (batch["teamB_elo"].unsqueeze(-1) - 1500.0) / 400.0

        # 2) 4-Way Cross-Attention
        teamA_vs_histA = self.attend_history(teamA_emb, batch["teamA_hist_numeric"], batch["teamA_hist_opp_ids"], batch["teamA_hist_mask"])
        teamA_vs_histB = self.attend_history(teamA_emb, batch["teamB_hist_numeric"], batch["teamB_hist_opp_ids"], batch["teamB_hist_mask"])

        teamB_vs_histB = self.attend_history(teamB_emb, batch["teamB_hist_numeric"], batch["teamB_hist_opp_ids"], batch["teamB_hist_mask"])
        teamB_vs_histA = self.attend_history(teamB_emb, batch["teamA_hist_numeric"], batch["teamA_hist_opp_ids"], batch["teamA_hist_mask"])

        # 3) Siamese Tower Construction (Strictly Symmetric)
        tower_A_input = torch.cat([teamA_emb, seedA_emb, teamA_elo, teamA_vs_histA, teamA_vs_histB], dim=-1)
        tower_B_input = torch.cat([teamB_emb, seedB_emb, teamB_elo, teamB_vs_histB, teamB_vs_histA], dim=-1)

        repr_A = self.siamese_tower(tower_A_input)
        repr_B = self.siamese_tower(tower_B_input)

        # 4) Symmetric Merging
        x = torch.cat([repr_A - repr_B, repr_A * repr_B], dim=-1)

        # 5) Residual MLP Backbone
        x = self.res_block1(x)
        x = self.res_block2(x)

        # 6) Prediction Heads
        box_mu      = self.box_mu     (x)
        box_log_var = self.box_log_var(x)
        win_logit   = self.win_logit  (x).squeeze(-1)

        return (box_mu, box_log_var), win_logit
    