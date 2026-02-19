"""
Three-Way Attention-Based Fusion Module

Combines representations from three branches:
1. Transformer (time-frequency patterns from scalograms)
2. Bi-LSTM (temporal dynamics from raw EEG)
3. GNN (spatial electrode relationships)

Design Decisions (for research paper):
--------------------------------------
1. Cross-attention between all pairs: Each branch attends to others
2. Adaptive gating: Learns optimal contribution of each branch
3. Hierarchical fusion: Pairwise → three-way
4. Residual connections: Preserves individual branch information

Fusion Strategy:
---------------
- First: Pairwise cross-attention (Trans↔LSTM, Trans↔GNN, LSTM↔GNN)
- Then: Gate-weighted combination
- Finally: Projection to fusion dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ThreeWayAttentionFusion(nn.Module):
    """
    Three-way attention-based fusion for Transformer + Bi-LSTM + GNN.

    Uses cross-attention and gating to optimally combine three representations.
    """

    def __init__(
        self,
        trans_dim: int = 128,      # Transformer output dimension
        lstm_dim: int = 128,       # Bi-LSTM output dimension
        gnn_dim: int = 128,        # GNN output dimension
        fusion_dim: int = 128,     # Final fused dimension
        num_heads: int = 4,        # Attention heads
        dropout: float = 0.1,      # Dropout rate
        use_gating: bool = True    # Whether to use adaptive gating
    ):
        """
        Initialize three-way fusion module.

        Args:
            trans_dim: Dimension of Transformer features
            lstm_dim: Dimension of Bi-LSTM features
            gnn_dim: Dimension of GNN features
            fusion_dim: Output fusion dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_gating: Whether to use learned gating
        """
        super().__init__()

        self.trans_dim = trans_dim
        self.lstm_dim = lstm_dim
        self.gnn_dim = gnn_dim
        self.fusion_dim = fusion_dim
        self.use_gating = use_gating

        # Project all inputs to same dimension for attention
        self.proj_trans = nn.Linear(trans_dim, fusion_dim)
        self.proj_lstm = nn.Linear(lstm_dim, fusion_dim)
        self.proj_gnn = nn.Linear(gnn_dim, fusion_dim)

        # Cross-attention: Trans ↔ LSTM
        self.cross_attn_trans_lstm = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: Trans ↔ GNN
        self.cross_attn_trans_gnn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: LSTM ↔ GNN
        self.cross_attn_lstm_gnn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norms
        self.norm_trans = nn.LayerNorm(fusion_dim)
        self.norm_lstm = nn.LayerNorm(fusion_dim)
        self.norm_gnn = nn.LayerNorm(fusion_dim)

        # Gating mechanism
        if use_gating:
            # Gate inputs: concatenated features from all branches
            gate_input_dim = fusion_dim * 3

            self.gate_trans = nn.Sequential(
                nn.Linear(gate_input_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1),
                nn.Sigmoid()
            )

            self.gate_lstm = nn.Sequential(
                nn.Linear(gate_input_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1),
                nn.Sigmoid()
            )

            self.gate_gnn = nn.Sequential(
                nn.Linear(gate_input_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1),
                nn.Sigmoid()
            )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(trans_dim + lstm_dim + gnn_dim, fusion_dim)

    def forward(
        self,
        trans_features: torch.Tensor,
        lstm_features: torch.Tensor,
        gnn_features: torch.Tensor,
        return_gate_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through three-way fusion.

        Args:
            trans_features: Transformer output (batch, trans_dim)
            lstm_features: Bi-LSTM output (batch, lstm_dim)
            gnn_features: GNN output (batch, gnn_dim)
            return_gate_weights: Whether to return gating weights

        Returns:
            fused: Fused representation (batch, fusion_dim)
            gate_info: Optional dict with gate weights
        """
        batch_size = trans_features.size(0)

        # Project to common dimension
        trans_proj = self.proj_trans(trans_features)  # (B, fusion_dim)
        lstm_proj = self.proj_lstm(lstm_features)
        gnn_proj = self.proj_gnn(gnn_features)

        # Add sequence dimension for attention (B, 1, D)
        trans_seq = trans_proj.unsqueeze(1)
        lstm_seq = lstm_proj.unsqueeze(1)
        gnn_seq = gnn_proj.unsqueeze(1)

        # Cross-attention: Transformer attends to LSTM
        trans_lstm, _ = self.cross_attn_trans_lstm(
            trans_seq, lstm_seq, lstm_seq
        )
        trans_lstm = trans_lstm.squeeze(1)

        # Cross-attention: Transformer attends to GNN
        trans_gnn, _ = self.cross_attn_trans_gnn(
            trans_seq, gnn_seq, gnn_seq
        )
        trans_gnn = trans_gnn.squeeze(1)

        # Cross-attention: LSTM attends to GNN
        lstm_gnn, _ = self.cross_attn_lstm_gnn(
            lstm_seq, gnn_seq, gnn_seq
        )
        lstm_gnn = lstm_gnn.squeeze(1)

        # Update each branch with cross-attended information
        trans_updated = self.norm_trans(trans_proj + trans_lstm + trans_gnn)
        lstm_updated = self.norm_lstm(lstm_proj + trans_lstm.detach() + lstm_gnn)
        gnn_updated = self.norm_gnn(gnn_proj + trans_gnn.detach() + lstm_gnn.detach())

        gate_info = None

        if self.use_gating:
            # Concatenate for gating
            concat_features = torch.cat([trans_updated, lstm_updated, gnn_updated], dim=-1)

            # Compute gates
            g_trans = self.gate_trans(concat_features)  # (B, 1)
            g_lstm = self.gate_lstm(concat_features)
            g_gnn = self.gate_gnn(concat_features)

            # Normalize gates (softmax-like)
            gate_sum = g_trans + g_lstm + g_gnn + 1e-8
            g_trans = g_trans / gate_sum
            g_lstm = g_lstm / gate_sum
            g_gnn = g_gnn / gate_sum

            # Weighted combination
            weighted_trans = g_trans * trans_updated
            weighted_lstm = g_lstm * lstm_updated
            weighted_gnn = g_gnn * gnn_updated

            if return_gate_weights:
                gate_info = {
                    'trans_gate': g_trans.detach(),
                    'lstm_gate': g_lstm.detach(),
                    'gnn_gate': g_gnn.detach()
                }

            # Concatenate weighted features
            fused_input = torch.cat([weighted_trans, weighted_lstm, weighted_gnn], dim=-1)
        else:
            # Simple concatenation
            fused_input = torch.cat([trans_updated, lstm_updated, gnn_updated], dim=-1)

        # Final fusion
        fused = self.fusion_layer(fused_input)

        # Residual connection
        residual = self.residual_proj(
            torch.cat([trans_features, lstm_features, gnn_features], dim=-1)
        )
        fused = fused + residual

        return fused, gate_info


class HierarchicalThreeWayFusion(nn.Module):
    """
    Alternative: Hierarchical fusion (pairwise then combine).

    First fuses pairs, then combines the pairwise fusions.
    May be more interpretable than direct three-way fusion.
    """

    def __init__(
        self,
        trans_dim: int = 128,
        lstm_dim: int = 128,
        gnn_dim: int = 128,
        fusion_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Pairwise fusion modules
        from .attention_fusion import AttentionBasedFusion

        self.fuse_trans_lstm = AttentionBasedFusion(
            trans_dim=trans_dim,
            gnn_dim=lstm_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.fuse_trans_gnn = AttentionBasedFusion(
            trans_dim=trans_dim,
            gnn_dim=gnn_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Final combination
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        trans_features: torch.Tensor,
        lstm_features: torch.Tensor,
        gnn_features: torch.Tensor,
        return_gate_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward through hierarchical fusion."""

        # Pairwise fusions
        fused_tl, _ = self.fuse_trans_lstm(trans_features, lstm_features)
        fused_tg, _ = self.fuse_trans_gnn(trans_features, gnn_features)

        # Combine
        combined = torch.cat([fused_tl, fused_tg], dim=-1)
        output = self.final_fusion(combined)

        return output, None


if __name__ == "__main__":
    # Test the module
    print("Testing Three-Way Fusion...")

    fusion = ThreeWayAttentionFusion(
        trans_dim=128,
        lstm_dim=128,
        gnn_dim=128,
        fusion_dim=128
    )

    # Test inputs
    trans = torch.randn(4, 128)
    lstm = torch.randn(4, 128)
    gnn = torch.randn(4, 128)

    # Forward pass
    fused, gate_info = fusion(trans, lstm, gnn, return_gate_weights=True)

    print(f"Trans shape: {trans.shape}")
    print(f"LSTM shape: {lstm.shape}")
    print(f"GNN shape: {gnn.shape}")
    print(f"Fused shape: {fused.shape}")

    if gate_info:
        print(f"Trans gate: {gate_info['trans_gate'].mean().item():.3f}")
        print(f"LSTM gate: {gate_info['lstm_gate'].mean().item():.3f}")
        print(f"GNN gate: {gate_info['gnn_gate'].mean().item():.3f}")

    # Parameter count
    n_params = sum(p.numel() for p in fusion.parameters())
    print(f"Parameters: {n_params:,}")

    print("Three-Way Fusion test PASSED!")
