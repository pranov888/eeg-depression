"""
Attention-Based Feature Fusion Module

This module fuses representations from the Transformer (time-frequency) and
GNN (spatial) branches using cross-attention and gating mechanisms.

Design Decisions (for research paper):
--------------------------------------
1. WHY Cross-Attention?
   - Allows each branch to attend to the other
   - Learns which aspects of one modality are relevant given the other
   - More expressive than simple concatenation or averaging

2. WHY Gating?
   - Different samples may benefit more from one branch vs. the other
   - Learned gates adaptively weight the contributions
   - Provides interpretability (which branch was more important?)

3. FUSION STRATEGY:
   a. Self-attention on each branch (refine representations)
   b. Cross-attention (exchange information between branches)
   c. Gated combination (adaptive weighting)
   d. Final projection (fused representation)

References:
-----------
- Vaswani et al. (2017). Attention is all you need. NeurIPS.
- Lu et al. (2019). ViLBERT: Pretraining task-agnostic visiolinguistic representations.
- Zhang et al. (2023). Multi-modal fusion with transformers for EEG analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class CrossAttention(nn.Module):
    """
    Cross-attention module where one modality attends to another.

    Query from modality A, Key/Value from modality B.
    Learns: "Given A, what aspects of B are relevant?"
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: Query tensor from modality A (batch, seq_len_q, dim)
            key_value: Key/Value tensor from modality B (batch, seq_len_kv, dim)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]

        # Project to Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)

        if return_attention:
            return out, attn
        return out, None


class GatingMechanism(nn.Module):
    """
    Learned gating mechanism for adaptive feature combination.

    Learns to weight contributions from different sources based on
    the input content.
    """

    def __init__(self, dim: int, num_sources: int = 2):
        """
        Args:
            dim: Feature dimension
            num_sources: Number of sources to gate
        """
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * num_sources, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, num_sources),
            nn.Softmax(dim=-1)
        )

    def forward(self, *sources: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gated combination of sources.

        Args:
            sources: Variable number of source tensors (batch, dim)

        Returns:
            Gated output and gate weights
        """
        # Concatenate sources for gate computation
        concat = torch.cat(sources, dim=-1)

        # Compute gate weights
        weights = self.gate(concat)  # (batch, num_sources)

        # Weighted combination
        stacked = torch.stack(sources, dim=-1)  # (batch, dim, num_sources)
        output = (stacked * weights.unsqueeze(1)).sum(dim=-1)

        return output, weights


class AttentionBasedFusion(nn.Module):
    """
    Attention-based fusion module for combining Transformer and GNN outputs.

    Architecture:
    1. Self-attention refinement on each branch
    2. Bidirectional cross-attention
    3. Gated fusion with learned weights
    4. Final projection

    This design allows:
    - Each branch to refine its own representation
    - Information exchange between modalities
    - Adaptive weighting based on input
    - Rich fused representation
    """

    def __init__(
        self,
        trans_dim: int = 128,
        gnn_dim: int = 128,
        fusion_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        """
        Args:
            trans_dim: Transformer output dimension
            gnn_dim: GNN output dimension
            fusion_dim: Output dimension after fusion
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_gating: Whether to use gating mechanism
        """
        super().__init__()

        self.trans_dim = trans_dim
        self.gnn_dim = gnn_dim
        self.fusion_dim = fusion_dim
        self.use_gating = use_gating

        # Project to common dimension if needed
        self.trans_proj = nn.Linear(trans_dim, fusion_dim) if trans_dim != fusion_dim else nn.Identity()
        self.gnn_proj = nn.Linear(gnn_dim, fusion_dim) if gnn_dim != fusion_dim else nn.Identity()

        # Self-attention for each branch
        self.trans_self_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.gnn_self_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention
        self.trans_cross_attn = CrossAttention(fusion_dim, num_heads, dropout)
        self.gnn_cross_attn = CrossAttention(fusion_dim, num_heads, dropout)

        # Layer norms
        self.trans_norm1 = nn.LayerNorm(fusion_dim)
        self.trans_norm2 = nn.LayerNorm(fusion_dim)
        self.gnn_norm1 = nn.LayerNorm(fusion_dim)
        self.gnn_norm2 = nn.LayerNorm(fusion_dim)

        # Gating mechanism
        if use_gating:
            self.gate = GatingMechanism(fusion_dim, num_sources=2)

        # Final fusion layer
        input_dim = fusion_dim if use_gating else fusion_dim * 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def forward(
        self,
        trans_feat: torch.Tensor,
        gnn_feat: torch.Tensor,
        return_gate_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for feature fusion.

        Args:
            trans_feat: Transformer features (batch, trans_dim)
            gnn_feat: GNN features (batch, gnn_dim)
            return_gate_weights: Whether to return attention/gate weights

        Returns:
            Fused features (batch, fusion_dim) and optional weights dict
        """
        # Project to common dimension
        trans_h = self.trans_proj(trans_feat)
        gnn_h = self.gnn_proj(gnn_feat)

        # Add sequence dimension for attention (batch, 1, dim)
        trans_seq = trans_h.unsqueeze(1)
        gnn_seq = gnn_h.unsqueeze(1)

        # Self-attention refinement
        trans_self, trans_self_attn = self.trans_self_attn(
            trans_seq, trans_seq, trans_seq
        )
        trans_seq = self.trans_norm1(trans_seq + trans_self)

        gnn_self, gnn_self_attn = self.gnn_self_attn(
            gnn_seq, gnn_seq, gnn_seq
        )
        gnn_seq = self.gnn_norm1(gnn_seq + gnn_self)

        # Cross-attention: each branch attends to the other
        # Transformer attends to GNN: "What spatial info is relevant?"
        trans_cross, trans_cross_attn = self.trans_cross_attn(
            trans_seq, gnn_seq, return_attention=True
        )
        trans_seq = self.trans_norm2(trans_seq + trans_cross)

        # GNN attends to Transformer: "What temporal info is relevant?"
        gnn_cross, gnn_cross_attn = self.gnn_cross_attn(
            gnn_seq, trans_seq, return_attention=True
        )
        gnn_seq = self.gnn_norm2(gnn_seq + gnn_cross)

        # Remove sequence dimension
        trans_out = trans_seq.squeeze(1)
        gnn_out = gnn_seq.squeeze(1)

        # Gated or concatenated fusion
        if self.use_gating:
            fused, gate_weights = self.gate(trans_out, gnn_out)
        else:
            fused = torch.cat([trans_out, gnn_out], dim=-1)
            gate_weights = None

        # Final projection
        output = self.fusion_layer(fused)

        if return_gate_weights:
            weights_dict = {
                'gate_weights': gate_weights,
                'trans_cross_attn': trans_cross_attn,
                'gnn_cross_attn': gnn_cross_attn
            }
            return output, weights_dict

        return output, None


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion that combines features at multiple levels.

    This is an extension for future work that could fuse features
    at different granularities (e.g., local patches + global).
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        fusion_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            dims: Dimensions of features at each level
            fusion_dim: Output dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        # Project each level to common dimension
        self.projections = nn.ModuleList([
            nn.Linear(d, fusion_dim) for d in dims
        ])

        # Attention-based aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Fuse features from multiple levels.

        Args:
            features: Tuple of feature tensors at different levels

        Returns:
            Fused representation
        """
        # Project all features
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]

        # Stack as sequence
        stacked = torch.stack(projected, dim=1)  # (batch, n_levels, fusion_dim)

        # Self-attention aggregation
        attended, _ = self.attention(stacked, stacked, stacked)
        attended = self.norm(stacked + attended)

        # Global pooling (mean over levels)
        output = attended.mean(dim=1)

        return output
