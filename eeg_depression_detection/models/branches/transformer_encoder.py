"""
Transformer Encoder for EEG Scalogram Analysis

Processes CWT scalograms using Vision Transformer-style architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Convert scalogram into patch embeddings.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 128),
        patch_size: Tuple[int, int] = (8, 16),
        in_channels: int = 1,
        embed_dim: int = 128
    ):
        """
        Args:
            img_size: (height, width) of input scalogram
            patch_size: (patch_height, patch_width)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_dim = patch_size[0] * patch_size[1] * in_channels

        self.projection = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Scalogram of shape (batch, channels, height, width)

        Returns:
            Patch embeddings of shape (batch, n_patches, embed_dim)
        """
        B, C, H, W = x.shape
        ph, pw = self.patch_size

        # Reshape into patches: (B, C, H/ph, ph, W/pw, pw)
        x = x.reshape(B, C, H // ph, ph, W // pw, pw)
        # Permute and flatten patches: (B, n_patches, patch_dim)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, self.patch_dim)

        # Project to embedding dimension
        x = self.projection(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for patch sequences.
    """

    def __init__(self, n_positions: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_positions, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return self.dropout(x + self.pos_embed[:, :x.size(1)])


class EEGTransformerEncoder(nn.Module):
    """
    Transformer Encoder for EEG scalogram classification.

    Based on Vision Transformer (ViT) architecture adapted for time-frequency analysis.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 128),
        patch_size: Tuple[int, int] = (8, 16),
        in_channels: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        """
        Args:
            img_size: (height, width) of input scalogram
            patch_size: (patch_height, patch_width)
            in_channels: Number of input channels (1 for single scalogram, n_channels for multi)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_ff: Feed-forward dimension
            dropout: Dropout probability
            use_cls_token: Whether to use a CLS token for classification
        """
        super().__init__()

        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=d_model
        )
        n_patches = self.patch_embed.n_patches

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            n_positions = n_patches + 1
        else:
            n_positions = n_patches

        # Positional encoding
        self.pos_encoding = PositionalEncoding(n_positions, d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Output layer norm
        self.norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Scalogram of shape (batch, channels, height, width)
               or (batch, height, width) for single channel
            return_attention: Whether to return attention weights

        Returns:
            Output embedding of shape (batch, d_model)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, d_model)

        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Normalize
        x = self.norm(x)

        # Extract CLS token or use mean pooling
        if self.use_cls_token:
            output = x[:, 0]  # CLS token
        else:
            output = x.mean(dim=1)  # Mean pooling

        return output

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps from all layers.

        Args:
            x: Input scalogram

        Returns:
            Attention maps for visualization
        """
        # This requires modifying the transformer to store attention weights
        # For now, return None (implement if needed for XAI)
        return None


class MultiScaleTransformer(nn.Module):
    """
    Multi-scale Transformer that processes scalograms at different resolutions.
    """

    def __init__(
        self,
        scales: Tuple[Tuple[int, int], ...] = ((64, 128), (32, 64), (16, 32)),
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            scales: Tuple of (height, width) for each scale
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Layers per scale
            dropout: Dropout probability
        """
        super().__init__()

        self.encoders = nn.ModuleList()
        for scale in scales:
            encoder = EEGTransformerEncoder(
                img_size=scale,
                patch_size=(scale[0] // 4, scale[1] // 8),
                d_model=d_model // len(scales),
                nhead=max(1, nhead // len(scales)),
                num_layers=num_layers,
                dropout=dropout
            )
            self.encoders.append(encoder)

        # Fusion layer
        self.fusion = nn.Linear(d_model, d_model)

    def forward(self, scalograms: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            scalograms: Tuple of scalograms at different scales

        Returns:
            Fused embedding
        """
        embeddings = []
        for encoder, scalogram in zip(self.encoders, scalograms):
            emb = encoder(scalogram)
            embeddings.append(emb)

        # Concatenate and fuse
        combined = torch.cat(embeddings, dim=-1)
        return self.fusion(combined)
