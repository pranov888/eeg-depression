"""
Bidirectional LSTM Encoder for EEG Temporal Dynamics

This module captures sequential temporal patterns in EEG signals that may be
missed by the Transformer (which focuses on time-frequency) and GNN (spatial).

Design Decisions (for research paper):
--------------------------------------
1. Bidirectional: Captures both past→future and future→past dependencies
2. Multi-layer: Hierarchical temporal abstraction
3. Channel attention: Weights electrode importance before temporal modeling
4. Residual connections: Prevents gradient vanishing in deep networks

What Bi-LSTM Captures (vs other branches):
------------------------------------------
- Transformer: Global time-frequency patterns in scalograms
- GNN: Spatial inter-electrode relationships
- Bi-LSTM: Sequential temporal dynamics, transient events, temporal evolution

EEG-specific temporal patterns:
- Sleep spindles, K-complexes (if present)
- Rhythmic oscillation evolution
- Event-related dynamics
- Temporal asymmetries in depression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ChannelAttention(nn.Module):
    """
    Attention mechanism to weight electrode channels before temporal processing.

    Learns which electrodes are most informative for depression detection.
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        """
        Args:
            n_channels: Number of EEG channels (19 for 10-20 system)
            reduction: Reduction ratio for bottleneck
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Attention-weighted tensor (batch, channels, time)
        """
        # x: (B, C, T)
        b, c, t = x.size()

        # Global pooling across time
        avg_out = self.avg_pool(x).view(b, c)  # (B, C)
        max_out = self.max_pool(x).view(b, c)  # (B, C)

        # Channel attention weights
        avg_att = self.fc(avg_out)
        max_att = self.fc(max_out)
        attention = self.sigmoid(avg_att + max_att).unsqueeze(-1)  # (B, C, 1)

        return x * attention


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for EEG temporal sequence modeling.

    Processes multi-channel EEG as a sequence to capture temporal dynamics.
    """

    def __init__(
        self,
        input_dim: int = 19,          # Number of EEG channels
        hidden_dim: int = 128,         # LSTM hidden dimension
        num_layers: int = 2,           # Number of LSTM layers
        dropout: float = 0.3,          # Dropout rate
        output_dim: int = 128,         # Output feature dimension
        use_channel_attention: bool = True,
        bidirectional: bool = True
    ):
        """
        Initialize Bi-LSTM encoder.

        Args:
            input_dim: Number of input channels (EEG electrodes)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
            output_dim: Final output dimension
            use_channel_attention: Whether to use channel attention
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Optional channel attention
        self.use_channel_attention = use_channel_attention
        if use_channel_attention:
            self.channel_attention = ChannelAttention(input_dim)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Layer normalization after LSTM
        lstm_output_dim = hidden_dim * self.num_directions
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # Temporal attention for sequence aggregation
        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights for better training."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through Bi-LSTM encoder.

        Args:
            x: Input EEG tensor
               - If (batch, channels, time): Multi-channel EEG
               - If (batch, time, channels): Will be transposed
            return_sequence: Whether to return full sequence or just aggregated
            return_attention: Whether to return attention weights

        Returns:
            output: Encoded representation (batch, output_dim)
            or (output, attention_weights) if return_attention=True
        """
        # Handle input shape
        if x.dim() == 2:
            # (batch, time) -> assume single channel, add channel dim
            x = x.unsqueeze(1)

        # Ensure shape is (batch, channels, time)
        if x.size(1) > x.size(2):
            # Likely (batch, time, channels), transpose
            x = x.transpose(1, 2)

        batch_size, n_channels, seq_len = x.size()

        # Apply channel attention
        if self.use_channel_attention:
            x = self.channel_attention(x)  # (B, C, T)

        # Transpose to (batch, time, channels) for LSTM
        x = x.transpose(1, 2)  # (B, T, C)

        # Input projection
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Bi-LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # (B, T, hidden_dim * num_directions)

        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)

        if return_sequence:
            return self.output_proj(lstm_out)

        # Temporal attention aggregation
        attention_scores = self.temporal_attention(lstm_out)  # (B, T, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T, 1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (B, hidden_dim * 2)

        # Output projection
        output = self.output_proj(context)  # (B, output_dim)

        if return_attention:
            return output, attention_weights.squeeze(-1)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both channel and temporal attention weights for interpretability.

        Args:
            x: Input EEG tensor (batch, channels, time)

        Returns:
            channel_attention: (batch, channels)
            temporal_attention: (batch, time)
        """
        # This is useful for explainability
        output, temporal_attn = self.forward(x, return_attention=True)

        # Get channel attention if available
        if self.use_channel_attention:
            with torch.no_grad():
                # Compute channel attention weights
                if x.size(1) > x.size(2):
                    x = x.transpose(1, 2)
                b, c, t = x.size()
                avg_out = F.adaptive_avg_pool1d(x, 1).view(b, c)
                max_out = F.adaptive_max_pool1d(x, 1).view(b, c)
                avg_att = self.channel_attention.fc(avg_out)
                max_att = self.channel_attention.fc(max_out)
                channel_attn = torch.sigmoid(avg_att + max_att)
        else:
            channel_attn = torch.ones(x.size(0), x.size(1), device=x.device)

        return channel_attn, temporal_attn


class EEGBiLSTMEncoder(nn.Module):
    """
    Complete Bi-LSTM encoder with optional multi-scale processing.

    Can process EEG at multiple temporal resolutions for richer representations.
    """

    def __init__(
        self,
        input_dim: int = 19,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        multi_scale: bool = False
    ):
        super().__init__()

        self.multi_scale = multi_scale

        # Main encoder
        self.encoder = BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=output_dim if not multi_scale else output_dim // 2
        )

        if multi_scale:
            # Downsampled encoder for longer-range patterns
            self.encoder_slow = BiLSTMEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=1,
                dropout=dropout,
                output_dim=output_dim // 2
            )

            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG tensor (batch, channels, time)

        Returns:
            Encoded features (batch, output_dim)
        """
        # Main encoding at full resolution
        out_fast = self.encoder(x)

        if self.multi_scale:
            # Downsample by factor of 4 for slow encoder
            if x.dim() == 3:
                x_slow = F.avg_pool1d(x, kernel_size=4, stride=4)
            else:
                x_slow = x[:, ::4, :]

            out_slow = self.encoder_slow(x_slow)

            # Concatenate and fuse
            out = torch.cat([out_fast, out_slow], dim=-1)
            out = self.fusion(out)
            return out

        return out_fast


# Factory function
def create_bilstm_encoder(
    input_dim: int = 19,
    hidden_dim: int = 128,
    output_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    multi_scale: bool = False
) -> nn.Module:
    """
    Factory function to create Bi-LSTM encoder.

    Args:
        input_dim: Number of EEG channels
        hidden_dim: LSTM hidden dimension
        output_dim: Output feature dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        multi_scale: Whether to use multi-scale processing

    Returns:
        Bi-LSTM encoder module
    """
    return EEGBiLSTMEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        multi_scale=multi_scale
    )


if __name__ == "__main__":
    # Test the module
    print("Testing BiLSTM Encoder...")

    # Create encoder
    encoder = EEGBiLSTMEncoder(
        input_dim=19,
        hidden_dim=128,
        output_dim=128,
        num_layers=2
    )

    # Test input: (batch=4, channels=19, time=1000)
    x = torch.randn(4, 19, 1000)

    # Forward pass
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test attention
    base_encoder = encoder.encoder
    out, attn = base_encoder(x, return_attention=True)
    print(f"Attention shape: {attn.shape}")

    # Parameter count
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {n_params:,}")

    print("BiLSTM Encoder test PASSED!")
