"""
Complete EEG Depression Detection Model V2: Transformer + Bi-LSTM + GNN

This is the enhanced version that includes all three branches:
1. Transformer: Time-frequency patterns from CWT scalograms
2. Bi-LSTM: Temporal dynamics from raw EEG sequences
3. GNN: Spatial electrode relationships from WPD features

Architecture Overview:
---------------------
                         Raw EEG (19 channels × 1000 samples)
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
              ┌─────────┐        ┌─────────┐        ┌─────────┐
              │   CWT   │        │  Direct │        │   WPD   │
              │Scalogram│        │Sequence │        │Features │
              └────┬────┘        └────┬────┘        └────┬────┘
                   │                  │                  │
                   ▼                  ▼                  ▼
              ┌─────────┐        ┌─────────┐        ┌─────────┐
              │Transformer│      │ Bi-LSTM │        │   GNN   │
              │ Encoder │        │ Encoder │        │ Encoder │
              └────┬────┘        └────┬────┘        └────┬────┘
                   │                  │                  │
                   │    (128-dim)     │    (128-dim)     │    (128-dim)
                   │                  │                  │
                   └──────────────────┼──────────────────┘
                                      │
                                      ▼
                            ┌─────────────────┐
                            │   Three-Way     │
                            │ Attention Fusion│
                            └────────┬────────┘
                                     │
                                     ▼ (128-dim)
                            ┌─────────────────┐
                            │  Classification │
                            │      Head       │
                            └────────┬────────┘
                                     │
                                     ▼
                            Depression Prediction

Improvements over V1:
--------------------
1. Added Bi-LSTM for temporal dynamics
2. Three-way fusion instead of two-way
3. Channel attention in Bi-LSTM
4. Multi-scale temporal processing option
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Import our modules
from .branches.transformer_encoder import EEGTransformerEncoder
from .branches.bilstm_encoder import EEGBiLSTMEncoder
from .branches.gnn_encoder import EEGGraphAttentionNetwork, create_eeg_graph_batch, ElectrodeGraph
from .fusion.three_way_fusion import ThreeWayAttentionFusion


@dataclass
class ModelConfigV2:
    """Configuration for the full model V2 with three branches."""

    # Transformer config (for scalograms)
    trans_d_model: int = 128
    trans_nhead: int = 4
    trans_num_layers: int = 4
    trans_dim_ff: int = 512
    trans_dropout: float = 0.1
    trans_patch_size: Tuple[int, int] = (8, 16)
    scalogram_size: Tuple[int, int] = (64, 128)

    # Bi-LSTM config (for raw EEG sequences)
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_output_dim: int = 128
    lstm_use_channel_attention: bool = True
    n_eeg_channels: int = 19

    # GNN config (for WPD features)
    gnn_node_feat_dim: int = 576
    gnn_hidden_dim: int = 128
    gnn_num_heads: int = 4
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.3
    num_electrodes: int = 19

    # Fusion config
    fusion_dim: int = 128
    fusion_num_heads: int = 4
    fusion_dropout: float = 0.1
    use_gating: bool = True

    # Classifier config
    classifier_hidden_dims: Tuple[int, ...] = (64, 32)
    classifier_dropout: Tuple[float, ...] = (0.5, 0.3)
    num_classes: int = 1

    # Memory optimization
    gradient_checkpointing: bool = True


class ClassificationHead(nn.Module):
    """Classification head for binary depression detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout_rates: Tuple[float, ...] = (0.5, 0.3),
        num_classes: int = 1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AdvancedEEGDepressionDetectorV2(nn.Module):
    """
    Complete end-to-end model V2 with three branches.

    Combines:
    - Transformer encoder for CWT scalograms
    - Bi-LSTM encoder for raw EEG temporal sequences
    - GNN encoder for WPD spatial features
    - Three-way attention fusion
    - Binary classification head
    """

    def __init__(self, config: ModelConfigV2 = None):
        """
        Initialize the full model V2.

        Args:
            config: Model configuration (uses defaults if None)
        """
        super().__init__()

        if config is None:
            config = ModelConfigV2()
        self.config = config

        # Branch 1: Transformer for CWT scalograms
        self.transformer = EEGTransformerEncoder(
            img_size=config.scalogram_size,
            patch_size=config.trans_patch_size,
            in_channels=1,
            d_model=config.trans_d_model,
            nhead=config.trans_nhead,
            num_layers=config.trans_num_layers,
            dim_ff=config.trans_dim_ff,
            dropout=config.trans_dropout,
            use_cls_token=True
        )

        # Branch 2: Bi-LSTM for raw EEG sequences
        self.bilstm = EEGBiLSTMEncoder(
            input_dim=config.n_eeg_channels,
            hidden_dim=config.lstm_hidden_dim,
            output_dim=config.lstm_output_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout
        )

        # Branch 3: GNN for WPD features
        self.gnn = EEGGraphAttentionNetwork(
            node_feat_dim=config.gnn_node_feat_dim,
            hidden_dim=config.gnn_hidden_dim,
            num_heads=config.gnn_num_heads,
            num_layers=config.gnn_num_layers,
            dropout=config.gnn_dropout,
            output_dim=config.gnn_hidden_dim,
            pooling='mean'
        )

        # Three-way attention fusion
        self.fusion = ThreeWayAttentionFusion(
            trans_dim=config.trans_d_model,
            lstm_dim=config.lstm_output_dim,
            gnn_dim=config.gnn_hidden_dim,
            fusion_dim=config.fusion_dim,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_dropout,
            use_gating=config.use_gating
        )

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=config.fusion_dim,
            hidden_dims=config.classifier_hidden_dims,
            dropout_rates=config.classifier_dropout,
            num_classes=config.num_classes
        )

        # Graph builder
        self.graph_builder = ElectrodeGraph()

    def forward(
        self,
        scalograms: torch.Tensor,
        raw_eeg: torch.Tensor,
        wpd_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all three branches and fusion.

        Args:
            scalograms: CWT scalograms (batch, H, W) or (batch, 1, H, W)
            raw_eeg: Raw EEG sequences (batch, channels, time) or (batch, time, channels)
            wpd_features: WPD features (batch, n_channels, feat_dim)
            edge_index: Graph edge indices (optional, computed if None)
            batch: Batch assignment for graph (optional)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with:
            - 'logits': Raw model output
            - 'probs': Sigmoid probabilities
            - Branch features and gate weights if return_features=True
        """
        outputs = {}
        batch_size = scalograms.shape[0]

        # Ensure scalogram has channel dimension
        if scalograms.dim() == 3:
            scalograms = scalograms.unsqueeze(1)

        # Branch 1: Transformer
        trans_features = self.transformer(scalograms)
        if return_features:
            outputs['trans_features'] = trans_features

        # Branch 2: Bi-LSTM
        lstm_features = self.bilstm(raw_eeg)
        if return_features:
            outputs['lstm_features'] = lstm_features

        # Branch 3: GNN
        if edge_index is None or batch is None:
            x, edge_index, batch = create_eeg_graph_batch(wpd_features)
            # Ensure all tensors on same device
            device = wpd_features.device
            x = x.to(device)
            edge_index = edge_index.to(device)
            batch = batch.to(device)
        else:
            x = wpd_features

        gnn_features = self.gnn(x, edge_index, batch)
        if return_features:
            outputs['gnn_features'] = gnn_features

        # Three-way fusion
        fused, gate_info = self.fusion(
            trans_features,
            lstm_features,
            gnn_features,
            return_gate_weights=return_features
        )
        if return_features:
            outputs['fused_features'] = fused
            outputs['gate_weights'] = gate_info

        # Classification
        logits = self.classifier(fused)
        outputs['logits'] = logits
        outputs['probs'] = torch.sigmoid(logits)

        return outputs

    def predict(
        self,
        scalograms: torch.Tensor,
        raw_eeg: torch.Tensor,
        wpd_features: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Make binary predictions."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(scalograms, raw_eeg, wpd_features)
            probs = outputs['probs'].squeeze(-1)
            return (probs > threshold).long()

    def get_branch_contributions(
        self,
        scalograms: torch.Tensor,
        raw_eeg: torch.Tensor,
        wpd_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get contribution of each branch to the final prediction.

        Useful for interpretability analysis.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                scalograms, raw_eeg, wpd_features,
                return_features=True
            )

            if outputs['gate_weights'] is not None:
                return {
                    'transformer': outputs['gate_weights']['trans_gate'].mean().item(),
                    'bilstm': outputs['gate_weights']['lstm_gate'].mean().item(),
                    'gnn': outputs['gate_weights']['gnn_gate'].mean().item()
                }
            else:
                return {'transformer': 0.33, 'bilstm': 0.33, 'gnn': 0.33}


def model_summary_v2(model: nn.Module) -> Dict[str, int]:
    """Get model parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Per-branch counts
    trans_params = sum(p.numel() for p in model.transformer.parameters())
    lstm_params = sum(p.numel() for p in model.bilstm.parameters())
    gnn_params = sum(p.numel() for p in model.gnn.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'transformer_parameters': trans_params,
        'bilstm_parameters': lstm_params,
        'gnn_parameters': gnn_params,
        'fusion_parameters': fusion_params,
        'classifier_parameters': classifier_params
    }


if __name__ == "__main__":
    # Test the model
    print("Testing AdvancedEEGDepressionDetectorV2...")

    config = ModelConfigV2()
    model = AdvancedEEGDepressionDetectorV2(config)

    # Test inputs
    batch_size = 4
    scalograms = torch.randn(batch_size, 1, 64, 128)
    raw_eeg = torch.randn(batch_size, 19, 1000)  # 19 channels, 1000 samples (4 sec @ 250Hz)
    wpd_features = torch.randn(batch_size, 19, 576)

    # Create graph batch
    x, edge_index, batch = create_eeg_graph_batch(wpd_features)

    # Forward pass
    outputs = model(scalograms, raw_eeg, x, edge_index, batch, return_features=True)

    print(f"\nInput shapes:")
    print(f"  Scalograms: {scalograms.shape}")
    print(f"  Raw EEG: {raw_eeg.shape}")
    print(f"  WPD features: {wpd_features.shape}")

    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Probs: {outputs['probs'].shape}")
    print(f"  Trans features: {outputs['trans_features'].shape}")
    print(f"  LSTM features: {outputs['lstm_features'].shape}")
    print(f"  GNN features: {outputs['gnn_features'].shape}")
    print(f"  Fused features: {outputs['fused_features'].shape}")

    if outputs['gate_weights']:
        print(f"\nGate weights:")
        print(f"  Transformer: {outputs['gate_weights']['trans_gate'].mean().item():.3f}")
        print(f"  Bi-LSTM: {outputs['gate_weights']['lstm_gate'].mean().item():.3f}")
        print(f"  GNN: {outputs['gate_weights']['gnn_gate'].mean().item():.3f}")

    # Model summary
    summary = model_summary_v2(model)
    print(f"\nModel Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:,}")

    print("\nModel V2 test PASSED!")
