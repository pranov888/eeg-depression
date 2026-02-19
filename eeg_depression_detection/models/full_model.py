"""
Complete EEG Depression Detection Model

This module integrates all components into a unified model:
- Wavelet feature extraction (WPD + CWT)
- Transformer encoder for time-frequency analysis
- Graph Attention Network for spatial electrode relationships
- Attention-based fusion for combining modalities
- Classification head for depression detection

The model is designed for:
1. High accuracy through complementary representations
2. Memory efficiency for <8GB VRAM
3. Interpretability through attention mechanisms
4. Clinical relevance through electrode-level analysis

Architecture Overview:
---------------------
                      Input: Raw EEG (19 channels × T samples)
                                    |
                    +---------------+---------------+
                    |                               |
              WPD Extraction                  CWT Scalograms
            (576 features/channel)           (64 × 128 images)
                    |                               |
              GNN Encoder                  Transformer Encoder
           (spatial relationships)         (time-frequency patterns)
                    |                               |
                    +---------------+---------------+
                                    |
                        Attention-Based Fusion
                                    |
                        Classification Head
                                    |
                        Depression Prediction

Design Decisions (for research paper):
--------------------------------------
1. Dual-branch captures complementary information
2. Cross-attention fusion allows information exchange
3. Gating adaptively weights branch contributions
4. End-to-end training optimizes all components jointly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Import our modules
from .branches.transformer_encoder import EEGTransformerEncoder
from .branches.gnn_encoder import EEGGraphAttentionNetwork, create_eeg_graph_batch, ElectrodeGraph
from .fusion.attention_fusion import AttentionBasedFusion


@dataclass
class ModelConfig:
    """Configuration for the full model."""

    # Transformer config
    trans_d_model: int = 128
    trans_nhead: int = 4
    trans_num_layers: int = 4
    trans_dim_ff: int = 512
    trans_dropout: float = 0.1
    trans_patch_size: Tuple[int, int] = (8, 16)
    scalogram_size: Tuple[int, int] = (64, 128)

    # GNN config
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
    """
    Classification head for binary depression detection.

    Uses a multi-layer perceptron with batch normalization
    and dropout for regularization.
    """

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

        # Final output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Logits (batch, num_classes)
        """
        return self.classifier(x)


class AdvancedEEGDepressionDetector(nn.Module):
    """
    Complete end-to-end model for EEG-based depression detection.

    Combines:
    - Transformer encoder for CWT scalograms (time-frequency patterns)
    - GNN encoder for WPD features (spatial electrode relationships)
    - Attention-based fusion with gating
    - Binary classification head

    The model accepts pre-computed features (WPD + CWT) rather than
    raw EEG, as feature extraction is computationally expensive and
    can be pre-computed and cached.
    """

    def __init__(self, config: ModelConfig = None):
        """
        Initialize the full model.

        Args:
            config: Model configuration (uses defaults if None)
        """
        super().__init__()

        if config is None:
            config = ModelConfig()
        self.config = config

        # Transformer branch for CWT scalograms
        self.transformer = EEGTransformerEncoder(
            img_size=config.scalogram_size,
            patch_size=config.trans_patch_size,
            in_channels=1,  # Single averaged scalogram
            d_model=config.trans_d_model,
            nhead=config.trans_nhead,
            num_layers=config.trans_num_layers,
            dim_ff=config.trans_dim_ff,
            dropout=config.trans_dropout,
            use_cls_token=True
        )

        # GNN branch for WPD features
        self.gnn = EEGGraphAttentionNetwork(
            node_feat_dim=config.gnn_node_feat_dim,
            hidden_dim=config.gnn_hidden_dim,
            num_heads=config.gnn_num_heads,
            num_layers=config.gnn_num_layers,
            dropout=config.gnn_dropout,
            output_dim=config.gnn_hidden_dim,
            pooling='mean'
        )

        # Attention-based fusion
        self.fusion = AttentionBasedFusion(
            trans_dim=config.trans_d_model,
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

        # Graph builder for electrode connectivity
        self.graph_builder = ElectrodeGraph()

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        # Transformer checkpointing
        if hasattr(self.transformer.transformer, 'layers'):
            for layer in self.transformer.transformer.layers:
                layer.use_checkpoint = True

    def forward(
        self,
        scalograms: torch.Tensor,
        wpd_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            scalograms: CWT scalograms (batch, height, width) or (batch, 1, height, width)
            wpd_features: WPD features (batch, n_channels, feat_dim)
            edge_index: Graph edge indices (2, n_edges) - computed if None
            batch: Batch assignment for graph nodes - computed if None
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
            - 'logits': Raw model output (batch, 1)
            - 'probs': Sigmoid probabilities (batch, 1)
            - 'trans_features': Transformer output (if return_features)
            - 'gnn_features': GNN output (if return_features)
            - 'fused_features': Fused representation (if return_features)
            - 'gate_weights': Fusion gate weights (if return_features)
        """
        outputs = {}
        batch_size = scalograms.shape[0]

        # Ensure scalogram has channel dimension
        if scalograms.dim() == 3:
            scalograms = scalograms.unsqueeze(1)

        # Transformer branch: process scalograms
        trans_features = self.transformer(scalograms)
        if return_features:
            outputs['trans_features'] = trans_features

        # GNN branch: process WPD features
        # Prepare graph data
        if edge_index is None or batch is None:
            # Create graph batch internally
            x, edge_index, batch = create_eeg_graph_batch(wpd_features)
        else:
            # When edge_index is provided, wpd_features should already be
            # the flattened node features (batch_size * n_nodes, feat_dim)
            x = wpd_features

        gnn_features = self.gnn(x, edge_index, batch)
        if return_features:
            outputs['gnn_features'] = gnn_features

        # Fusion: combine both branches
        fused, gate_info = self.fusion(
            trans_features,
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
        wpd_features: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Make binary predictions.

        Args:
            scalograms: CWT scalograms
            wpd_features: WPD features
            threshold: Classification threshold

        Returns:
            Binary predictions (batch,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(scalograms, wpd_features)
            probs = outputs['probs'].squeeze(-1)
            return (probs > threshold).long()

    def get_attention_weights(
        self,
        scalograms: torch.Tensor,
        wpd_features: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Get attention weights for interpretability.

        Returns attention weights from:
        - Transformer self-attention
        - GNN graph attention
        - Cross-attention in fusion
        - Gate weights

        Useful for understanding model decisions.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                scalograms, wpd_features, return_features=True
            )

            # Get GNN attention
            x, edge_index, batch = create_eeg_graph_batch(wpd_features)
            _, gnn_attn = self.gnn(x, edge_index, batch, return_attention=True)

            return {
                'fusion_gates': outputs.get('gate_weights'),
                'gnn_attention': gnn_attn
            }


class EEGDepressionModel(nn.Module):
    """
    Simplified interface for the depression detection model.

    This is a convenience wrapper that handles feature extraction
    internally (assuming pre-computed features are passed).
    """

    def __init__(
        self,
        trans_dim: int = 128,
        gnn_dim: int = 128,
        fusion_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        config = ModelConfig(
            trans_d_model=trans_dim,
            gnn_hidden_dim=gnn_dim,
            fusion_dim=fusion_dim,
            fusion_num_heads=num_heads,
            trans_dropout=dropout,
            gnn_dropout=dropout,
            fusion_dropout=dropout
        )

        self.model = AdvancedEEGDepressionDetector(config)

    def forward(
        self,
        scalograms: torch.Tensor,
        wpd_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass returning only logits.

        Args:
            scalograms: CWT scalograms
            wpd_features: WPD features

        Returns:
            Logits (batch, 1)
        """
        outputs = self.model(scalograms, wpd_features)
        return outputs['logits']


def create_model(config_path: str = None, **kwargs) -> AdvancedEEGDepressionDetector:
    """
    Factory function to create model from config file or kwargs.

    Args:
        config_path: Path to YAML config file
        **kwargs: Override config values

    Returns:
        Initialized model
    """
    if config_path is not None:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config_dict.update(kwargs)
        config = ModelConfig(**config_dict)
    else:
        config = ModelConfig(**kwargs)

    return AdvancedEEGDepressionDetector(config)


# Model summary function
def model_summary(model: nn.Module) -> Dict[str, int]:
    """
    Get model parameter counts.

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }
