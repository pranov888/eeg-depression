"""
Graph Attention Network (GAT) Encoder for EEG Spatial Modeling

This module implements a Graph Attention Network that models the spatial
relationships between EEG electrodes. The key insight is that EEG electrodes
form a natural graph structure on the scalp, and depression is associated
with altered functional connectivity patterns.

Design Decisions (for research paper):
--------------------------------------
1. WHY GAT over GCN?
   - GAT uses attention to learn edge importance
   - Different electrode pairs may have different relevance for depression
   - Attention weights provide interpretability (which connections matter?)

2. WHY Graph structure for EEG?
   - Electrodes have spatial positions (non-Euclidean data)
   - Brain regions communicate (functional connectivity)
   - Depression affects specific pathways (e.g., frontal-limbic)

3. HYBRID ADJACENCY (spatial + functional):
   - Spatial: Based on electrode positions (k-nearest neighbors)
   - Functional: Based on signal correlation (connectivity > threshold)
   - Combining both captures anatomical and functional relationships

References:
-----------
- Veličković et al. (2018). Graph Attention Networks. ICLR.
- Wang et al. (2021). EEG-based emotion recognition using GNN.
- Zhang et al. (2023). Depression detection via EEG connectivity graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from sklearn.neighbors import kneighbors_graph


class ElectrodeGraph:
    """
    Constructs graph structure for EEG electrodes.

    The graph combines:
    1. Spatial adjacency: Electrodes close together are connected
    2. Functional connectivity: Correlated channels are connected

    This hybrid approach captures both anatomical proximity and
    functional relationships between brain regions.
    """

    # Standard 10-20 electrode positions (normalized 3D coordinates)
    # These are approximate positions on a unit sphere
    ELECTRODE_POSITIONS_10_20 = {
        'Fp1': [-0.31, 0.95, -0.03],
        'Fp2': [0.31, 0.95, -0.03],
        'F7': [-0.81, 0.59, -0.03],
        'F3': [-0.55, 0.67, 0.50],
        'Fz': [0.00, 0.71, 0.70],
        'F4': [0.55, 0.67, 0.50],
        'F8': [0.81, 0.59, -0.03],
        'T3': [-1.00, 0.00, -0.03],
        'C3': [-0.71, 0.00, 0.71],
        'Cz': [0.00, 0.00, 1.00],
        'C4': [0.71, 0.00, 0.71],
        'T4': [1.00, 0.00, -0.03],
        'T5': [-0.81, -0.59, -0.03],
        'P3': [-0.55, -0.67, 0.50],
        'Pz': [0.00, -0.71, 0.70],
        'P4': [0.55, -0.67, 0.50],
        'T6': [0.81, -0.59, -0.03],
        'O1': [-0.31, -0.95, -0.03],
        'O2': [0.31, -0.95, -0.03]
    }

    ELECTRODE_ORDER = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]

    def __init__(
        self,
        electrode_names: List[str] = None,
        spatial_k: int = 6,
        functional_threshold: float = 0.5,
        combine_method: str = 'union'
    ):
        """
        Initialize electrode graph constructor.

        Args:
            electrode_names: List of electrode names (default: 10-20 montage)
            spatial_k: Number of nearest neighbors for spatial adjacency
            functional_threshold: Correlation threshold for functional edges
            combine_method: How to combine spatial and functional ('union', 'intersection')
        """
        self.electrode_names = electrode_names or self.ELECTRODE_ORDER
        self.spatial_k = spatial_k
        self.functional_threshold = functional_threshold
        self.combine_method = combine_method

        # Get positions for specified electrodes
        self.positions = np.array([
            self.ELECTRODE_POSITIONS_10_20.get(name, [0, 0, 0])
            for name in self.electrode_names
        ])

        # Pre-compute spatial adjacency (fixed for all samples)
        self.spatial_adj = self._compute_spatial_adjacency()

    def _compute_spatial_adjacency(self) -> np.ndarray:
        """
        Compute spatial adjacency based on electrode positions.

        Uses k-nearest neighbors to connect electrodes that are
        physically close on the scalp.

        Returns:
            Adjacency matrix of shape (n_electrodes, n_electrodes)
        """
        n = len(self.electrode_names)
        k = min(self.spatial_k, n - 1)

        # K-nearest neighbors graph
        adj = kneighbors_graph(
            self.positions,
            n_neighbors=k,
            mode='connectivity',
            include_self=False
        ).toarray()

        # Make symmetric (undirected graph)
        adj = np.maximum(adj, adj.T)

        return adj

    def compute_functional_adjacency(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Compute functional adjacency based on signal correlation.

        Channels with high correlation are considered functionally connected.
        This captures brain connectivity patterns.

        Args:
            eeg_data: EEG data of shape (n_channels, n_samples)

        Returns:
            Adjacency matrix based on correlation
        """
        # Compute correlation matrix
        corr_matrix = np.corrcoef(eeg_data)

        # Threshold to get adjacency
        adj = (np.abs(corr_matrix) > self.functional_threshold).astype(float)

        # Remove self-loops
        np.fill_diagonal(adj, 0)

        return adj

    def get_edge_index(
        self,
        eeg_data: Optional[np.ndarray] = None,
        use_functional: bool = True
    ) -> torch.Tensor:
        """
        Get edge index for PyTorch Geometric.

        Args:
            eeg_data: EEG data for functional connectivity (optional)
            use_functional: Whether to include functional edges

        Returns:
            Edge index tensor of shape (2, n_edges)
        """
        adj = self.spatial_adj.copy()

        if use_functional and eeg_data is not None:
            func_adj = self.compute_functional_adjacency(eeg_data)

            if self.combine_method == 'union':
                adj = np.maximum(adj, func_adj)
            elif self.combine_method == 'intersection':
                adj = np.minimum(adj, func_adj)
            elif self.combine_method == 'functional_only':
                adj = func_adj

        # Convert to edge index format
        edge_index = np.array(np.where(adj > 0))

        return torch.tensor(edge_index, dtype=torch.long)

    def get_edge_attr(
        self,
        eeg_data: np.ndarray,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge attributes (correlation strength).

        Args:
            eeg_data: EEG data
            edge_index: Edge indices

        Returns:
            Edge attributes (correlation values)
        """
        corr_matrix = np.corrcoef(eeg_data)

        src, dst = edge_index.numpy()
        edge_attr = np.abs(corr_matrix[src, dst])

        return torch.tensor(edge_attr, dtype=torch.float).unsqueeze(-1)


class GATLayer(nn.Module):
    """
    Graph Attention Layer with residual connection and layer normalization.

    Improvements over vanilla GAT:
    1. Pre-normalization (more stable training)
    2. Residual connections (better gradient flow)
    3. Feed-forward network (more expressive)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True
    ):
        super().__init__()

        self.concat = concat
        self.in_channels = in_channels
        actual_out = out_channels * heads if concat else out_channels
        self.actual_out = actual_out

        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(actual_out)

        # GAT convolution
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(actual_out, actual_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(actual_out * 2, actual_out),
            nn.Dropout(dropout)
        )

        # Residual projection if dimensions don't match
        if in_channels != actual_out:
            self.residual = nn.Linear(in_channels, actual_out)
        else:
            self.residual = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass with residual connections.

        Args:
            x: Node features (n_nodes, in_channels)
            edge_index: Graph connectivity (2, n_edges)
            return_attention: Whether to return attention weights

        Returns:
            Updated node features
        """
        # Pre-norm + GAT + residual
        h = self.norm1(x)
        if return_attention:
            h, attn = self.gat(h, edge_index, return_attention_weights=True)
        else:
            h = self.gat(h, edge_index)
            attn = None
        h = h + self.residual(x)

        # Pre-norm + FFN + residual
        h = h + self.ffn(self.norm2(h))

        if return_attention:
            return h, attn
        return h


class EEGGraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network for EEG electrode-level feature processing.

    This network treats each electrode as a node in a graph, with edges
    representing spatial proximity and/or functional connectivity.

    Architecture:
    1. Input projection: Map WPD features to hidden dimension
    2. Multiple GAT layers: Learn inter-electrode relationships
    3. Global pooling: Aggregate node features to graph-level

    The attention mechanism learns which electrode pairs are most
    important for depression classification, providing interpretability.
    """

    def __init__(
        self,
        node_feat_dim: int = 576,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        output_dim: int = 128,
        pooling: str = 'mean'
    ):
        """
        Initialize GNN encoder.

        Args:
            node_feat_dim: Input feature dimension per node (electrode)
            hidden_dim: Hidden dimension (should match Transformer for fusion)
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout probability
            output_dim: Output embedding dimension
            pooling: Global pooling method ('mean', 'max', 'both')
        """
        super().__init__()

        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # GAT layers
        # Design: All layers maintain hidden_dim, last layer doesn't concatenate
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            # Last layer doesn't concatenate heads (to get final hidden_dim output)
            concat = (i < num_layers - 1)

            # All layers take hidden_dim as input (after input projection)
            # With concat=True: out_channels = hidden_dim // num_heads, actual = hidden_dim
            # With concat=False: out_channels = hidden_dim, actual = hidden_dim
            layer = GATLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads if concat else hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=concat
            )
            self.gat_layers.append(layer)

        # Output projection
        pool_dim = hidden_dim * 2 if pooling == 'both' else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(pool_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        # Graph builder
        self.graph_builder = ElectrodeGraph()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features of shape (total_nodes, node_feat_dim)
               For batched graphs: (batch_size * n_nodes, node_feat_dim)
            edge_index: Edge connectivity (2, n_edges)
            batch: Batch assignment for each node (total_nodes,)
            return_attention: Whether to return attention weights

        Returns:
            Graph-level embedding of shape (batch_size, output_dim)
        """
        # Input projection
        h = self.input_proj(x)

        # GAT layers
        attention_weights = []
        for layer in self.gat_layers:
            if return_attention:
                h, attn = layer(h, edge_index, return_attention=True)
                attention_weights.append(attn)
            else:
                h = layer(h, edge_index)

        # Global pooling
        if batch is None:
            # Single graph (no batching)
            if self.pooling == 'mean':
                pooled = h.mean(dim=0, keepdim=True)
            elif self.pooling == 'max':
                pooled = h.max(dim=0, keepdim=True)[0]
            else:  # both
                pooled = torch.cat([
                    h.mean(dim=0, keepdim=True),
                    h.max(dim=0, keepdim=True)[0]
                ], dim=-1)
        else:
            # Batched graphs
            if self.pooling == 'mean':
                pooled = global_mean_pool(h, batch)
            elif self.pooling == 'max':
                pooled = global_max_pool(h, batch)
            else:  # both
                pooled = torch.cat([
                    global_mean_pool(h, batch),
                    global_max_pool(h, batch)
                ], dim=-1)

        # Output projection
        output = self.output_proj(pooled)

        if return_attention:
            return output, attention_weights
        return output

    def get_attention_maps(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get attention weights from all GAT layers.

        Useful for interpretability: shows which electrode connections
        the model considers important.

        Returns:
            List of attention weight tensors for each layer
        """
        _, attention_weights = self.forward(
            x, edge_index, return_attention=True
        )
        return attention_weights


def create_eeg_graph_batch(
    features: torch.Tensor,
    eeg_signals: Optional[np.ndarray] = None,
    use_functional: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create batched graph data from EEG features.

    Args:
        features: Node features of shape (batch, n_channels, feat_dim)
        eeg_signals: Raw EEG signals for functional connectivity (batch, n_channels, n_samples)
        use_functional: Whether to compute functional connectivity

    Returns:
        x: Batched node features (batch * n_channels, feat_dim)
        edge_index: Batched edge indices
        batch: Batch assignment tensor
    """
    batch_size, n_channels, feat_dim = features.shape

    # Build graph
    graph_builder = ElectrodeGraph()

    # For simplicity, use same spatial adjacency for all samples
    # (functional connectivity would vary per sample)
    base_edge_index = graph_builder.get_edge_index(use_functional=False)

    # Batch the graphs
    all_x = []
    all_edge_index = []
    all_batch = []

    for i in range(batch_size):
        all_x.append(features[i])

        # Offset edge indices for this sample
        offset = i * n_channels
        edge_index_i = base_edge_index + offset
        all_edge_index.append(edge_index_i)

        # Batch assignment
        batch_i = torch.full((n_channels,), i, dtype=torch.long)
        all_batch.append(batch_i)

    x = torch.cat(all_x, dim=0)
    edge_index = torch.cat(all_edge_index, dim=1)
    batch = torch.cat(all_batch, dim=0)

    return x, edge_index, batch
