"""
Layer-wise Relevance Propagation (LRP) for EEG Depression Detection

LRP is an explainability technique that decomposes the model's prediction
back to the input features, showing which parts of the input contributed
most to the decision.

Theory:
-------
LRP is based on the conservation principle: the total relevance at any layer
equals the relevance at the output. For a prediction f(x), we backpropagate
relevance R through each layer using specific rules.

Key LRP Rules:
--------------
1. LRP-0 (Basic): R_i = sum_j (a_i * w_ij / sum_k(a_k * w_kj)) * R_j
2. LRP-ε (Epsilon): Adds small ε to denominator for numerical stability
3. LRP-γ (Gamma): Favors positive contributions: w_ij → w_ij + γ*w_ij^+
4. LRP-αβ: Treats positive/negative contributions separately

For EEG Analysis:
-----------------
LRP reveals:
- Which electrodes contribute most to depression detection
- Which time points are most relevant
- Which frequency components (via scalogram) matter

References:
-----------
- Bach et al. (2015): "On Pixel-Wise Explanations for Non-Linear Classifier
  Decisions by Layer-Wise Relevance Propagation"
- Montavon et al. (2017): "Explaining nonlinear classification decisions
  with deep Taylor decomposition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class LRPConfig:
    """Configuration for LRP computation."""
    epsilon: float = 1e-9          # Numerical stability
    gamma: float = 0.25            # Gamma rule parameter
    rule: str = 'epsilon'          # 'epsilon', 'gamma', 'alpha_beta'
    alpha: float = 2.0             # Alpha for alpha-beta rule
    beta: float = 1.0              # Beta for alpha-beta rule
    detach_bias: bool = True       # Whether to detach bias terms


class LRPLinear(nn.Module):
    """LRP-compatible linear layer wrapper."""

    def __init__(self, layer: nn.Linear, config: LRPConfig):
        super().__init__()
        self.layer = layer
        self.config = config
        self.input = None
        self.output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x.detach().clone()
        self.output = self.layer(x)
        return self.output

    def lrp(self, relevance: torch.Tensor) -> torch.Tensor:
        """
        Backpropagate relevance through linear layer.

        Args:
            relevance: Relevance from higher layer (batch, out_features)

        Returns:
            Relevance for input (batch, in_features)
        """
        if self.input is None:
            raise RuntimeError("Forward pass must be called before LRP")

        weight = self.layer.weight  # (out_features, in_features)
        bias = self.layer.bias if self.layer.bias is not None else 0

        if self.config.rule == 'epsilon':
            return self._lrp_epsilon(relevance, weight, bias)
        elif self.config.rule == 'gamma':
            return self._lrp_gamma(relevance, weight, bias)
        elif self.config.rule == 'alpha_beta':
            return self._lrp_alpha_beta(relevance, weight, bias)
        else:
            return self._lrp_epsilon(relevance, weight, bias)

    def _lrp_epsilon(
        self,
        relevance: torch.Tensor,
        weight: torch.Tensor,
        bias: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        """LRP-ε rule: adds epsilon for numerical stability."""
        # z_j = sum_i (a_i * w_ij) + b_j
        z = self.output + self.config.epsilon * self.output.sign()

        # s_j = R_j / z_j
        s = relevance / z

        # c_ij = a_i * w_ij
        # R_i = sum_j (c_ij * s_j)
        return torch.mm(s, weight) * self.input

    def _lrp_gamma(
        self,
        relevance: torch.Tensor,
        weight: torch.Tensor,
        bias: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        """LRP-γ rule: favors positive contributions."""
        gamma = self.config.gamma

        # Modified weights: w+ = w + gamma * max(0, w)
        weight_pos = weight + gamma * F.relu(weight)

        # Recompute forward with modified weights
        z = F.linear(self.input, weight_pos)
        z = z + self.config.epsilon * z.sign()

        s = relevance / z
        return torch.mm(s, weight_pos) * self.input

    def _lrp_alpha_beta(
        self,
        relevance: torch.Tensor,
        weight: torch.Tensor,
        bias: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        """LRP-αβ rule: separate positive and negative contributions."""
        alpha = self.config.alpha
        beta = self.config.beta

        # Positive and negative parts
        weight_pos = F.relu(weight)
        weight_neg = weight - weight_pos

        input_pos = F.relu(self.input)
        input_neg = self.input - input_pos

        # z+ and z-
        z_pos = F.linear(input_pos, weight_pos) + F.linear(input_neg, weight_neg)
        z_neg = F.linear(input_pos, weight_neg) + F.linear(input_neg, weight_pos)

        z_pos = z_pos + self.config.epsilon * (z_pos >= 0).float()
        z_neg = z_neg - self.config.epsilon * (z_neg < 0).float()

        # Relevance propagation
        s_pos = relevance / z_pos
        s_neg = relevance / z_neg

        r_pos = (torch.mm(s_pos, weight_pos) * input_pos +
                 torch.mm(s_pos, weight_neg) * input_neg)
        r_neg = (torch.mm(s_neg, weight_neg) * input_pos +
                 torch.mm(s_neg, weight_pos) * input_neg)

        return alpha * r_pos - beta * r_neg


class LRPConv2d(nn.Module):
    """LRP-compatible Conv2d layer wrapper."""

    def __init__(self, layer: nn.Conv2d, config: LRPConfig):
        super().__init__()
        self.layer = layer
        self.config = config
        self.input = None
        self.output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x.detach().clone()
        self.output = self.layer(x)
        return self.output

    def lrp(self, relevance: torch.Tensor) -> torch.Tensor:
        """Backpropagate relevance through conv layer."""
        if self.input is None:
            raise RuntimeError("Forward pass must be called before LRP")

        weight = self.layer.weight
        stride = self.layer.stride
        padding = self.layer.padding

        # LRP-ε for conv
        z = self.output + self.config.epsilon * self.output.sign()
        s = relevance / z

        # Gradient-based relevance propagation
        # Use transposed convolution to propagate back
        c = F.conv_transpose2d(
            s, weight,
            stride=stride,
            padding=padding,
            output_padding=0
        )

        # Handle size mismatch
        if c.shape != self.input.shape:
            # Crop or pad to match input size
            diff_h = self.input.shape[2] - c.shape[2]
            diff_w = self.input.shape[3] - c.shape[3]
            if diff_h > 0 or diff_w > 0:
                c = F.pad(c, [0, diff_w, 0, diff_h])
            elif diff_h < 0 or diff_w < 0:
                c = c[:, :, :self.input.shape[2], :self.input.shape[3]]

        return c * self.input


class LRPAttention(nn.Module):
    """LRP for attention mechanisms (simplified)."""

    def __init__(self, config: LRPConfig):
        super().__init__()
        self.config = config
        self.attention_weights = None
        self.input = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard attention forward pass."""
        self.input = (query.detach().clone(),
                      key.detach().clone(),
                      value.detach().clone())

        # Scaled dot-product attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        self.attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(self.attention_weights, value)

        return output, self.attention_weights

    def lrp(self, relevance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backpropagate relevance through attention.

        For attention: output = softmax(QK^T/sqrt(d)) * V
        We distribute relevance based on attention weights.
        """
        if self.attention_weights is None:
            raise RuntimeError("Forward pass must be called before LRP")

        query, key, value = self.input

        # Relevance to value: proportional to attention weights
        # R_v = A^T * R_out
        r_value = torch.matmul(self.attention_weights.transpose(-2, -1), relevance)

        # Relevance to attention (simplified - uniform distribution to Q and K)
        r_query = relevance.mean(dim=-1, keepdim=True).expand_as(query)
        r_key = relevance.mean(dim=-1, keepdim=True).expand_as(key)

        return r_query, r_key, r_value


class EEGRelevanceAnalyzer:
    """
    Main class for computing LRP relevance for EEG depression detection.

    Provides methods to analyze:
    - Electrode relevance (spatial)
    - Temporal relevance (time points)
    - Frequency relevance (from scalograms)
    """

    def __init__(
        self,
        model: nn.Module,
        config: LRPConfig = None,
        device: str = 'cuda'
    ):
        """
        Initialize LRP analyzer.

        Args:
            model: The trained EEG model
            config: LRP configuration
            device: Computation device
        """
        self.model = model
        self.config = config or LRPConfig()
        self.device = device

        # Storage for activations
        self.activations = OrderedDict()
        self.hooks = []

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook

        # Register hooks for key layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_relevance(
        self,
        scalogram: torch.Tensor,
        wpd_features: torch.Tensor,
        target_class: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute LRP relevance for a single sample.

        Args:
            scalogram: CWT scalogram (1, H, W) or (1, 1, H, W)
            wpd_features: WPD features (1, n_channels, feat_dim)
            target_class: Class to explain (1=depression, 0=healthy)

        Returns:
            Dictionary with relevance maps for different components
        """
        self.model.eval()
        self._register_hooks()

        # Ensure batch dimension
        if scalogram.dim() == 3:
            scalogram = scalogram.unsqueeze(0)
        if scalogram.dim() == 3:
            scalogram = scalogram.unsqueeze(1)

        scalogram = scalogram.to(self.device)
        scalogram.requires_grad_(True)
        scalogram.retain_grad()  # Keep grad for non-leaf tensor

        wpd_features = wpd_features.to(self.device)

        # Forward pass
        from models.branches.gnn_encoder import create_eeg_graph_batch
        x, edge_index, batch = create_eeg_graph_batch(wpd_features.cpu())
        x = x.to(self.device)
        x.requires_grad_(True)
        x.retain_grad()  # Keep grad for non-leaf tensor
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)

        outputs = self.model(scalogram, x, edge_index, batch, return_features=True)
        logits = outputs['logits']

        # Start relevance from output
        if target_class == 1:
            relevance_output = torch.sigmoid(logits)
        else:
            relevance_output = 1 - torch.sigmoid(logits)

        # Compute gradients (gradient × input approximation of LRP)
        relevance_output.backward(torch.ones_like(relevance_output))

        # Extract relevance maps
        relevance = {}

        # Scalogram relevance (time-frequency)
        if scalogram.grad is not None:
            scalogram_relevance = (scalogram.grad * scalogram).squeeze()
            relevance['scalogram'] = scalogram_relevance.detach().cpu()

            # Aggregate to frequency bands
            relevance['frequency'] = scalogram_relevance.mean(dim=-1).detach().cpu()

            # Aggregate to time
            relevance['temporal'] = scalogram_relevance.mean(dim=0).detach().cpu()

        # WPD/Electrode relevance (spatial)
        if x.grad is not None:
            electrode_relevance = (x.grad * x).sum(dim=-1)  # Sum over features
            # Reshape to (batch, n_electrodes)
            n_electrodes = 19
            electrode_relevance = electrode_relevance.view(-1, n_electrodes)
            relevance['electrodes'] = electrode_relevance.detach().cpu().squeeze()

        # Feature-level relevance
        if 'trans_features' in outputs:
            relevance['transformer_features'] = outputs['trans_features'].detach().cpu()
        if 'gnn_features' in outputs:
            relevance['gnn_features'] = outputs['gnn_features'].detach().cpu()

        self._remove_hooks()
        return relevance

    def compute_electrode_importance(
        self,
        dataloader,
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute average electrode importance across multiple samples.

        Args:
            dataloader: DataLoader with samples
            n_samples: Number of samples to analyze

        Returns:
            Dictionary with electrode importance statistics
        """
        electrode_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2'
        ]

        all_relevance = []
        all_labels = []

        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break

            scalograms = batch['scalogram']
            wpd_features = batch['wpd_features']
            labels = batch['label']

            for i in range(scalograms.size(0)):
                if count >= n_samples:
                    break

                relevance = self.compute_relevance(
                    scalograms[i:i+1],
                    wpd_features[i:i+1],
                    target_class=int(labels[i])
                )

                if 'electrodes' in relevance:
                    all_relevance.append(relevance['electrodes'].numpy())
                    all_labels.append(int(labels[i]))
                    count += 1

        all_relevance = np.array(all_relevance)
        all_labels = np.array(all_labels)

        # Compute statistics
        results = {
            'electrode_names': electrode_names,
            'mean_relevance': all_relevance.mean(axis=0),
            'std_relevance': all_relevance.std(axis=0),
            'mdd_relevance': all_relevance[all_labels == 1].mean(axis=0),
            'healthy_relevance': all_relevance[all_labels == 0].mean(axis=0),
            'relevance_difference': (
                all_relevance[all_labels == 1].mean(axis=0) -
                all_relevance[all_labels == 0].mean(axis=0)
            )
        }

        return results

    def compute_frequency_importance(
        self,
        dataloader,
        n_samples: int = 100,
        freq_bands: Dict[str, Tuple[float, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute frequency band importance.

        Args:
            dataloader: DataLoader with samples
            n_samples: Number of samples to analyze
            freq_bands: Dictionary of frequency band ranges

        Returns:
            Dictionary with frequency importance
        """
        if freq_bands is None:
            freq_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }

        all_freq_relevance = []
        all_labels = []

        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break

            scalograms = batch['scalogram']
            labels = batch['label']
            wpd_features = batch['wpd_features']

            for i in range(scalograms.size(0)):
                if count >= n_samples:
                    break

                relevance = self.compute_relevance(
                    scalograms[i:i+1],
                    wpd_features[i:i+1],
                    target_class=int(labels[i])
                )

                if 'frequency' in relevance:
                    all_freq_relevance.append(relevance['frequency'].numpy())
                    all_labels.append(int(labels[i]))
                    count += 1

        all_freq_relevance = np.array(all_freq_relevance)
        all_labels = np.array(all_labels)

        # Map scalogram rows to frequency bands (approximate)
        n_freqs = all_freq_relevance.shape[1] if len(all_freq_relevance) > 0 else 64
        freq_range = np.linspace(0.5, 45, n_freqs)

        band_importance = {}
        for band_name, (f_low, f_high) in freq_bands.items():
            mask = (freq_range >= f_low) & (freq_range <= f_high)
            if mask.any():
                band_importance[band_name] = {
                    'mean': all_freq_relevance[:, mask].mean(),
                    'mdd': all_freq_relevance[all_labels == 1][:, mask].mean() if (all_labels == 1).any() else 0,
                    'healthy': all_freq_relevance[all_labels == 0][:, mask].mean() if (all_labels == 0).any() else 0
                }

        return {
            'freq_bands': freq_bands,
            'band_importance': band_importance,
            'raw_freq_relevance': all_freq_relevance.mean(axis=0) if len(all_freq_relevance) > 0 else np.array([])
        }


def create_lrp_analyzer(
    model: nn.Module,
    rule: str = 'epsilon',
    device: str = 'cuda'
) -> EEGRelevanceAnalyzer:
    """
    Factory function to create LRP analyzer.

    Args:
        model: Trained model
        rule: LRP rule ('epsilon', 'gamma', 'alpha_beta')
        device: Computation device

    Returns:
        Configured LRP analyzer
    """
    config = LRPConfig(rule=rule)
    return EEGRelevanceAnalyzer(model, config, device)


if __name__ == "__main__":
    print("LRP Module - Layer-wise Relevance Propagation for EEG Analysis")
    print("=" * 60)
    print("\nThis module provides:")
    print("  1. LRP relevance computation for neural networks")
    print("  2. Electrode importance analysis")
    print("  3. Frequency band importance analysis")
    print("  4. Temporal pattern importance")
    print("\nUse create_lrp_analyzer(model) to get started.")
