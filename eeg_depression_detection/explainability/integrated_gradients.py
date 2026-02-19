"""
Integrated Gradients for EEG Feature Attribution

Integrated Gradients (IG) is a principled method for attributing model predictions
to input features. It satisfies two key axioms:

1. Sensitivity: If a feature affects the prediction, it gets non-zero attribution
2. Implementation Invariance: Two models with same input-output mapping get same attributions

The method computes attributions by integrating gradients along a path from a
baseline (reference) input to the actual input.

Formula:
--------
IG_i(x) = (x_i - x'_i) × ∫₀¹ (∂F(x' + α(x-x'))/∂x_i) dα

Where:
- x is the input
- x' is the baseline
- F is the model function
- α interpolates between baseline and input

Design Decisions (for research paper):
--------------------------------------
1. Baseline Selection:
   - Zero baseline: Represents "no signal" (clinically meaningful)
   - Mean baseline: Average patient (alternative)
   - Noise baseline: Random noise (robustness)

2. Integration Steps:
   - 50 steps provides good approximation
   - More steps = more accurate but slower

3. Batch Processing:
   - Process multiple interpolation points in parallel
   - Significant speedup for GPU

References:
-----------
- Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. ICML.
- Sturmfels et al. (2020). Visualizing the Impact of Feature Attribution Baselines.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from tqdm import tqdm


class IntegratedGradients:
    """
    Integrated Gradients explainer for EEG depression model.

    Computes feature attributions that show which parts of the input
    (WPD features, scalograms) contribute most to the prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        baseline_type: str = 'zero',
        n_steps: int = 50,
        batch_size: int = 16,
        device: str = 'cuda'
    ):
        """
        Initialize Integrated Gradients explainer.

        Args:
            model: The trained model
            baseline_type: Type of baseline ('zero', 'mean', 'noise')
            n_steps: Number of integration steps
            batch_size: Batch size for parallel processing
            device: Device to use
        """
        self.model = model
        self.baseline_type = baseline_type
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = device

        self.model.eval()

    def _get_baseline(
        self,
        scalogram: torch.Tensor,
        wpd_features: torch.Tensor,
        data_mean: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create baseline inputs.

        Args:
            scalogram: Input scalogram
            wpd_features: Input WPD features
            data_mean: Optional mean values for mean baseline

        Returns:
            (baseline_scalogram, baseline_wpd)
        """
        if self.baseline_type == 'zero':
            baseline_scalo = torch.zeros_like(scalogram)
            baseline_wpd = torch.zeros_like(wpd_features)

        elif self.baseline_type == 'mean':
            if data_mean is not None:
                baseline_scalo = data_mean['scalogram'].to(scalogram.device)
                baseline_wpd = data_mean['wpd_features'].to(wpd_features.device)
            else:
                baseline_scalo = torch.zeros_like(scalogram)
                baseline_wpd = torch.zeros_like(wpd_features)

        elif self.baseline_type == 'noise':
            baseline_scalo = torch.randn_like(scalogram) * 0.01
            baseline_wpd = torch.randn_like(wpd_features) * 0.01

        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")

        return baseline_scalo, baseline_wpd

    def _interpolate(
        self,
        baseline: torch.Tensor,
        input_tensor: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Interpolate between baseline and input.

        Args:
            baseline: Baseline tensor
            input_tensor: Input tensor
            alpha: Interpolation factor (0 to 1)

        Returns:
            Interpolated tensor
        """
        return baseline + alpha * (input_tensor - baseline)

    def _compute_gradients(
        self,
        scalogram: torch.Tensor,
        wpd_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        target_class: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients of output with respect to inputs.

        Args:
            scalogram: Input scalogram
            wpd_features: Input WPD features
            edge_index: Graph edge indices
            batch: Batch assignment
            target_class: Target class for gradient computation

        Returns:
            (grad_scalogram, grad_wpd)
        """
        scalogram = scalogram.clone().requires_grad_(True)
        wpd_features = wpd_features.clone().requires_grad_(True)

        # Forward pass
        outputs = self.model(
            scalogram, wpd_features, edge_index, batch
        )

        # Get target output
        if target_class == 1:
            target = outputs['probs'].squeeze()
        else:
            target = 1 - outputs['probs'].squeeze()

        # Backward pass
        target.sum().backward()

        return scalogram.grad, wpd_features.grad

    def attribute(
        self,
        scalogram: torch.Tensor,
        wpd_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        target_class: int = 1,
        return_convergence: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Integrated Gradients attributions.

        Args:
            scalogram: Input scalogram (1, H, W) or (1, 1, H, W)
            wpd_features: Input WPD features (1, n_channels, n_features)
            edge_index: Graph edge indices
            batch: Batch assignment
            target_class: Class to explain (0 or 1)
            return_convergence: Whether to return convergence delta

        Returns:
            Dictionary with attributions for each input type
        """
        # Ensure single sample
        if scalogram.dim() == 3:
            scalogram = scalogram.unsqueeze(0)
        if scalogram.shape[0] != 1:
            raise ValueError("attribute() expects single sample")

        scalogram = scalogram.to(self.device)
        wpd_features = wpd_features.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)

        # Get baseline
        baseline_scalo, baseline_wpd = self._get_baseline(scalogram, wpd_features)

        # Compute gradients at interpolation points
        all_grads_scalo = []
        all_grads_wpd = []

        alphas = torch.linspace(0, 1, self.n_steps + 1)[:-1]  # Exclude endpoint

        for alpha in tqdm(alphas, desc="Computing IG", leave=False):
            # Interpolated inputs
            interp_scalo = self._interpolate(baseline_scalo, scalogram, alpha.item())
            interp_wpd = self._interpolate(baseline_wpd, wpd_features, alpha.item())

            # Compute gradients
            grad_scalo, grad_wpd = self._compute_gradients(
                interp_scalo, interp_wpd, edge_index, batch, target_class
            )

            all_grads_scalo.append(grad_scalo)
            all_grads_wpd.append(grad_wpd)

        # Average gradients (Riemann sum approximation of integral)
        avg_grad_scalo = torch.stack(all_grads_scalo).mean(dim=0)
        avg_grad_wpd = torch.stack(all_grads_wpd).mean(dim=0)

        # Integrated gradients = (input - baseline) * average_gradients
        ig_scalo = (scalogram - baseline_scalo) * avg_grad_scalo
        ig_wpd = (wpd_features - baseline_wpd) * avg_grad_wpd

        result = {
            'scalogram_attributions': ig_scalo.detach().cpu(),
            'wpd_attributions': ig_wpd.detach().cpu()
        }

        # Convergence check (optional)
        if return_convergence:
            # Sum of attributions should approximate F(x) - F(baseline)
            with torch.no_grad():
                pred_input = self.model(scalogram, wpd_features, edge_index, batch)['probs']
                pred_baseline = self.model(baseline_scalo, baseline_wpd, edge_index, batch)['probs']

            delta = pred_input - pred_baseline
            attr_sum = ig_scalo.sum() + ig_wpd.sum()

            result['convergence_delta'] = (delta - attr_sum).abs().item()

        return result

    def attribute_batch(
        self,
        dataloader,
        target_class: int = 1
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Compute attributions for multiple samples.

        Args:
            dataloader: DataLoader with samples to explain
            target_class: Class to explain

        Returns:
            Dictionary with lists of attributions
        """
        all_scalo_attrs = []
        all_wpd_attrs = []

        for batch_data in tqdm(dataloader, desc="Computing batch attributions"):
            for i in range(batch_data['scalogram'].shape[0]):
                scalogram = batch_data['scalogram'][i:i+1]
                wpd_features = batch_data['wpd_features'][i:i+1]

                # Create graph for single sample
                from models.branches.gnn_encoder import create_eeg_graph_batch
                _, edge_index, batch = create_eeg_graph_batch(wpd_features)

                attrs = self.attribute(
                    scalogram, wpd_features, edge_index, batch, target_class
                )

                all_scalo_attrs.append(attrs['scalogram_attributions'])
                all_wpd_attrs.append(attrs['wpd_attributions'])

        return {
            'scalogram_attributions': all_scalo_attrs,
            'wpd_attributions': all_wpd_attrs
        }


def summarize_attributions(
    attributions: Dict[str, torch.Tensor],
    electrode_names: List[str] = None,
    feature_names: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Summarize attributions for interpretation.

    Args:
        attributions: Attributions from attribute()
        electrode_names: Names of electrodes
        feature_names: Names of WPD features

    Returns:
        Summary statistics for each electrode and feature
    """
    wpd_attrs = attributions['wpd_attributions'].squeeze().numpy()

    # Sum absolute attributions per electrode
    electrode_importance = np.abs(wpd_attrs).sum(axis=1)

    # Sum absolute attributions per feature type
    feature_importance = np.abs(wpd_attrs).sum(axis=0)

    result = {
        'electrode_importance': electrode_importance,
        'feature_importance': feature_importance
    }

    # Add names if provided
    if electrode_names:
        result['electrode_ranking'] = sorted(
            zip(electrode_names, electrode_importance),
            key=lambda x: x[1],
            reverse=True
        )

    return result
