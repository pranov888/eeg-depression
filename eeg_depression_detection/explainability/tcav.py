"""
TCAV (Testing with Concept Activation Vectors) for EEG Depression Detection

TCAV is a concept-based interpretability method that tests whether a model
uses human-understandable concepts in its predictions.

Theory:
-------
Instead of asking "which pixels/features matter?", TCAV asks:
"Does the model use concept X (e.g., alpha asymmetry) to make predictions?"

Key Steps:
1. Define concepts relevant to depression (e.g., "alpha asymmetry", "theta increase")
2. Collect positive and negative examples for each concept
3. Train a linear classifier (CAV) to separate concept from non-concept in activation space
4. Use directional derivatives to measure concept influence on predictions

For EEG Depression:
------------------
Clinically relevant concepts:
- Alpha asymmetry (left-right frontal alpha power difference)
- Theta increase (elevated theta in frontal regions)
- Beta suppression (reduced beta activity)
- Delta abnormality (excessive delta in awake state)
- Coherence reduction (reduced inter-electrode coherence)

References:
-----------
- Kim et al. (2018): "Interpretability Beyond Feature Attribution:
  Quantitative Testing with Concept Activation Vectors (TCAV)"
- Ghorbani et al. (2019): "Towards Automatic Concept-based Explanations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings


@dataclass
class Concept:
    """
    Represents a human-understandable concept for TCAV.

    Attributes:
        name: Concept name (e.g., "alpha_asymmetry")
        description: Human-readable description
        detector: Function to detect concept presence in raw data
        threshold: Threshold for concept presence
    """
    name: str
    description: str
    detector: Callable[[np.ndarray], float]
    threshold: float = 0.0


class EEGConceptLibrary:
    """
    Library of EEG-specific concepts relevant to depression detection.

    Each concept has a detector function that computes a score indicating
    concept presence in the EEG data.
    """

    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate
        self.concepts = self._build_concept_library()

    def _compute_band_power(
        self,
        signal: np.ndarray,
        freq_low: float,
        freq_high: float
    ) -> float:
        """Compute power in a frequency band using FFT."""
        from scipy import signal as sig

        if signal.ndim == 1:
            signal = signal.reshape(1, -1)

        powers = []
        for ch in range(signal.shape[0]):
            freqs, psd = sig.welch(
                signal[ch],
                fs=self.sampling_rate,
                nperseg=min(256, len(signal[ch]))
            )
            mask = (freqs >= freq_low) & (freqs <= freq_high)
            if mask.any():
                powers.append(psd[mask].mean())

        return np.mean(powers) if powers else 0.0

    def _build_concept_library(self) -> Dict[str, Concept]:
        """Build library of EEG concepts for depression."""

        concepts = {}

        # 1. Alpha Asymmetry (FAA - Frontal Alpha Asymmetry)
        # Depression is associated with greater right frontal alpha (less left activation)
        def detect_alpha_asymmetry(eeg: np.ndarray) -> float:
            """
            Compute frontal alpha asymmetry.
            Positive = right > left (depression pattern)
            """
            # Assuming standard 10-20: F3=index 3, F4=index 5
            if eeg.shape[0] < 6:
                return 0.0

            f3_alpha = self._compute_band_power(eeg[3], 8, 13)  # F3 (left)
            f4_alpha = self._compute_band_power(eeg[5], 8, 13)  # F4 (right)

            if f3_alpha + f4_alpha > 0:
                # Log ratio: ln(right) - ln(left)
                asymmetry = np.log(f4_alpha + 1e-10) - np.log(f3_alpha + 1e-10)
                return asymmetry
            return 0.0

        concepts['alpha_asymmetry'] = Concept(
            name='alpha_asymmetry',
            description='Frontal alpha asymmetry (right > left indicates depression)',
            detector=detect_alpha_asymmetry,
            threshold=0.0  # Positive asymmetry = depression pattern
        )

        # 2. Elevated Theta
        # Depression is associated with increased frontal theta
        def detect_theta_elevation(eeg: np.ndarray) -> float:
            """Compute frontal theta power relative to posterior."""
            if eeg.shape[0] < 17:
                return 0.0

            # Frontal electrodes: Fp1, Fp2, F3, Fz, F4 (indices 0,1,3,4,5)
            frontal_theta = np.mean([
                self._compute_band_power(eeg[i], 4, 8)
                for i in [0, 1, 3, 4, 5]
            ])

            # Posterior: P3, Pz, P4, O1, O2 (indices 12,13,14,15,16,17,18)
            posterior_theta = np.mean([
                self._compute_band_power(eeg[i], 4, 8)
                for i in [13, 14, 15, 17, 18]
            ])

            if posterior_theta > 0:
                return frontal_theta / (posterior_theta + 1e-10)
            return 0.0

        concepts['theta_elevation'] = Concept(
            name='theta_elevation',
            description='Elevated frontal theta relative to posterior',
            detector=detect_theta_elevation,
            threshold=1.0  # Ratio > 1 indicates elevated frontal theta
        )

        # 3. Beta Suppression
        # Some depression subtypes show reduced beta
        def detect_beta_suppression(eeg: np.ndarray) -> float:
            """Compute global beta power (lower = suppression)."""
            beta_power = np.mean([
                self._compute_band_power(eeg[i], 13, 30)
                for i in range(min(eeg.shape[0], 19))
            ])
            return -beta_power  # Negative so higher score = more suppression

        concepts['beta_suppression'] = Concept(
            name='beta_suppression',
            description='Reduced beta activity across scalp',
            detector=detect_beta_suppression,
            threshold=0.0
        )

        # 4. Delta Abnormality
        # Excessive delta in awake state may indicate dysfunction
        def detect_delta_abnormality(eeg: np.ndarray) -> float:
            """Compute excessive delta relative to total power."""
            delta_power = np.mean([
                self._compute_band_power(eeg[i], 0.5, 4)
                for i in range(min(eeg.shape[0], 19))
            ])
            total_power = np.mean([
                self._compute_band_power(eeg[i], 0.5, 45)
                for i in range(min(eeg.shape[0], 19))
            ])
            if total_power > 0:
                return delta_power / total_power
            return 0.0

        concepts['delta_abnormality'] = Concept(
            name='delta_abnormality',
            description='Excessive delta power relative to total',
            detector=detect_delta_abnormality,
            threshold=0.3  # > 30% delta is abnormal in awake adults
        )

        # 5. Alpha Power Reduction
        # Reduced alpha is common in depression
        def detect_alpha_reduction(eeg: np.ndarray) -> float:
            """Compute global alpha power (lower = reduction)."""
            alpha_power = np.mean([
                self._compute_band_power(eeg[i], 8, 13)
                for i in range(min(eeg.shape[0], 19))
            ])
            return -alpha_power  # Negative so higher score = more reduction

        concepts['alpha_reduction'] = Concept(
            name='alpha_reduction',
            description='Reduced global alpha power',
            detector=detect_alpha_reduction,
            threshold=0.0
        )

        # 6. Interhemispheric Coherence
        # Depression may show reduced coherence
        def detect_coherence_reduction(eeg: np.ndarray) -> float:
            """Compute interhemispheric coherence."""
            from scipy import signal as sig

            if eeg.shape[0] < 19:
                return 0.0

            # Pairs: (F3,F4), (C3,C4), (P3,P4), (O1,O2)
            pairs = [(3, 5), (8, 10), (13, 15), (17, 18)]
            coherences = []

            for left_idx, right_idx in pairs:
                try:
                    freqs, coh = sig.coherence(
                        eeg[left_idx], eeg[right_idx],
                        fs=self.sampling_rate,
                        nperseg=min(256, eeg.shape[1])
                    )
                    # Alpha band coherence
                    mask = (freqs >= 8) & (freqs <= 13)
                    if mask.any():
                        coherences.append(coh[mask].mean())
                except:
                    pass

            if coherences:
                return -np.mean(coherences)  # Negative so higher = reduced
            return 0.0

        concepts['coherence_reduction'] = Concept(
            name='coherence_reduction',
            description='Reduced interhemispheric alpha coherence',
            detector=detect_coherence_reduction,
            threshold=0.0
        )

        return concepts

    def get_concept(self, name: str) -> Optional[Concept]:
        """Get concept by name."""
        return self.concepts.get(name)

    def list_concepts(self) -> List[str]:
        """List all available concepts."""
        return list(self.concepts.keys())

    def compute_concept_scores(
        self,
        eeg_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all concept scores for given EEG data.

        Args:
            eeg_data: EEG array (n_channels, n_samples)

        Returns:
            Dictionary of concept scores
        """
        scores = {}
        for name, concept in self.concepts.items():
            try:
                scores[name] = concept.detector(eeg_data)
            except Exception as e:
                scores[name] = 0.0
                warnings.warn(f"Error computing {name}: {e}")
        return scores


class CAV:
    """
    Concept Activation Vector.

    A linear classifier that separates concept from non-concept examples
    in the model's activation space.
    """

    def __init__(self, concept_name: str):
        self.concept_name = concept_name
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.cav_vector = None
        self.accuracy = None
        self.is_fitted = False

    def fit(
        self,
        concept_activations: np.ndarray,
        non_concept_activations: np.ndarray
    ) -> float:
        """
        Train CAV to separate concept from non-concept.

        Args:
            concept_activations: Activations for concept examples (n, d)
            non_concept_activations: Activations for non-concept examples (n, d)

        Returns:
            Classification accuracy
        """
        # Prepare data
        X = np.vstack([concept_activations, non_concept_activations])
        y = np.array([1] * len(concept_activations) + [0] * len(non_concept_activations))

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit classifier
        self.classifier.fit(X_train, y_train)
        self.accuracy = self.classifier.score(X_test, y_test)

        # Extract CAV (normal vector to decision boundary)
        self.cav_vector = self.classifier.coef_[0]
        self.cav_vector = self.cav_vector / (np.linalg.norm(self.cav_vector) + 1e-10)

        self.is_fitted = True
        return self.accuracy

    def get_direction(self) -> np.ndarray:
        """Get the CAV direction vector."""
        if not self.is_fitted:
            raise RuntimeError("CAV must be fitted first")
        return self.cav_vector


class TCAV:
    """
    Testing with Concept Activation Vectors.

    Main class for running TCAV analysis on EEG models.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_name: str,
        concept_library: EEGConceptLibrary = None,
        device: str = 'cuda'
    ):
        """
        Initialize TCAV analyzer.

        Args:
            model: The trained model
            layer_name: Name of layer to extract activations from
            concept_library: Library of EEG concepts
            device: Computation device
        """
        self.model = model
        self.layer_name = layer_name
        self.concept_library = concept_library or EEGConceptLibrary()
        self.device = device

        self.cavs: Dict[str, CAV] = {}
        self.activations = None
        self.hook_handle = None
        self._keep_grad = False  # Whether to keep gradients

    def _register_hook(self, keep_grad: bool = False):
        """Register forward hook to capture activations."""
        self._keep_grad = keep_grad

        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            if self._keep_grad:
                # Keep in computation graph for gradient computation
                self.activations = act
                if act.requires_grad:
                    act.retain_grad()
            else:
                # Detach for simple activation extraction
                self.activations = act.detach()

        # Find the target layer
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.hook_handle = module.register_forward_hook(hook)
                return

        raise ValueError(f"Layer {self.layer_name} not found in model")

    def _remove_hook(self):
        """Remove the forward hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

    def get_activations(
        self,
        scalogram: torch.Tensor,
        wpd_features: torch.Tensor
    ) -> np.ndarray:
        """
        Get activations for a single input.

        Args:
            scalogram: CWT scalogram
            wpd_features: WPD features

        Returns:
            Flattened activation array
        """
        self.model.eval()
        self._register_hook()

        # Ensure dimensions
        if scalogram.dim() == 3:
            scalogram = scalogram.unsqueeze(0)
        if scalogram.dim() == 3:
            scalogram = scalogram.unsqueeze(1)

        scalogram = scalogram.to(self.device)
        wpd_features = wpd_features.to(self.device)

        # Forward pass
        from models.branches.gnn_encoder import create_eeg_graph_batch
        x, edge_index, batch = create_eeg_graph_batch(wpd_features.cpu())
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)

        with torch.no_grad():
            _ = self.model(scalogram, x, edge_index, batch)

        activations = self.activations.cpu().numpy().flatten()
        self._remove_hook()

        return activations

    def collect_concept_examples(
        self,
        dataloader,
        concept_name: str,
        n_examples: int = 50
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Collect positive and negative examples for a concept.

        Args:
            dataloader: DataLoader with EEG data
            concept_name: Name of concept to collect
            n_examples: Number of examples per class

        Returns:
            Tuple of (concept_activations, non_concept_activations)
        """
        concept = self.concept_library.get_concept(concept_name)
        if concept is None:
            raise ValueError(f"Unknown concept: {concept_name}")

        concept_acts = []
        non_concept_acts = []

        for batch in dataloader:
            if len(concept_acts) >= n_examples and len(non_concept_acts) >= n_examples:
                break

            scalograms = batch['scalogram']
            wpd_features = batch['wpd_features']

            # Need raw EEG to compute concepts
            if 'raw_eeg' in batch:
                raw_eegs = batch['raw_eeg'].numpy()
            else:
                # Use WPD features as proxy (less accurate)
                raw_eegs = wpd_features.numpy()

            for i in range(scalograms.size(0)):
                if len(concept_acts) >= n_examples and len(non_concept_acts) >= n_examples:
                    break

                # Compute concept score
                try:
                    score = concept.detector(raw_eegs[i])
                except:
                    continue

                # Get activations
                activations = self.get_activations(
                    scalograms[i:i+1],
                    wpd_features[i:i+1]
                )

                # Classify as concept or non-concept
                if score > concept.threshold and len(concept_acts) < n_examples:
                    concept_acts.append(activations)
                elif score <= concept.threshold and len(non_concept_acts) < n_examples:
                    non_concept_acts.append(activations)

        return concept_acts, non_concept_acts

    def train_cav(
        self,
        dataloader,
        concept_name: str,
        n_examples: int = 50
    ) -> float:
        """
        Train a CAV for a specific concept.

        Args:
            dataloader: DataLoader with EEG data
            concept_name: Name of concept
            n_examples: Number of examples per class

        Returns:
            CAV classification accuracy
        """
        print(f"  Collecting examples for '{concept_name}'...")
        concept_acts, non_concept_acts = self.collect_concept_examples(
            dataloader, concept_name, n_examples
        )

        if len(concept_acts) < 10 or len(non_concept_acts) < 10:
            warnings.warn(
                f"Insufficient examples for {concept_name}: "
                f"{len(concept_acts)} concept, {len(non_concept_acts)} non-concept"
            )
            return 0.0

        print(f"  Training CAV ({len(concept_acts)} concept, {len(non_concept_acts)} non-concept)...")

        # Train CAV
        cav = CAV(concept_name)
        accuracy = cav.fit(
            np.array(concept_acts),
            np.array(non_concept_acts)
        )

        self.cavs[concept_name] = cav
        print(f"  CAV accuracy: {accuracy:.3f}")

        return accuracy

    def compute_tcav_score(
        self,
        dataloader,
        concept_name: str,
        target_class: int = 1,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Compute TCAV score for a concept.

        TCAV score = fraction of inputs where moving toward concept
                     increases prediction for target class.

        Args:
            dataloader: DataLoader with test samples
            concept_name: Concept to test
            target_class: Target class (1=depression)
            n_samples: Number of samples to test

        Returns:
            Dictionary with TCAV score and statistics
        """
        if concept_name not in self.cavs:
            raise ValueError(f"CAV not trained for {concept_name}")

        cav = self.cavs[concept_name]
        cav_direction = cav.get_direction()

        positive_count = 0
        total_count = 0
        sensitivities = []

        self.model.eval()
        self._register_hook(keep_grad=True)  # Keep gradients for TCAV

        for batch in dataloader:
            if total_count >= n_samples:
                break

            scalograms = batch['scalogram']
            wpd_features = batch['wpd_features']
            labels = batch['label']

            for i in range(scalograms.size(0)):
                if total_count >= n_samples:
                    break

                # Only test samples from target class
                if int(labels[i]) != target_class:
                    continue

                # Ensure dimensions
                scal = scalograms[i:i+1]
                if scal.dim() == 3:
                    scal = scal.unsqueeze(1)
                scal = scal.to(self.device)
                scal.requires_grad_(True)
                wpd = wpd_features[i:i+1].to(self.device)

                # Forward pass - need gradients enabled
                from models.branches.gnn_encoder import create_eeg_graph_batch
                x, edge_index, batch_assign = create_eeg_graph_batch(wpd.cpu())
                x = x.to(self.device)
                x.requires_grad_(True)
                edge_index = edge_index.to(self.device)
                batch_assign = batch_assign.to(self.device)

                # Enable gradients for forward pass
                with torch.enable_grad():
                    outputs = self.model(scal, x, edge_index, batch_assign)
                    logits = outputs['logits']

                # Compute gradient of output w.r.t. activations
                # This requires the activation to be part of computation graph
                if self.activations is not None:
                    activations = self.activations.view(-1)

                    # Directional derivative: grad(output) · cav_direction
                    grad_output = torch.autograd.grad(
                        logits.sum(),
                        self.activations,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True
                    )[0]

                    if grad_output is not None:
                        grad_flat = grad_output.view(-1).cpu().numpy()

                        # Ensure dimensions match
                        min_len = min(len(grad_flat), len(cav_direction))
                        sensitivity = np.dot(grad_flat[:min_len], cav_direction[:min_len])

                        sensitivities.append(sensitivity)
                        if sensitivity > 0:
                            positive_count += 1
                        total_count += 1

        self._remove_hook()

        if total_count == 0:
            return {
                'tcav_score': 0.0,
                'n_samples': 0,
                'cav_accuracy': cav.accuracy,
                'error': 'No valid samples'
            }

        tcav_score = positive_count / total_count

        # Statistical significance test
        # Null hypothesis: TCAV score = 0.5 (random)
        sensitivities = np.array(sensitivities)
        if len(sensitivities) > 1:
            t_stat, p_value = stats.ttest_1samp(sensitivities, 0)
        else:
            t_stat, p_value = 0, 1.0

        return {
            'tcav_score': tcav_score,
            'n_samples': total_count,
            'positive_count': positive_count,
            'cav_accuracy': cav.accuracy,
            'mean_sensitivity': float(sensitivities.mean()) if len(sensitivities) > 0 else 0,
            'std_sensitivity': float(sensitivities.std()) if len(sensitivities) > 0 else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }

    def run_full_analysis(
        self,
        dataloader,
        concepts: List[str] = None,
        n_cav_examples: int = 50,
        n_tcav_samples: int = 100
    ) -> Dict[str, Dict]:
        """
        Run complete TCAV analysis for multiple concepts.

        Args:
            dataloader: DataLoader with EEG data
            concepts: List of concept names (None = all)
            n_cav_examples: Examples for CAV training
            n_tcav_samples: Samples for TCAV scoring

        Returns:
            Dictionary with results for each concept
        """
        if concepts is None:
            concepts = self.concept_library.list_concepts()

        results = {}

        print(f"\nRunning TCAV Analysis")
        print(f"=" * 50)
        print(f"Concepts to analyze: {concepts}")
        print(f"CAV examples: {n_cav_examples}, TCAV samples: {n_tcav_samples}")
        print()

        for concept_name in concepts:
            print(f"\nAnalyzing concept: {concept_name}")
            print("-" * 40)

            try:
                # Train CAV
                cav_accuracy = self.train_cav(dataloader, concept_name, n_cav_examples)

                if cav_accuracy < 0.6:
                    print(f"  Warning: Low CAV accuracy ({cav_accuracy:.3f})")
                    print(f"  Concept may not be linearly separable in activation space")

                # Compute TCAV scores for both classes
                print(f"  Computing TCAV score for MDD class...")
                tcav_mdd = self.compute_tcav_score(
                    dataloader, concept_name, target_class=1, n_samples=n_tcav_samples
                )

                print(f"  Computing TCAV score for Healthy class...")
                tcav_healthy = self.compute_tcav_score(
                    dataloader, concept_name, target_class=0, n_samples=n_tcav_samples
                )

                results[concept_name] = {
                    'cav_accuracy': cav_accuracy,
                    'mdd': tcav_mdd,
                    'healthy': tcav_healthy,
                    'description': self.concept_library.get_concept(concept_name).description
                }

                print(f"\n  Results for '{concept_name}':")
                print(f"    CAV Accuracy: {cav_accuracy:.3f}")
                print(f"    TCAV (MDD): {tcav_mdd['tcav_score']:.3f} (p={tcav_mdd['p_value']:.4f})")
                print(f"    TCAV (Healthy): {tcav_healthy['tcav_score']:.3f} (p={tcav_healthy['p_value']:.4f})")

            except Exception as e:
                print(f"  Error analyzing {concept_name}: {e}")
                results[concept_name] = {'error': str(e)}

        return results


def create_tcav_analyzer(
    model: nn.Module,
    layer_name: str = 'fusion',
    device: str = 'cuda'
) -> TCAV:
    """
    Factory function to create TCAV analyzer.

    Args:
        model: Trained model
        layer_name: Layer to analyze
        device: Computation device

    Returns:
        Configured TCAV analyzer
    """
    return TCAV(model, layer_name, device=device)


if __name__ == "__main__":
    print("TCAV Module - Testing with Concept Activation Vectors")
    print("=" * 60)
    print("\nAvailable EEG Concepts:")

    library = EEGConceptLibrary()
    for name in library.list_concepts():
        concept = library.get_concept(name)
        print(f"  - {name}: {concept.description}")

    print("\nUse create_tcav_analyzer(model, layer_name) to get started.")
