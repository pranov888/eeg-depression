"""
Wavelet Packet Decomposition (WPD) Feature Extractor

Extracts energy, entropy, and statistical features from multiple wavelet families.
"""

import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from typing import List, Dict, Tuple, Optional


class WPDFeatureExtractor:
    """
    Multi-wavelet Wavelet Packet Decomposition for EEG feature extraction.

    Supports multiple wavelet families (db4, sym5, coif3) and extracts
    energy, entropy, and statistical features from terminal nodes.
    """

    def __init__(
        self,
        wavelets: List[str] = ['db4', 'sym5', 'coif3'],
        level: int = 5,
        features: List[str] = ['energy', 'entropy', 'log_energy', 'mean', 'std', 'skewness', 'kurtosis']
    ):
        """
        Initialize WPD feature extractor.

        Args:
            wavelets: List of wavelet families to use
            level: Decomposition level (level 5 gives 32 terminal nodes)
            features: List of features to extract from each node
        """
        self.wavelets = wavelets
        self.level = level
        self.features = features
        self.n_nodes = 2 ** level  # Number of terminal nodes

    def _compute_node_features(self, coeffs: np.ndarray) -> Dict[str, float]:
        """
        Compute features from wavelet packet node coefficients.

        Args:
            coeffs: Wavelet coefficients for a single node

        Returns:
            Dictionary of feature names to values
        """
        features = {}
        eps = 1e-10  # Small value to avoid log(0)

        # Energy
        if 'energy' in self.features:
            features['energy'] = np.sum(coeffs ** 2)

        # Shannon Entropy
        if 'entropy' in self.features:
            p = coeffs ** 2
            p_sum = np.sum(p) + eps
            p_norm = p / p_sum
            p_norm = np.clip(p_norm, eps, 1)
            features['entropy'] = -np.sum(p_norm * np.log(p_norm))

        # Log Energy Entropy
        if 'log_energy' in self.features:
            features['log_energy'] = np.sum(np.log(coeffs ** 2 + eps))

        # Statistical features
        if 'mean' in self.features:
            features['mean'] = np.mean(coeffs)

        if 'std' in self.features:
            features['std'] = np.std(coeffs)

        if 'skewness' in self.features:
            features['skewness'] = skew(coeffs) if len(coeffs) > 2 else 0.0

        if 'kurtosis' in self.features:
            features['kurtosis'] = kurtosis(coeffs) if len(coeffs) > 3 else 0.0

        return features

    def _extract_single_wavelet(
        self,
        signal: np.ndarray,
        wavelet: str
    ) -> np.ndarray:
        """
        Extract features using a single wavelet family.

        Args:
            signal: 1D signal array
            wavelet: Wavelet name (e.g., 'db4')

        Returns:
            Feature array of shape (n_nodes * n_features,)
        """
        # Create wavelet packet tree
        wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=self.level)

        # Get all terminal nodes at the specified level
        nodes = [node.path for node in wp.get_level(self.level, 'freq')]

        all_features = []
        for node_path in nodes:
            node = wp[node_path]
            if node.data is not None and len(node.data) > 0:
                node_features = self._compute_node_features(node.data)
                all_features.extend([node_features[f] for f in self.features])
            else:
                # Pad with zeros if node is empty
                all_features.extend([0.0] * len(self.features))

        return np.array(all_features)

    def extract_channel(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract WPD features from a single EEG channel.

        Args:
            signal: 1D signal array of shape (n_samples,)

        Returns:
            Feature array of shape (n_wavelets * n_nodes * n_features,)
        """
        all_wavelet_features = []

        for wavelet in self.wavelets:
            wavelet_features = self._extract_single_wavelet(signal, wavelet)
            all_wavelet_features.append(wavelet_features)

        return np.concatenate(all_wavelet_features)

    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract WPD features from multi-channel EEG data.

        Args:
            data: EEG data of shape (n_channels, n_samples)

        Returns:
            Feature array of shape (n_channels, n_features_per_channel)
            where n_features_per_channel = n_wavelets * n_nodes * n_features
        """
        n_channels = data.shape[0]
        features_per_channel = len(self.wavelets) * self.n_nodes * len(self.features)

        features = np.zeros((n_channels, features_per_channel))

        for ch in range(n_channels):
            features[ch] = self.extract_channel(data[ch])

        return features

    def get_feature_names(self) -> List[str]:
        """
        Get names for all extracted features.

        Returns:
            List of feature names
        """
        names = []
        for wavelet in self.wavelets:
            for node_idx in range(self.n_nodes):
                for feature in self.features:
                    names.append(f"{wavelet}_node{node_idx}_{feature}")
        return names

    @property
    def n_features_per_channel(self) -> int:
        """Number of features extracted per channel."""
        return len(self.wavelets) * self.n_nodes * len(self.features)


def extract_band_powers(
    signal: np.ndarray,
    sampling_rate: float,
    bands: Dict[str, Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    Extract band powers using wavelet decomposition.

    Args:
        signal: 1D signal array
        sampling_rate: Sampling rate in Hz
        bands: Dictionary of band names to (low, high) frequency ranges

    Returns:
        Dictionary of band names to power values
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    # Use db4 wavelet for band power estimation
    wavelet = 'db4'

    # Determine decomposition level based on sampling rate
    # Level l gives frequency bands of sr/(2^(l+1)) to sr/(2^l)
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    level = min(max_level, 7)

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Map coefficients to frequency bands
    band_powers = {}

    for band_name, (low, high) in bands.items():
        power = 0.0
        for i, c in enumerate(coeffs):
            # Approximate frequency range for this level
            if i == 0:  # Approximation coefficients
                f_low = 0
                f_high = sampling_rate / (2 ** level)
            else:
                level_idx = level - i + 1
                f_low = sampling_rate / (2 ** (level_idx + 1))
                f_high = sampling_rate / (2 ** level_idx)

            # Check if this level overlaps with the band
            if f_low < high and f_high > low:
                power += np.sum(c ** 2)

        band_powers[band_name] = power

    return band_powers
