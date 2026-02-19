"""
Continuous Wavelet Transform (CWT) Scalogram Generator

Generates time-frequency scalogram images for deep learning models.
"""

import numpy as np
import pywt
from scipy import signal as scipy_signal
from scipy.ndimage import zoom
from typing import Tuple, Optional, Union


class CWTScalogramGenerator:
    """
    Generate time-frequency scalograms using Continuous Wavelet Transform.

    These scalogram images can be used as input to CNNs or Transformers.
    """

    def __init__(
        self,
        wavelet: str = 'cmor1.5-1.0',
        freq_range: Tuple[float, float] = (1, 45),
        num_scales: int = 64,
        output_size: Tuple[int, int] = (64, 128),
        sampling_rate: float = 250
    ):
        """
        Initialize CWT scalogram generator.

        Args:
            wavelet: Mother wavelet name (default: Complex Morlet)
            freq_range: (min_freq, max_freq) in Hz
            num_scales: Number of frequency scales
            output_size: (height, width) of output scalogram
            sampling_rate: Sampling rate of input signal (Hz)
        """
        self.wavelet = wavelet
        self.freq_range = freq_range
        self.num_scales = num_scales
        self.output_size = output_size
        self.sampling_rate = sampling_rate

        # Pre-compute scales for the frequency range
        self._compute_scales()

    def _compute_scales(self):
        """Compute wavelet scales corresponding to desired frequencies."""
        # Get wavelet object to determine center frequency
        try:
            wavelet = pywt.ContinuousWavelet(self.wavelet)
            # Center frequency
            fc = pywt.central_frequency(self.wavelet)
        except:
            # Fallback for wavelets without center_frequency
            fc = 1.0

        # Compute scales for logarithmically spaced frequencies
        freqs = np.logspace(
            np.log10(self.freq_range[0]),
            np.log10(self.freq_range[1]),
            self.num_scales
        )[::-1]  # Reverse for low to high frequency (top to bottom in scalogram)

        # Scale = fc * sampling_rate / frequency
        self.scales = fc * self.sampling_rate / freqs
        self.frequencies = freqs

    def _resize_scalogram(
        self,
        scalogram: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize scalogram to target size using interpolation.

        Args:
            scalogram: Input scalogram of shape (n_scales, n_times)
            target_size: (height, width) target size

        Returns:
            Resized scalogram
        """
        if scalogram.shape == target_size:
            return scalogram

        zoom_factors = (
            target_size[0] / scalogram.shape[0],
            target_size[1] / scalogram.shape[1]
        )

        return zoom(scalogram, zoom_factors, order=1)

    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate scalogram from a single-channel signal.

        Args:
            signal: 1D signal array of shape (n_samples,)

        Returns:
            Scalogram of shape (output_size[0], output_size[1])
        """
        # Compute CWT
        coefficients, _ = pywt.cwt(
            signal,
            self.scales,
            self.wavelet,
            sampling_period=1.0/self.sampling_rate
        )

        # Take magnitude (for complex wavelets)
        if np.iscomplexobj(coefficients):
            scalogram = np.abs(coefficients)
        else:
            scalogram = coefficients

        # Resize to output size
        scalogram = self._resize_scalogram(scalogram, self.output_size)

        return scalogram

    def generate_multichannel(self, data: np.ndarray) -> np.ndarray:
        """
        Generate scalograms for multi-channel EEG data.

        Args:
            data: EEG data of shape (n_channels, n_samples)

        Returns:
            Scalograms of shape (n_channels, output_size[0], output_size[1])
        """
        n_channels = data.shape[0]
        scalograms = np.zeros((n_channels, *self.output_size))

        for ch in range(n_channels):
            scalograms[ch] = self.generate(data[ch])

        return scalograms

    def generate_averaged(self, data: np.ndarray) -> np.ndarray:
        """
        Generate averaged scalogram across all channels.

        Args:
            data: EEG data of shape (n_channels, n_samples)

        Returns:
            Averaged scalogram of shape (output_size[0], output_size[1])
        """
        scalograms = self.generate_multichannel(data)
        return np.mean(scalograms, axis=0)

    def normalize(
        self,
        scalogram: np.ndarray,
        method: str = 'minmax'
    ) -> np.ndarray:
        """
        Normalize scalogram values.

        Args:
            scalogram: Input scalogram
            method: 'minmax', 'zscore', or 'log'

        Returns:
            Normalized scalogram
        """
        if method == 'minmax':
            min_val = scalogram.min()
            max_val = scalogram.max()
            if max_val - min_val > 0:
                return (scalogram - min_val) / (max_val - min_val)
            return scalogram - min_val

        elif method == 'zscore':
            mean = scalogram.mean()
            std = scalogram.std()
            if std > 0:
                return (scalogram - mean) / std
            return scalogram - mean

        elif method == 'log':
            return np.log1p(np.abs(scalogram))

        else:
            raise ValueError(f"Unknown normalization method: {method}")


class MultiResolutionCWT:
    """
    Generate multi-resolution scalograms at different time scales.

    Useful for capturing both fine-grained and coarse temporal patterns.
    """

    def __init__(
        self,
        wavelet: str = 'cmor1.5-1.0',
        freq_range: Tuple[float, float] = (1, 45),
        resolutions: Tuple[int, ...] = (32, 64, 128),
        sampling_rate: float = 250
    ):
        """
        Initialize multi-resolution CWT generator.

        Args:
            wavelet: Mother wavelet name
            freq_range: (min_freq, max_freq) in Hz
            resolutions: Tuple of time resolutions (number of time bins)
            sampling_rate: Sampling rate (Hz)
        """
        self.generators = []
        for resolution in resolutions:
            gen = CWTScalogramGenerator(
                wavelet=wavelet,
                freq_range=freq_range,
                num_scales=64,
                output_size=(64, resolution),
                sampling_rate=sampling_rate
            )
            self.generators.append(gen)

        self.resolutions = resolutions

    def generate(self, signal: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Generate multi-resolution scalograms.

        Args:
            signal: 1D signal array

        Returns:
            Tuple of scalograms at different resolutions
        """
        return tuple(gen.generate(signal) for gen in self.generators)


def compute_cwt_features(
    scalogram: np.ndarray,
    n_freq_bands: int = 5
) -> np.ndarray:
    """
    Extract statistical features from scalogram.

    Args:
        scalogram: Scalogram of shape (n_freq, n_time)
        n_freq_bands: Number of frequency bands to divide scalogram into

    Returns:
        Feature vector
    """
    features = []

    # Global features
    features.extend([
        np.mean(scalogram),
        np.std(scalogram),
        np.max(scalogram),
        np.sum(scalogram)
    ])

    # Band-wise features
    band_size = scalogram.shape[0] // n_freq_bands
    for i in range(n_freq_bands):
        start = i * band_size
        end = (i + 1) * band_size if i < n_freq_bands - 1 else scalogram.shape[0]
        band = scalogram[start:end, :]

        features.extend([
            np.mean(band),
            np.std(band),
            np.max(band)
        ])

    # Temporal features (energy over time)
    temporal_energy = np.sum(scalogram, axis=0)
    features.extend([
        np.mean(temporal_energy),
        np.std(temporal_energy),
        np.max(temporal_energy) / (np.mean(temporal_energy) + 1e-10)  # Peak ratio
    ])

    return np.array(features)
