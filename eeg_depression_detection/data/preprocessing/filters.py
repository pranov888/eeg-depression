"""
EEG Signal Filtering Module
- Bandpass filtering
- Notch filtering
- Resampling
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Optional


def bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    sampling_rate: float,
    order: int = 5
) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.

    Args:
        data: EEG data of shape (n_channels, n_samples) or (n_samples,)
        low_freq: Lower cutoff frequency (Hz)
        high_freq: Upper cutoff frequency (Hz)
        sampling_rate: Sampling rate (Hz)
        order: Filter order

    Returns:
        Filtered data with same shape as input
    """
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Ensure frequencies are valid
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))

    b, a = signal.butter(order, [low, high], btype='band')

    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, channel) for channel in data])


def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    sampling_rate: float,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove power line noise.

    Args:
        data: EEG data of shape (n_channels, n_samples) or (n_samples,)
        notch_freq: Frequency to remove (Hz), typically 50 or 60
        sampling_rate: Sampling rate (Hz)
        quality_factor: Quality factor (higher = narrower notch)

    Returns:
        Filtered data with same shape as input
    """
    nyquist = sampling_rate / 2
    freq = notch_freq / nyquist

    if freq >= 1.0:
        return data  # Notch freq above Nyquist, skip

    b, a = signal.iirnotch(freq, quality_factor)

    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, channel) for channel in data])


def notch_filter_harmonics(
    data: np.ndarray,
    base_freq: float,
    sampling_rate: float,
    n_harmonics: int = 3,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter at base frequency and its harmonics.

    Args:
        data: EEG data
        base_freq: Base frequency (e.g., 50 or 60 Hz)
        sampling_rate: Sampling rate (Hz)
        n_harmonics: Number of harmonics to filter
        quality_factor: Quality factor for notch filters

    Returns:
        Filtered data
    """
    filtered = data.copy()
    nyquist = sampling_rate / 2

    for i in range(1, n_harmonics + 1):
        freq = base_freq * i
        if freq < nyquist:
            filtered = notch_filter(filtered, freq, sampling_rate, quality_factor)

    return filtered


def resample(
    data: np.ndarray,
    original_rate: float,
    target_rate: float
) -> np.ndarray:
    """
    Resample EEG data to target sampling rate.

    Args:
        data: EEG data of shape (n_channels, n_samples)
        original_rate: Original sampling rate (Hz)
        target_rate: Target sampling rate (Hz)

    Returns:
        Resampled data
    """
    if original_rate == target_rate:
        return data

    ratio = target_rate / original_rate
    n_samples_new = int(data.shape[-1] * ratio)

    if data.ndim == 1:
        return signal.resample(data, n_samples_new)
    else:
        return np.array([signal.resample(channel, n_samples_new) for channel in data])


class EEGFilter:
    """
    Comprehensive EEG filtering pipeline.
    """

    def __init__(
        self,
        sampling_rate: float,
        bandpass: Tuple[float, float] = (1, 45),
        notch_freqs: List[float] = [50, 60],
        target_rate: Optional[float] = None
    ):
        """
        Initialize filter pipeline.

        Args:
            sampling_rate: Original sampling rate (Hz)
            bandpass: (low, high) cutoff frequencies
            notch_freqs: List of notch frequencies to remove
            target_rate: Target sampling rate for resampling (None = no resampling)
        """
        self.sampling_rate = sampling_rate
        self.bandpass = bandpass
        self.notch_freqs = notch_freqs
        self.target_rate = target_rate

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply full filtering pipeline."""
        filtered = data.copy()

        # Bandpass filter
        if self.bandpass is not None:
            filtered = bandpass_filter(
                filtered,
                self.bandpass[0],
                self.bandpass[1],
                self.sampling_rate
            )

        # Notch filters
        for freq in self.notch_freqs:
            filtered = notch_filter_harmonics(
                filtered, freq, self.sampling_rate
            )

        # Resample if needed
        if self.target_rate is not None and self.target_rate != self.sampling_rate:
            filtered = resample(filtered, self.sampling_rate, self.target_rate)

        return filtered
