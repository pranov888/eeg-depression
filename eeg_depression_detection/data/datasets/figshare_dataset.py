"""
Figshare MDD EEG Dataset Loader (EDF Format)

Dataset: MDD Patients and Healthy Controls EEG Data
Source: https://figshare.com/articles/dataset/EEG_Data_New/4244171

Dataset Description:
-------------------
- 64 subjects total: 34 MDD patients (MDD S*), 30 healthy controls (H S*)
- 19 electrodes (standard 10-20 montage)
- Sampling rate: 256 Hz
- Recording duration: ~5 minutes per condition
- Conditions: EC (Eyes Closed), EO (Eyes Open), TASK
- Data format: .edf files

File naming convention:
- "MDD S1 EC.edf" = MDD patient, Subject 1, Eyes Closed
- "H S1 EC.edf" = Healthy control, Subject 1, Eyes Closed
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import pickle
from tqdm import tqdm
import mne

# Suppress MNE info messages
mne.set_log_level('WARNING')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.preprocessing.filters import EEGFilter
from features.wavelet.wpd_extractor import WPDFeatureExtractor
from features.wavelet.cwt_extractor import CWTScalogramGenerator


class FigshareEEGDataset(Dataset):
    """
    PyTorch Dataset for Figshare MDD EEG data (EDF format).
    """

    ELECTRODE_NAMES = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]

    def __init__(
        self,
        data_dir: str,
        condition: str = 'EC',
        epoch_length: float = 4.0,
        epoch_overlap: float = 0.5,
        target_sr: int = 250,
        bandpass: Tuple[float, float] = (1, 45),
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        precompute_features: bool = True,
        max_subjects: Optional[int] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing EDF files
            condition: 'EC' (Eyes Closed), 'EO' (Eyes Open), or 'TASK'
            epoch_length: Epoch length in seconds
            epoch_overlap: Overlap ratio between epochs (0 to 1)
            target_sr: Target sampling rate after resampling
            bandpass: Bandpass filter range (low, high) in Hz
            cache_dir: Directory for caching extracted features
            transform: Optional transform to apply
            precompute_features: Whether to pre-compute WPD/CWT features
            max_subjects: Limit number of subjects (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.condition = condition
        self.epoch_length = epoch_length
        self.epoch_overlap = epoch_overlap
        self.target_sr = target_sr
        self.bandpass = bandpass
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.transform = transform
        self.precompute_features = precompute_features
        self.max_subjects = max_subjects

        # Initialize filter
        self.filter = EEGFilter(
            sampling_rate=256,
            bandpass=bandpass,
            notch_freqs=[50, 60],
            target_rate=target_sr
        )

        # Initialize feature extractors
        self.wpd_extractor = WPDFeatureExtractor(
            wavelets=['db4', 'sym5', 'coif3'],
            level=5,
            features=['energy', 'entropy', 'log_energy', 'mean', 'std', 'skewness']
        )

        self.cwt_extractor = CWTScalogramGenerator(
            wavelet='cmor1.5-1.0',
            freq_range=(1, 45),
            num_scales=64,
            output_size=(64, 128),
            sampling_rate=target_sr
        )

        # Data storage
        self.samples = []
        self.labels = []
        self.subject_ids = []

        # Load data
        self._load_data()

    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse EDF filename to extract metadata."""
        # Pattern: "MDD S1 EC.edf" or "H S1 EC.edf"
        # Also handle files like "6921143_H S15 EO.edf"

        name = Path(filename).stem

        # Remove leading numbers/underscores
        name = re.sub(r'^\d+_', '', name)

        # Match pattern
        match = re.match(r'(MDD|H)\s+S(\d+)\s+(EC|EO|TASK)', name)
        if match:
            group = match.group(1)
            subject_num = int(match.group(2))
            condition = match.group(3)

            return {
                'group': group,
                'subject_num': subject_num,
                'condition': condition,
                'label': 1 if group == 'MDD' else 0,
                'subject_id': f"{group}_S{subject_num}"
            }
        return None

    def _load_data(self):
        """Load and preprocess all EEG data."""
        # Use v2 cache file to include raw_eeg for Bi-LSTM
        cache_file = self.cache_dir / f'figshare_{self.condition}_features_v2.pkl'

        if cache_file.exists():
            print(f"Loading cached features from {cache_file}")
            self._load_cache(cache_file)
            return

        print(f"Processing raw EEG data (condition: {self.condition})...")

        # Find all EDF files for the specified condition
        edf_files = list(self.data_dir.glob('*.edf'))
        print(f"Found {len(edf_files)} total EDF files")

        # Filter by condition and parse
        files_to_process = []
        for f in edf_files:
            meta = self._parse_filename(f.name)
            if meta and meta['condition'] == self.condition:
                files_to_process.append((f, meta))

        print(f"Found {len(files_to_process)} files for condition '{self.condition}'")

        # Limit subjects if debugging
        if self.max_subjects:
            unique_subjects = list(set([m['subject_id'] for _, m in files_to_process]))
            selected_subjects = unique_subjects[:self.max_subjects]
            files_to_process = [(f, m) for f, m in files_to_process if m['subject_id'] in selected_subjects]
            print(f"Limited to {len(files_to_process)} files ({self.max_subjects} subjects)")

        # Process each file
        for filepath, meta in tqdm(files_to_process, desc="Processing EDF files"):
            self._process_edf_file(filepath, meta)

        print(f"\nDataset summary:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Unique subjects: {len(set(self.subject_ids))}")
        print(f"  Class distribution: {self.get_class_distribution()}")

        # Cache features
        if self.precompute_features and len(self.samples) > 0:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._save_cache(cache_file)

    def _process_edf_file(self, filepath: Path, meta: Dict):
        """Process a single EDF file."""
        try:
            # Load EDF using MNE
            raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)

            # Get data and info
            original_sr = raw.info['sfreq']
            data = raw.get_data()  # (n_channels, n_samples)

            # Select channels if needed (some files may have extra channels)
            n_channels = min(data.shape[0], 19)
            data = data[:n_channels]

            # Apply filtering
            filtered_data = self.filter(data)

            # Segment into epochs
            epochs = self._segment_epochs(filtered_data)

            # Extract features for each epoch
            for epoch_idx, epoch in enumerate(epochs):
                # Check for valid data
                if np.isnan(epoch).any() or np.isinf(epoch).any():
                    continue

                # Check amplitude (reject bad epochs)
                if np.abs(epoch).max() > 500:  # μV threshold
                    continue

                if self.precompute_features:
                    try:
                        # Extract WPD features
                        wpd_features = self.wpd_extractor.extract(epoch)

                        # Extract CWT scalogram (averaged across channels)
                        scalogram = self.cwt_extractor.generate_averaged(epoch)

                        # Normalize scalogram
                        scalogram = self.cwt_extractor.normalize(scalogram, method='minmax')

                        # Normalize raw epoch for Bi-LSTM
                        raw_eeg = epoch.copy()
                        # Z-score normalization per channel
                        raw_eeg = (raw_eeg - raw_eeg.mean(axis=1, keepdims=True)) / (raw_eeg.std(axis=1, keepdims=True) + 1e-8)

                        sample = {
                            'wpd_features': wpd_features.astype(np.float32),
                            'scalogram': scalogram.astype(np.float32),
                            'raw_eeg': raw_eeg.astype(np.float32),  # For Bi-LSTM
                        }
                    except Exception as e:
                        continue
                else:
                    sample = {
                        'raw_epoch': epoch.astype(np.float32),
                    }

                self.samples.append(sample)
                self.labels.append(meta['label'])
                self.subject_ids.append(meta['subject_id'])

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    def _segment_epochs(self, data: np.ndarray) -> List[np.ndarray]:
        """Segment continuous EEG into epochs."""
        n_channels, n_samples = data.shape
        epoch_samples = int(self.epoch_length * self.target_sr)
        step_samples = int(epoch_samples * (1 - self.epoch_overlap))

        epochs = []
        start = 0

        while start + epoch_samples <= n_samples:
            epoch = data[:, start:start + epoch_samples]
            epochs.append(epoch)
            start += step_samples

        return epochs

    def _load_cache(self, cache_file: Path):
        """Load cached features."""
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        self.samples = cached['samples']
        self.labels = cached['labels']
        self.subject_ids = cached['subject_ids']
        print(f"Loaded {len(self.samples)} samples from cache")

    def _save_cache(self, cache_file: Path):
        """Save extracted features to cache."""
        cached = {
            'samples': self.samples,
            'labels': self.labels,
            'subject_ids': self.subject_ids
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached, f)
        print(f"Cached features saved to {cache_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]

        item = {
            'wpd_features': torch.tensor(sample['wpd_features'], dtype=torch.float32),
            'scalogram': torch.tensor(sample['scalogram'], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'subject_id': self.subject_ids[idx]
        }

        # Add raw_eeg if available (for Bi-LSTM branch in V2 model)
        if 'raw_eeg' in sample:
            item['raw_eeg'] = torch.tensor(sample['raw_eeg'], dtype=torch.float32)

        if self.transform:
            item = self.transform(item)

        return item

    def get_subject_indices(self, subject_id: str) -> List[int]:
        """Get all indices belonging to a specific subject."""
        return [i for i, sid in enumerate(self.subject_ids) if sid == subject_id]

    def get_unique_subjects(self) -> List[str]:
        """Get list of unique subject IDs."""
        return list(set(self.subject_ids))

    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class SubjectSplitDataset(Dataset):
    """Wrapper dataset that filters by subject IDs."""

    def __init__(self, base_dataset: FigshareEEGDataset, subject_ids: List[str]):
        self.base_dataset = base_dataset
        self.subject_ids = set(subject_ids)
        self.indices = [
            i for i, sid in enumerate(base_dataset.subject_ids)
            if sid in self.subject_ids
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.base_dataset[self.indices[idx]]


def create_dataloaders(
    dataset: FigshareEEGDataset,
    train_subjects: List[str],
    val_subjects: List[str],
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with subject-wise split."""
    train_dataset = SubjectSplitDataset(dataset, train_subjects)
    val_dataset = SubjectSplitDataset(dataset, val_subjects)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--max_subjects', type=int, default=5)
    args = parser.parse_args()

    print("Testing FigshareEEGDataset...")
    dataset = FigshareEEGDataset(
        data_dir=args.data_dir,
        condition='EC',
        max_subjects=args.max_subjects
    )

    print(f"\nDataset length: {len(dataset)}")
    print(f"Unique subjects: {dataset.get_unique_subjects()}")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample shapes:")
        print(f"  WPD features: {sample['wpd_features'].shape}")
        print(f"  Scalogram: {sample['scalogram'].shape}")
        print(f"  Label: {sample['label']}")
