#!/usr/bin/env python3
"""Preprocess and cache all EEG data."""

import sys
sys.path.insert(0, '.')

from data.datasets.figshare_dataset import FigshareEEGDataset

print("Starting preprocessing...")
print("This will take ~2 hours (58 files × ~2 min each)")
print("-" * 50)

dataset = FigshareEEGDataset(
    data_dir='data/raw/figshare',
    condition='EC',
    precompute_features=True,
    cache_dir='data/raw/figshare/cache'
)

print("-" * 50)
print(f"Done! {len(dataset)} samples cached")
print(f"Subjects: {len(dataset.get_unique_subjects())}")
print(f"Classes: {dataset.get_class_distribution()}")
print("-" * 50)
print("Now run: python scripts/train.py --data_dir data/raw/figshare --output_dir outputs --epochs 100 --batch_size 16 --cv_type loso --mixed_precision")
