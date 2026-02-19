#!/usr/bin/env python3
"""
Quick pipeline test with minimal data.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from data.datasets.figshare_dataset import FigshareEEGDataset, create_dataloaders
from models.full_model import AdvancedEEGDepressionDetector, ModelConfig
from models.branches.gnn_encoder import create_eeg_graph_batch
from training.trainer import Trainer

def main():
    print("=" * 60)
    print("Quick Pipeline Test (3 subjects, 3 epochs)")
    print("=" * 60)

    # Load dataset with just 3 subjects
    print("\n1. Loading dataset (3 subjects)...")
    dataset = FigshareEEGDataset(
        data_dir="data/raw/figshare",
        condition='EC',
        max_subjects=3
    )

    print(f"   Samples: {len(dataset)}")
    print(f"   Subjects: {dataset.get_unique_subjects()}")
    print(f"   Classes: {dataset.get_class_distribution()}")

    # Split subjects
    subjects = dataset.get_unique_subjects()
    train_subjects = subjects[:2]
    val_subjects = subjects[2:]

    print(f"\n2. Creating dataloaders...")
    print(f"   Train subjects: {train_subjects}")
    print(f"   Val subjects: {val_subjects}")

    train_loader, val_loader = create_dataloaders(
        dataset, train_subjects, val_subjects,
        batch_size=8, num_workers=0  # Use 0 workers for debugging
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Test one batch
    print("\n3. Testing single batch forward pass...")
    batch = next(iter(train_loader))
    scalograms = batch['scalogram'].cuda()
    wpd_features = batch['wpd_features']
    labels = batch['label'].cuda()

    x, edge_index, batch_assign = create_eeg_graph_batch(wpd_features)
    x = x.cuda()
    edge_index = edge_index.cuda()
    batch_assign = batch_assign.cuda()

    config = ModelConfig()
    model = AdvancedEEGDepressionDetector(config).cuda()

    with torch.no_grad():
        outputs = model(scalograms, x, edge_index, batch_assign)

    print(f"   Scalograms: {scalograms.shape}")
    print(f"   Node features: {x.shape}")
    print(f"   Logits: {outputs['logits'].shape}")
    print(f"   Predictions: {(outputs['probs'] > 0.5).sum().item()}/{len(labels)} positive")

    # Quick training
    print("\n4. Quick training test (3 epochs)...")
    model = AdvancedEEGDepressionDetector(config).cuda()
    trainer = Trainer(
        model=model,
        device='cuda',
        mixed_precision=True,
        gradient_accumulation_steps=1,
        gradient_clip=1.0
    )

    trainer.train(
        train_loader, val_loader,
        num_epochs=3,
        early_stopping_patience=10,
        fold=0
    )

    print("\n" + "=" * 60)
    print("PIPELINE TEST PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
