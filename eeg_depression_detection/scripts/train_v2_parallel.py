#!/usr/bin/env python3
"""
Parallel LOSO Training for V2 Model

Runs multiple folds simultaneously to speed up training.
With 8GB GPU, can run 2 folds in parallel (~2GB each).

Usage:
    python scripts/train_v2_parallel.py \
        --data_dir data/raw/figshare \
        --output_dir outputs_v2_parallel \
        --n_parallel 2 \
        --epochs 30
"""

import argparse
import sys
import json
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import queue
import time

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def train_single_fold(
    fold_idx: int,
    test_subject: str,
    train_subjects: List[str],
    data_dir: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    result_queue: mp.Queue
):
    """Train a single fold - runs in separate process."""
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    from tqdm import tqdm

    from data.datasets.figshare_dataset import FigshareEEGDataset, create_dataloaders
    from models.full_model_v2 import AdvancedEEGDepressionDetectorV2, ModelConfigV2
    from models.branches.gnn_encoder import create_eeg_graph_batch

    try:
        # Load dataset
        dataset = FigshareEEGDataset(
            data_dir=data_dir,
            condition='EC',
            precompute_features=True,
            cache_dir=Path(data_dir) / 'cache'
        )

        # Create dataloaders
        train_loader, test_loader = create_dataloaders(
            dataset, train_subjects, [test_subject],
            batch_size=batch_size, num_workers=2
        )

        # Create model
        model = AdvancedEEGDepressionDetectorV2(ModelConfigV2())
        model = model.to(device)

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler()

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                scalogram = batch['scalogram'].unsqueeze(1).to(device)
                raw_eeg = batch['raw_eeg'].to(device)
                wpd = batch['wpd_features']
                labels = batch['label'].float().to(device)

                x, edge_index, batch_idx = create_eeg_graph_batch(wpd)
                x = x.to(device)
                edge_index = edge_index.to(device)
                batch_idx = batch_idx.to(device)

                optimizer.zero_grad()

                with autocast():
                    outputs = model(scalogram, raw_eeg, x, edge_index, batch_idx)
                    loss = criterion(outputs['logits'].squeeze(), labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            scheduler.step()

        # Evaluation
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                scalogram = batch['scalogram'].unsqueeze(1).to(device)
                raw_eeg = batch['raw_eeg'].to(device)
                wpd = batch['wpd_features']
                labels = batch['label']

                x, edge_index, batch_idx = create_eeg_graph_batch(wpd)
                x = x.to(device)
                edge_index = edge_index.to(device)
                batch_idx = batch_idx.to(device)

                outputs = model(scalogram, raw_eeg, x, edge_index, batch_idx)
                probs = outputs['probs'].squeeze().cpu().numpy()

                if probs.ndim == 0:
                    probs = np.array([probs])

                preds = (probs > 0.5).astype(int)

                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())

        # Compute metrics
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_labels) if all_labels else 0

        result = {
            'fold': fold_idx,
            'subject': test_subject,
            'accuracy': accuracy,
            'n_samples': len(all_labels),
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
            'status': 'success'
        }

    except Exception as e:
        result = {
            'fold': fold_idx,
            'subject': test_subject,
            'status': 'error',
            'error': str(e)
        }

    result_queue.put(result)


def run_parallel_training(args):
    """Run parallel LOSO training."""
    from data.datasets.figshare_dataset import FigshareEEGDataset

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EEG Depression Detection V2 - PARALLEL Training")
    print("=" * 60)
    print(f"Parallel folds: {args.n_parallel}")
    print(f"Output: {output_dir}")

    # Load dataset to get subjects
    print("\nLoading dataset...")
    dataset = FigshareEEGDataset(
        data_dir=args.data_dir,
        condition='EC',
        precompute_features=True,
        cache_dir=Path(args.data_dir) / 'cache'
    )

    subjects = dataset.get_unique_subjects()
    n_subjects = len(subjects)
    print(f"Total subjects: {n_subjects}")
    print(f"Estimated time: ~{(n_subjects * 10) // args.n_parallel} minutes")

    # Prepare all folds
    folds = []
    for i, test_subject in enumerate(subjects):
        train_subjects = [s for s in subjects if s != test_subject]
        folds.append((i, test_subject, train_subjects))

    # Results storage
    all_results = []
    completed = 0

    # Process folds in parallel batches
    mp.set_start_method('spawn', force=True)

    print(f"\nStarting parallel training ({args.n_parallel} folds at a time)...")
    print("-" * 60)

    for batch_start in range(0, len(folds), args.n_parallel):
        batch_folds = folds[batch_start:batch_start + args.n_parallel]
        result_queue = mp.Queue()
        processes = []

        # Start processes for this batch
        for fold_idx, test_subject, train_subjects in batch_folds:
            p = mp.Process(
                target=train_single_fold,
                args=(
                    fold_idx, test_subject, train_subjects,
                    args.data_dir, output_dir,
                    args.epochs, args.batch_size, args.lr,
                    'cuda', result_queue
                )
            )
            p.start()
            processes.append(p)
            print(f"  Started fold {fold_idx + 1}/{n_subjects} (Subject: {test_subject})")

        # Wait for batch to complete
        for _ in batch_folds:
            result = result_queue.get(timeout=3600)  # 1 hour timeout
            all_results.append(result)
            completed += 1

            if result['status'] == 'success':
                print(f"  ✓ Fold {result['fold'] + 1} done: {result['accuracy']:.2%} ({completed}/{n_subjects})")
            else:
                print(f"  ✗ Fold {result['fold'] + 1} error: {result['error']}")

        # Join processes
        for p in processes:
            p.join()

    # Aggregate results
    print("\n" + "=" * 60)
    print("Aggregating Results...")

    successful_results = [r for r in all_results if r['status'] == 'success']

    all_preds = []
    all_probs = []
    all_labels = []
    all_subjects_list = []

    for r in successful_results:
        all_preds.extend(r['predictions'])
        all_probs.extend(r['probabilities'])
        all_labels.extend(r['labels'])
        all_subjects_list.extend([r['subject']] * len(r['predictions']))

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

    # Subject-level accuracy
    subject_preds = {}
    subject_labels = {}
    for pred, prob, label, subj in zip(all_preds, all_probs, all_labels, all_subjects_list):
        if subj not in subject_preds:
            subject_preds[subj] = []
            subject_labels[subj] = label
        subject_preds[subj].append(prob)

    subject_correct = sum(
        1 for subj in subject_preds
        if (np.mean(subject_preds[subj]) > 0.5) == subject_labels[subj]
    )

    results = {
        'sample_accuracy': float(accuracy_score(all_labels, all_preds)),
        'subject_accuracy': float(subject_correct / len(subject_preds)),
        'f1_score': float(f1_score(all_labels, all_preds)),
        'auc_roc': float(roc_auc_score(all_labels, all_probs)),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        'n_subjects': len(subject_preds),
        'n_samples': len(all_labels),
        'successful_folds': len(successful_results),
        'failed_folds': len(all_results) - len(successful_results)
    }

    # Print results
    print("\n" + "=" * 60)
    print("V2 PARALLEL TRAINING RESULTS")
    print("=" * 60)
    print(f"Sample Accuracy:  {results['sample_accuracy']:.4f}")
    print(f"Subject Accuracy: {results['subject_accuracy']:.4f}")
    print(f"AUC-ROC:          {results['auc_roc']:.4f}")
    print(f"F1-Score:         {results['f1_score']:.4f}")
    print(f"Sensitivity:      {results['sensitivity']:.4f}")
    print(f"Specificity:      {results['specificity']:.4f}")

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parallel V2 Training')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs_v2_parallel')
    parser.add_argument('--n_parallel', type=int, default=2, help='Number of parallel folds')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()
    run_parallel_training(args)
