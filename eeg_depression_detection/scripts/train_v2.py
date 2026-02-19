#!/usr/bin/env python3
"""
Training Script V2: Transformer + Bi-LSTM + GNN

This script trains the enhanced model with three branches:
1. Transformer: CWT scalograms (time-frequency)
2. Bi-LSTM: Raw EEG sequences (temporal dynamics)
3. GNN: WPD features (spatial connectivity)

Usage:
------
python scripts/train_v2.py --data_dir data/raw/figshare --output_dir outputs_v2
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.datasets.figshare_dataset import FigshareEEGDataset, create_dataloaders
from models.full_model_v2 import AdvancedEEGDepressionDetectorV2, ModelConfigV2, model_summary_v2
from models.branches.gnn_encoder import create_eeg_graph_batch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainerV2:
    """Trainer for V2 model with three branches."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        mixed_precision: bool = True,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 2,
        gradient_clip: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip = gradient_clip

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if mixed_precision else None

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            scalograms = batch['scalogram'].to(self.device)
            raw_eeg = batch['raw_eeg'].to(self.device)
            wpd_features = batch['wpd_features'].to(self.device)
            labels = batch['label'].float().to(self.device)

            # Create graph batch
            x, edge_index, batch_assign = create_eeg_graph_batch(wpd_features.cpu())
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            batch_assign = batch_assign.to(self.device)

            # Forward pass
            with autocast('cuda', enabled=self.mixed_precision):
                outputs = self.model(
                    scalograms, raw_eeg, x, edge_index, batch_assign
                )
                logits = outputs['logits'].squeeze(-1)
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            progress_bar.set_postfix({'loss': total_loss / num_batches})

        self.scheduler.step()
        return total_loss / num_batches


def leave_one_subject_out_cv_v2(
    dataset,
    model_config: ModelConfigV2,
    training_config: dict,
    device: str = 'cuda',
    output_dir: Path = None,
    resume_fold: int = 0
) -> dict:
    """
    LOSO cross-validation for V2 model.

    Same as V1 but with three-branch model.
    Supports resuming from a specific fold.
    """
    subjects = dataset.get_unique_subjects()
    n_subjects = len(subjects)

    print(f"\n{'='*60}")
    print(f"LOSO Cross-Validation (V2: Transformer + Bi-LSTM + GNN)")
    print(f"Total subjects: {n_subjects}")
    if resume_fold > 0:
        print(f"RESUMING from fold {resume_fold + 1}")
    print(f"{'='*60}\n")

    # Collect predictions across all folds
    all_preds = []
    all_probs = []
    all_labels = []
    all_subjects = []

    # Load previous results if resuming
    checkpoint_file = output_dir / 'checkpoint.json' if output_dir else None
    if resume_fold > 0 and checkpoint_file and checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            all_preds = checkpoint['preds']
            all_probs = checkpoint['probs']
            all_labels = checkpoint['labels']
            all_subjects = checkpoint['subjects']
        print(f"Loaded {len(all_preds)} predictions from previous folds")

    # Determine how many folds to run
    max_folds = training_config.get('max_folds', n_subjects)
    if max_folds is None:
        max_folds = n_subjects

    for fold_idx, test_subject in enumerate(subjects):
        # Skip completed folds when resuming
        if fold_idx < resume_fold:
            continue
        # Stop if we've reached max_folds
        if fold_idx >= max_folds:
            print(f"\nReached max_folds={max_folds}, stopping early.")
            break
        print(f"\n{'='*40}")
        print(f"Fold {fold_idx + 1}/{n_subjects}: Test subject = {test_subject}")
        print(f"{'='*40}")

        # Split subjects
        train_subjects = [s for s in subjects if s != test_subject]

        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            dataset,
            train_subjects=train_subjects,
            val_subjects=[test_subject],
            batch_size=training_config.get('batch_size', 16),
            num_workers=training_config.get('num_workers', 4)
        )

        # Create new model
        model = AdvancedEEGDepressionDetectorV2(model_config)

        # Create trainer
        trainer = TrainerV2(
            model=model,
            device=device,
            mixed_precision=training_config.get('mixed_precision', True),
            learning_rate=training_config.get('learning_rate', 1e-4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
            gradient_clip=training_config.get('gradient_clip', 1.0)
        )

        # Train
        max_epochs = training_config.get('epochs_per_fold', 30)
        for epoch in range(1, max_epochs + 1):
            train_loss = trainer.train_epoch(train_loader, epoch)
            if epoch % 10 == 0 or epoch == max_epochs:
                print(f"  Epoch {epoch}/{max_epochs} - Train Loss: {train_loss:.4f}")

        # Collect predictions
        trainer.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                scalograms = batch['scalogram'].to(device)
                raw_eeg = batch['raw_eeg'].to(device)
                wpd_features = batch['wpd_features'].to(device)
                labels = batch['label'].float().to(device)

                x, edge_index, batch_assign = create_eeg_graph_batch(wpd_features.cpu())
                x = x.to(device)
                edge_index = edge_index.to(device)
                batch_assign = batch_assign.to(device)

                with autocast('cuda', enabled=training_config.get('mixed_precision', True)):
                    outputs = trainer.model(scalograms, raw_eeg, x, edge_index, batch_assign)
                    probs = outputs['probs'].squeeze(-1)
                    preds = (probs > 0.5).long()

                all_preds.extend(preds.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_subjects.extend([test_subject] * len(labels))

        # Log subject prediction
        subject_label = int(all_labels[-1])
        subject_pred = int(np.mean([p for p, s in zip(all_preds, all_subjects) if s == test_subject]) > 0.5)
        print(f"  Subject {test_subject}: True={subject_label}, Pred={subject_pred}")

        # Save checkpoint after each fold
        if checkpoint_file:
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'fold': fold_idx + 1,
                    'preds': [float(p) for p in all_preds],
                    'probs': [float(p) for p in all_probs],
                    'labels': [float(l) for l in all_labels],
                    'subjects': all_subjects
                }, f)
            print(f"  Checkpoint saved (fold {fold_idx + 1}/{n_subjects})")

        # Save model weights
        if output_dir:
            models_dir = output_dir / 'models'
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f'fold_{fold_idx}.pt'
            torch.save(model.state_dict(), model_path)
            print(f"  Model saved: {model_path}")

        # Free memory
        del model, trainer
        torch.cuda.empty_cache()

    # Compute aggregated metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

    aggregated = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': len(all_labels),
        'total_subjects': n_subjects
    }

    # Subject-level accuracy
    subject_preds = {}
    subject_labels = {}
    for pred, prob, label, subj in zip(all_preds, all_probs, all_labels, all_subjects):
        if subj not in subject_preds:
            subject_preds[subj] = []
            subject_labels[subj] = label
        subject_preds[subj].append(prob)

    subject_correct = sum(
        1 for subj in subject_preds
        if (np.mean(subject_preds[subj]) > 0.5) == subject_labels[subj]
    )
    aggregated['subject_accuracy'] = subject_correct / n_subjects

    # Print results
    print(f"\n{'='*60}")
    print("LOSO Cross-Validation Results (V2 Model - Aggregated)")
    print(f"{'='*60}")
    print(f"Sample-level Accuracy: {aggregated['accuracy']:.4f}")
    print(f"Subject-level Accuracy: {aggregated['subject_accuracy']:.4f}")
    print(f"AUC-ROC: {aggregated['auc_roc']:.4f}")
    print(f"F1-Score: {aggregated['f1']:.4f}")
    print(f"Sensitivity: {aggregated['sensitivity']:.4f}")
    print(f"Specificity: {aggregated['specificity']:.4f}")
    print(f"MCC: {aggregated['mcc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")

    return {
        'aggregated': aggregated,
        'all_predictions': {
            'preds': all_preds.tolist(),
            'probs': all_probs.tolist(),
            'labels': all_labels.tolist(),
            'subjects': all_subjects
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train V2 Model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs_v2')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='Resume from output dir (e.g., outputs_v2/run_20260201_170941)')
    parser.add_argument('--resume_fold', type=int, default=0, help='Resume from specific fold number (0-indexed)')
    parser.add_argument('--max_folds', type=int, default=None, help='Maximum number of folds to run (for quick testing)')

    args = parser.parse_args()
    set_seed(args.seed)

    # Handle resume or create new output directory
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            print(f"ERROR: Resume directory not found: {output_dir}")
            return
        # Auto-detect resume fold from checkpoint
        checkpoint_file = output_dir / 'checkpoint.json'
        if checkpoint_file.exists() and args.resume_fold == 0:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                args.resume_fold = checkpoint.get('fold', 0)
            print(f"Auto-detected resume from fold {args.resume_fold}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f'run_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EEG Depression Detection V2: Transformer + Bi-LSTM + GNN")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # Load dataset (will reprocess to include raw_eeg)
    print("\nLoading dataset...")
    dataset = FigshareEEGDataset(
        data_dir=args.data_dir,
        condition='EC',
        precompute_features=True,
        cache_dir=Path(args.data_dir) / 'cache'
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Subjects: {len(dataset.get_unique_subjects())}")
    print(f"Classes: {dataset.get_class_distribution()}")

    # Check if raw_eeg is available
    sample = dataset[0]
    if 'raw_eeg' not in sample:
        print("\nERROR: raw_eeg not in dataset. Please delete cache and reprocess:")
        print(f"  rm {Path(args.data_dir) / 'cache' / 'figshare_EC_features_v2.pkl'}")
        return

    print(f"Raw EEG shape: {sample['raw_eeg'].shape}")

    # Model config
    model_config = ModelConfigV2()

    # Create model to show summary
    model = AdvancedEEGDepressionDetectorV2(model_config)
    summary = model_summary_v2(model)
    print(f"\nModel Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:,}")
    del model

    # Training config
    training_config = {
        'epochs_per_fold': args.epochs if not args.debug else 5,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': 2,
        'gradient_clip': 1.0,
        'num_workers': 4,
        'max_folds': args.max_folds
    }

    # Run LOSO CV
    results = leave_one_subject_out_cv_v2(
        dataset=dataset,
        model_config=model_config,
        training_config=training_config,
        device=args.device,
        output_dir=output_dir,
        resume_fold=args.resume_fold
    )

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        # Convert numpy to list for JSON
        json.dump({
            'aggregated': results['aggregated'],
            'config': {
                'model': 'V2: Transformer + Bi-LSTM + GNN',
                'training': training_config
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
