"""
Training Pipeline for EEG Depression Detection

This module implements:
1. Training loop with mixed precision and gradient accumulation
2. Subject-wise cross-validation (LOSO)
3. Early stopping and model checkpointing
4. Comprehensive logging and metrics tracking

Design Decisions (for research paper):
--------------------------------------
1. LOSO (Leave-One-Subject-Out) Cross-Validation:
   - Prevents data leakage between train and test
   - Simulates real clinical scenario (new patient)
   - Provides realistic performance estimates

2. Mixed Precision Training:
   - Reduces memory usage by ~40%
   - Enables larger batch sizes or models
   - Minimal accuracy impact

3. Gradient Accumulation:
   - Simulates larger batch sizes
   - Important for small-batch stability
   - Effective batch = batch_size × accumulation_steps

4. AdamW Optimizer:
   - Decoupled weight decay (better than L2)
   - Proven effective for Transformers
   - Good default for deep learning

References:
-----------
- Loshchilov & Hutter (2019). Decoupled weight decay regularization.
- Micikevicius et al. (2018). Mixed precision training.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)

from models.full_model import AdvancedEEGDepressionDetector, ModelConfig
from models.branches.gnn_encoder import create_eeg_graph_batch


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Stops training when validation metric doesn't improve for 'patience' epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.001,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like AUC, 'min' for loss
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'max':
            is_improvement = self.best_score is None or score > self.best_score + self.delta
        else:
            is_improvement = self.best_score is None or score < self.best_score - self.delta

        if is_improvement:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


class MetricsTracker:
    """
    Track and compute evaluation metrics.

    Computes comprehensive metrics for depression classification:
    - Accuracy, Precision, Recall, F1
    - Sensitivity, Specificity
    - AUC-ROC
    - Matthews Correlation Coefficient (MCC)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated predictions."""
        self.all_preds = []
        self.all_probs = []
        self.all_labels = []

    def update(
        self,
        preds: torch.Tensor,
        probs: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Add batch predictions.

        Args:
            preds: Binary predictions
            probs: Probability scores
            labels: Ground truth labels
        """
        self.all_preds.extend(preds.cpu().numpy().flatten())
        self.all_probs.extend(probs.cpu().numpy().flatten())
        self.all_labels.extend(labels.cpu().numpy().flatten())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric names to values
        """
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs)
        labels = np.array(self.all_labels)

        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'auc_roc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'mcc': matthews_corrcoef(labels, preds),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

        return metrics


class Trainer:
    """
    Main trainer class for the depression detection model.

    Handles:
    - Training loop with all optimizations
    - Validation
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        criterion: nn.Module = None,
        device: str = 'cuda',
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 2,
        gradient_clip: float = 1.0,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            optimizer: Optimizer (created if None)
            scheduler: LR scheduler (created if None)
            criterion: Loss function (created if None)
            device: Device to train on
            mixed_precision: Whether to use FP16
            gradient_accumulation_steps: Steps to accumulate gradients
            gradient_clip: Maximum gradient norm
            checkpoint_dir: Directory for model checkpoints
            log_dir: Directory for training logs
        """
        self.model = model.to(device)
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip = gradient_clip

        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = optimizer or AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Scheduler
        self.scheduler = scheduler or CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # Loss function with optional class weights
        self.criterion = criterion or nn.BCEWithLogitsLoss()

        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if mixed_precision else None

        # Metrics tracker
        self.metrics_tracker = MetricsTracker()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            scalograms = batch['scalogram'].to(self.device)
            wpd_features = batch['wpd_features'].to(self.device)
            labels = batch['label'].float().to(self.device)

            # Create graph batch
            x, edge_index, batch_assign = create_eeg_graph_batch(wpd_features.cpu())
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            batch_assign = batch_assign.to(self.device)

            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.mixed_precision):
                outputs = self.model(
                    scalograms,
                    x,  # Pass graph node features, not original wpd_features
                    edge_index,
                    batch_assign
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

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss / num_batches})

        # Step scheduler
        self.scheduler.step()

        return total_loss / num_batches

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            (validation_loss, metrics_dict)
        """
        self.model.eval()
        self.metrics_tracker.reset()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in progress_bar:
            scalograms = batch['scalogram'].to(self.device)
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
                    scalograms,
                    x,  # Pass graph node features, not original wpd_features
                    edge_index,
                    batch_assign
                )
                logits = outputs['logits'].squeeze(-1)
                probs = outputs['probs'].squeeze(-1)
                loss = self.criterion(logits, labels)

            # Update metrics
            preds = (probs > 0.5).long()
            self.metrics_tracker.update(preds, probs, labels.long())

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        metrics = self.metrics_tracker.compute()

        return avg_loss, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best: bool = True,
        fold: int = 0
    ) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save the best model
            fold: Fold number for cross-validation

        Returns:
            Training history
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            delta=0.001,
            mode='max'  # Maximize AUC
        )

        best_auc = 0
        best_model_state = None

        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)

            # Validation
            val_loss, val_metrics = self.validate(val_loader, epoch)

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(lr)

            # Print metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val AUC: {val_metrics['auc_roc']:.4f}")
            print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  LR: {lr:.6f}")

            # Save best model
            if val_metrics['auc_roc'] > best_auc:
                best_auc = val_metrics['auc_roc']
                best_model_state = self.model.state_dict().copy()

                if save_best:
                    self._save_checkpoint(
                        epoch, val_metrics,
                        filename=f'best_model_fold{fold}.pt'
                    )

            # Early stopping
            if early_stopping(val_metrics['auc_roc']):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        filename: str
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint


def leave_one_subject_out_cv(
    dataset,
    model_config: ModelConfig,
    training_config: Dict,
    device: str = 'cuda'
) -> Dict:
    """
    Perform Leave-One-Subject-Out cross-validation.

    This is the gold standard for EEG classification to prevent data leakage.
    Metrics are computed on AGGREGATED predictions across all folds.

    Args:
        dataset: Full dataset with subject information
        model_config: Model configuration
        training_config: Training configuration
        device: Device to use

    Returns:
        Dictionary with fold results and aggregated metrics
    """
    subjects = dataset.get_unique_subjects()
    n_subjects = len(subjects)

    print(f"\n{'='*60}")
    print(f"Leave-One-Subject-Out Cross-Validation")
    print(f"Total subjects: {n_subjects}")
    print(f"{'='*60}\n")

    # Collect ALL predictions across folds for proper metric computation
    all_preds = []
    all_probs = []
    all_labels = []
    all_subjects = []
    fold_train_losses = []

    for fold_idx, test_subject in enumerate(subjects):
        print(f"\n{'='*40}")
        print(f"Fold {fold_idx + 1}/{n_subjects}: Test subject = {test_subject}")
        print(f"{'='*40}")

        # Split subjects
        train_subjects = [s for s in subjects if s != test_subject]

        # Create data loaders
        from data.datasets.figshare_dataset import create_dataloaders
        train_loader, val_loader = create_dataloaders(
            dataset,
            train_subjects=train_subjects,
            val_subjects=[test_subject],
            batch_size=training_config.get('batch_size', 16),
            num_workers=training_config.get('num_workers', 4)
        )

        # Create new model for this fold
        model = AdvancedEEGDepressionDetector(model_config)

        # Create trainer - use train loss for early stopping since val has 1 class
        trainer = Trainer(
            model=model,
            device=device,
            mixed_precision=training_config.get('mixed_precision', True),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
            gradient_clip=training_config.get('gradient_clip', 1.0)
        )

        # Train with reduced epochs for LOSO (no meaningful val metrics per fold)
        max_epochs = min(training_config.get('epochs', 100), 30)  # Cap at 30 for LOSO

        # Simple training loop without early stopping on val metrics
        for epoch in range(1, max_epochs + 1):
            train_loss = trainer.train_epoch(train_loader, epoch)
            if epoch % 10 == 0 or epoch == max_epochs:
                print(f"  Epoch {epoch}/{max_epochs} - Train Loss: {train_loss:.4f}")

        fold_train_losses.append(train_loss)

        # Collect predictions for this fold's test subject
        trainer.model.eval()
        trainer.metrics_tracker.reset()

        with torch.no_grad():
            for batch in val_loader:
                scalograms = batch['scalogram'].to(device)
                wpd_features = batch['wpd_features'].to(device)
                labels = batch['label'].float().to(device)

                x, edge_index, batch_assign = create_eeg_graph_batch(wpd_features.cpu())
                x = x.to(device)
                edge_index = edge_index.to(device)
                batch_assign = batch_assign.to(device)

                with torch.amp.autocast('cuda', enabled=training_config.get('mixed_precision', True)):
                    outputs = trainer.model(scalograms, x, edge_index, batch_assign)
                    probs = outputs['probs'].squeeze(-1)
                    preds = (probs > 0.5).long()

                all_preds.extend(preds.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_subjects.extend([test_subject] * len(labels))

        # Get subject's true label for logging
        subject_label = int(all_labels[-1])
        subject_pred = int(np.mean([p for p, s in zip(all_preds, all_subjects) if s == test_subject]) > 0.5)
        print(f"  Subject {test_subject}: True={subject_label}, Pred={subject_pred}")

        # Free memory
        del model, trainer
        torch.cuda.empty_cache()

    # Compute AGGREGATED metrics across all subjects
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Sample-level metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, matthews_corrcoef
    )

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

    # Subject-level accuracy (majority vote per subject)
    subject_preds = {}
    subject_labels = {}
    for pred, prob, label, subj in zip(all_preds, all_probs, all_labels, all_subjects):
        if subj not in subject_preds:
            subject_preds[subj] = []
            subject_labels[subj] = label
        subject_preds[subj].append(prob)

    subject_correct = 0
    for subj in subject_preds:
        avg_prob = np.mean(subject_preds[subj])
        pred_label = 1 if avg_prob > 0.5 else 0
        if pred_label == subject_labels[subj]:
            subject_correct += 1

    aggregated['subject_accuracy'] = subject_correct / n_subjects

    print(f"\n{'='*60}")
    print("LOSO Cross-Validation Results (Aggregated)")
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
