#!/usr/bin/env python3
"""
Main Training Script for EEG Depression Detection

This script runs the complete training pipeline:
1. Load and preprocess the Figshare MDD dataset
2. Extract WPD and CWT features
3. Train the Transformer+GNN model
4. Evaluate with LOSO cross-validation
5. Generate explainability analysis

Usage:
------
python scripts/train.py --data_dir /path/to/figshare --output_dir /path/to/output

For quick testing:
python scripts/train.py --data_dir /path/to/figshare --debug

Author: [Your Name]
Date: [Date]
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.datasets.figshare_dataset import FigshareEEGDataset, create_dataloaders
from models.full_model import AdvancedEEGDepressionDetector, ModelConfig, model_summary
from training.trainer import Trainer, leave_one_subject_out_cv, MetricsTracker


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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(results: dict, output_dir: Path, filename: str):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    with open(output_dir / filename, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Train EEG Depression Detection Model'
    )

    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing Figshare EEG data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory for outputs (checkpoints, logs, results)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Directory for caching extracted features'
    )

    # Model arguments
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model configuration YAML'
    )
    parser.add_argument(
        '--trans_dim',
        type=int,
        default=128,
        help='Transformer model dimension'
    )
    parser.add_argument(
        '--gnn_dim',
        type=int,
        default=128,
        help='GNN hidden dimension'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience'
    )

    # Hardware arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        default=True,
        help='Use mixed precision training'
    )

    # Experiment arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode (quick run with reduced epochs)'
    )
    parser.add_argument(
        '--cv_type',
        type=str,
        default='loso',
        choices=['loso', 'kfold'],
        help='Cross-validation type'
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EEG Depression Detection Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {args.device}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Cross-validation: {args.cv_type.upper()}")

    # Debug mode adjustments
    if args.debug:
        print("\n[DEBUG MODE] Using reduced settings")
        args.epochs = 5
        args.patience = 2

    # =========================================================================
    # 1. Load Dataset
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    cache_dir = args.cache_dir or Path(args.data_dir) / 'cache'

    dataset = FigshareEEGDataset(
        data_dir=args.data_dir,
        condition='EC',  # Eyes Closed
        epoch_length=4.0,
        epoch_overlap=0.5,
        target_sr=250,
        bandpass=(1, 45),
        cache_dir=cache_dir,
        precompute_features=True
    )

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Unique subjects: {len(dataset.get_unique_subjects())}")
    print(f"  Class distribution: {dataset.get_class_distribution()}")

    # =========================================================================
    # 2. Create Model Configuration
    # =========================================================================
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model_config = ModelConfig(
        trans_d_model=args.trans_dim,
        trans_nhead=4,
        trans_num_layers=4,
        trans_dim_ff=args.trans_dim * 4,
        trans_dropout=0.1,
        gnn_hidden_dim=args.gnn_dim,
        gnn_num_heads=4,
        gnn_num_layers=3,
        gnn_dropout=0.3,
        fusion_dim=args.trans_dim,
        fusion_num_heads=4,
        use_gating=True,
        gradient_checkpointing=True
    )

    # Create model to print summary
    model = AdvancedEEGDepressionDetector(model_config)
    summary = model_summary(model)

    print(f"\nModel Summary:")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Trainable parameters: {summary['trainable_parameters']:,}")

    del model  # Free memory

    # =========================================================================
    # 3. Run Cross-Validation
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"Running {args.cv_type.upper()} Cross-Validation")
    print("=" * 60)

    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'early_stopping_patience': args.patience,
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': 2,
        'gradient_clip': 1.0,
        'num_workers': args.num_workers
    }

    if args.cv_type == 'loso':
        results = leave_one_subject_out_cv(
            dataset=dataset,
            model_config=model_config,
            training_config=training_config,
            device=args.device
        )
    else:
        # K-fold cross-validation (for comparison)
        from sklearn.model_selection import StratifiedKFold

        subjects = dataset.get_unique_subjects()
        labels = [dataset.labels[dataset.get_subject_indices(s)[0]] for s in subjects]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

        all_fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
            print(f"\nFold {fold_idx + 1}/5")

            train_subjects = [subjects[i] for i in train_idx]
            val_subjects = [subjects[i] for i in val_idx]

            train_loader, val_loader = create_dataloaders(
                dataset, train_subjects, val_subjects,
                batch_size=args.batch_size, num_workers=args.num_workers
            )

            model = AdvancedEEGDepressionDetector(model_config)
            trainer = Trainer(
                model=model,
                device=args.device,
                mixed_precision=args.mixed_precision
            )

            trainer.train(
                train_loader, val_loader,
                num_epochs=args.epochs,
                early_stopping_patience=args.patience,
                fold=fold_idx
            )

            _, fold_metrics = trainer.validate(val_loader, 0)
            all_fold_metrics.append(fold_metrics)

            del model, trainer
            torch.cuda.empty_cache()

        # Aggregate
        results = {'fold_metrics': all_fold_metrics}
        results['aggregated'] = {}
        for metric in all_fold_metrics[0].keys():
            if isinstance(all_fold_metrics[0][metric], (int, float)):
                values = [f[metric] for f in all_fold_metrics]
                results['aggregated'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }

    # =========================================================================
    # 4. Save Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    # Save configuration
    config_to_save = {
        'args': vars(args),
        'model_config': {
            'trans_d_model': model_config.trans_d_model,
            'gnn_hidden_dim': model_config.gnn_hidden_dim,
            'fusion_dim': model_config.fusion_dim
        },
        'training_config': training_config
    }
    save_results(config_to_save, output_dir, 'config.json')

    # Save results
    save_results(results, output_dir, 'results.json')

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if 'aggregated' in results:
        for metric, stats in results['aggregated'].items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()
