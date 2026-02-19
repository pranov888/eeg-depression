#!/usr/bin/env python3
"""
Explainability Analysis for V2 Model (Transformer + Bi-LSTM + GNN)

Analyzes which features/electrodes/frequencies are most important for predictions.

Methods:
1. Branch Contribution Analysis - Which branch contributes most to predictions
2. Integrated Gradients - Feature attribution for each branch
3. Attention Analysis - Where the model attends (Transformer + GNN attention)
4. Electrode Importance - Which electrodes are most predictive

Usage:
------
python explainability/run_explainability_v2.py \
    --checkpoint outputs_v2/run_20260203_204035/checkpoint.json \
    --data_dir data/raw/figshare \
    --output_dir explainability_results_v2
"""

import argparse
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> Tuple[torch.nn.Module, Dict]:
    """Load trained V2 model from checkpoint."""
    from models.full_model_v2 import AdvancedEEGDepressionDetectorV2, ModelConfigV2

    checkpoint_path = Path(checkpoint_path)

    # Handle both direct .pt files and checkpoint.json
    if checkpoint_path.name == 'checkpoint.json':
        with open(checkpoint_path) as f:
            ckpt_info = json.load(f)

        # Find best fold's model weights
        best_fold = None
        best_val_acc = 0
        for fold_key, fold_data in ckpt_info.get('fold_results', {}).items():
            if fold_data.get('val_accuracy', 0) > best_val_acc:
                best_val_acc = fold_data['val_accuracy']
                best_fold = fold_key

        if best_fold:
            fold_num = best_fold.replace('fold_', '')
            weights_path = checkpoint_path.parent / 'models' / f'fold_{fold_num}.pt'
        else:
            # Try to find any model weights
            models_dir = checkpoint_path.parent / 'models'
            if not models_dir.exists():
                # Also try 'checkpoints' for backwards compatibility
                models_dir = checkpoint_path.parent / 'checkpoints'
            weights_files = list(models_dir.glob('fold_*.pt')) if models_dir.exists() else []
            if weights_files:
                weights_path = weights_files[0]
            else:
                raise FileNotFoundError(f"No model weights found. Run training with model saving enabled.")
    else:
        weights_path = checkpoint_path
        ckpt_info = {}

    print(f"Loading model weights from: {weights_path}")

    # Create model
    config = ModelConfigV2()
    model = AdvancedEEGDepressionDetectorV2(config)

    # Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, ckpt_info


def analyze_branch_contributions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Analyze how much each branch contributes to predictions.
    Uses the model's built-in gate weights from three-way fusion.
    """
    print("\n" + "=" * 60)
    print("Analyzing Branch Contributions")
    print("=" * 60)

    trans_weights = []
    lstm_weights = []
    gnn_weights = []
    labels = []
    predictions = []

    model.eval()
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing samples"):
            if count >= n_samples:
                break

            scalograms = batch['scalogram'].to(device)
            raw_eeg = batch['raw_eeg'].to(device)
            wpd_features = batch['wpd_features'].to(device)
            batch_labels = batch['label'].numpy()

            # Forward pass with features
            outputs = model(
                scalograms, raw_eeg, wpd_features,
                return_features=True
            )

            # Collect gate weights
            if outputs.get('gate_weights'):
                trans_weights.append(outputs['gate_weights']['trans_gate'].cpu().numpy())
                lstm_weights.append(outputs['gate_weights']['lstm_gate'].cpu().numpy())
                gnn_weights.append(outputs['gate_weights']['gnn_gate'].cpu().numpy())

            labels.extend(batch_labels)
            predictions.extend((outputs['probs'].cpu().numpy() > 0.5).astype(int).flatten())

            count += len(batch_labels)

    # Aggregate results
    results = {
        'method': 'Branch Contribution Analysis',
        'n_samples': count
    }

    if trans_weights:
        trans_arr = np.concatenate(trans_weights).flatten()
        lstm_arr = np.concatenate(lstm_weights).flatten()
        gnn_arr = np.concatenate(gnn_weights).flatten()
        labels_arr = np.array(labels)

        results['overall'] = {
            'transformer': {'mean': float(trans_arr.mean()), 'std': float(trans_arr.std())},
            'bilstm': {'mean': float(lstm_arr.mean()), 'std': float(lstm_arr.std())},
            'gnn': {'mean': float(gnn_arr.mean()), 'std': float(gnn_arr.std())}
        }

        # Per-class analysis
        mdd_mask = labels_arr == 1
        healthy_mask = labels_arr == 0

        if mdd_mask.any():
            results['mdd_class'] = {
                'transformer': float(trans_arr[mdd_mask].mean()),
                'bilstm': float(lstm_arr[mdd_mask].mean()),
                'gnn': float(gnn_arr[mdd_mask].mean())
            }

        if healthy_mask.any():
            results['healthy_class'] = {
                'transformer': float(trans_arr[healthy_mask].mean()),
                'bilstm': float(lstm_arr[healthy_mask].mean()),
                'gnn': float(gnn_arr[healthy_mask].mean())
            }

        print(f"\nOverall Branch Contributions:")
        print(f"  Transformer: {trans_arr.mean():.3f} ± {trans_arr.std():.3f}")
        print(f"  Bi-LSTM:     {lstm_arr.mean():.3f} ± {lstm_arr.std():.3f}")
        print(f"  GNN:         {gnn_arr.mean():.3f} ± {gnn_arr.std():.3f}")

    return results


def compute_integrated_gradients(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    n_samples: int = 50,
    n_steps: int = 50
) -> Dict[str, Any]:
    """
    Compute Integrated Gradients for electrode importance.
    Focuses on WPD features which have direct electrode correspondence.
    """
    print("\n" + "=" * 60)
    print("Computing Integrated Gradients (Electrode Importance)")
    print("=" * 60)

    electrode_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]

    all_attributions = []
    all_labels = []
    count = 0

    model.eval()

    for batch in tqdm(dataloader, desc="Computing IG"):
        if count >= n_samples:
            break

        scalograms = batch['scalogram'].to(device)
        raw_eeg = batch['raw_eeg'].to(device)
        wpd_features = batch['wpd_features'].to(device)
        labels = batch['label'].numpy()

        for i in range(scalograms.size(0)):
            if count >= n_samples:
                break

            try:
                # Get single sample
                scal = scalograms[i:i+1]
                eeg = raw_eeg[i:i+1]
                wpd = wpd_features[i:i+1].requires_grad_(True)

                # Baseline (zeros)
                baseline = torch.zeros_like(wpd)

                # Compute integrated gradients
                attributions = torch.zeros_like(wpd)

                for step in range(n_steps):
                    alpha = step / n_steps
                    interpolated = baseline + alpha * (wpd - baseline)
                    interpolated = interpolated.requires_grad_(True)

                    outputs = model(scal, eeg, interpolated)
                    logits = outputs['logits']

                    # Gradient w.r.t. interpolated input
                    grad = torch.autograd.grad(
                        logits.sum(),
                        interpolated,
                        create_graph=False
                    )[0]

                    attributions += grad / n_steps

                # Scale by input
                attributions = attributions * (wpd - baseline)

                # Average across feature dimension to get per-electrode importance
                electrode_attr = attributions.abs().mean(dim=-1).detach().cpu().numpy()
                all_attributions.append(electrode_attr.flatten())
                all_labels.append(int(labels[i]))
                count += 1

            except Exception as e:
                print(f"  Error on sample {count}: {e}")
                continue

    # Aggregate results
    results = {
        'method': 'Integrated Gradients',
        'n_samples': count,
        'n_steps': n_steps
    }

    if all_attributions:
        attr_arr = np.array(all_attributions)
        labels_arr = np.array(all_labels)

        # Overall importance
        mean_importance = attr_arr.mean(axis=0)
        std_importance = attr_arr.std(axis=0)

        results['electrode_importance'] = {
            'names': electrode_names,
            'mean': mean_importance.tolist(),
            'std': std_importance.tolist()
        }

        # Per-class importance
        mdd_mask = labels_arr == 1
        healthy_mask = labels_arr == 0

        if mdd_mask.any():
            results['electrode_importance']['mdd_mean'] = attr_arr[mdd_mask].mean(axis=0).tolist()
        if healthy_mask.any():
            results['electrode_importance']['healthy_mean'] = attr_arr[healthy_mask].mean(axis=0).tolist()

        # Top electrodes
        top_indices = np.argsort(mean_importance)[::-1][:5]
        results['top_electrodes'] = [
            {'name': electrode_names[i], 'importance': float(mean_importance[i])}
            for i in top_indices
        ]

        print(f"\nTop 5 Electrodes by Importance:")
        for item in results['top_electrodes']:
            print(f"  {item['name']}: {item['importance']:.6f}")

    return results


def analyze_temporal_patterns(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    n_samples: int = 50
) -> Dict[str, Any]:
    """
    Analyze temporal patterns using Bi-LSTM gradients.
    Shows which time points are most important.
    """
    print("\n" + "=" * 60)
    print("Analyzing Temporal Patterns (Bi-LSTM)")
    print("=" * 60)

    all_temporal_importance = []
    all_labels = []
    count = 0

    model.eval()

    for batch in tqdm(dataloader, desc="Analyzing temporal"):
        if count >= n_samples:
            break

        scalograms = batch['scalogram'].to(device)
        raw_eeg = batch['raw_eeg'].to(device).requires_grad_(True)
        wpd_features = batch['wpd_features'].to(device)
        labels = batch['label'].numpy()

        for i in range(scalograms.size(0)):
            if count >= n_samples:
                break

            try:
                scal = scalograms[i:i+1]
                eeg = raw_eeg[i:i+1].requires_grad_(True)
                wpd = wpd_features[i:i+1]

                outputs = model(scal, eeg, wpd)
                logits = outputs['logits']

                # Gradient w.r.t. raw EEG
                grad = torch.autograd.grad(
                    logits.sum(),
                    eeg,
                    create_graph=False
                )[0]

                # Average across channels to get temporal importance
                temporal_importance = grad.abs().mean(dim=1).detach().cpu().numpy()
                all_temporal_importance.append(temporal_importance.flatten())
                all_labels.append(int(labels[i]))
                count += 1

            except Exception as e:
                continue

    results = {
        'method': 'Temporal Pattern Analysis',
        'n_samples': count
    }

    if all_temporal_importance:
        temporal_arr = np.array(all_temporal_importance)
        labels_arr = np.array(all_labels)

        # Bin into time segments (e.g., 10 segments)
        n_bins = 10
        seq_len = temporal_arr.shape[1]
        bin_size = seq_len // n_bins

        binned_importance = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size
            binned_importance.append(temporal_arr[:, start:end].mean(axis=1))

        binned_arr = np.array(binned_importance).T

        results['temporal_importance'] = {
            'n_bins': n_bins,
            'bin_means': binned_arr.mean(axis=0).tolist(),
            'bin_stds': binned_arr.std(axis=0).tolist()
        }

        # Per-class
        mdd_mask = labels_arr == 1
        healthy_mask = labels_arr == 0

        if mdd_mask.any():
            results['temporal_importance']['mdd_means'] = binned_arr[mdd_mask].mean(axis=0).tolist()
        if healthy_mask.any():
            results['temporal_importance']['healthy_means'] = binned_arr[healthy_mask].mean(axis=0).tolist()

        print(f"\nTemporal Importance by Segment:")
        for i, mean in enumerate(results['temporal_importance']['bin_means']):
            print(f"  Segment {i+1}: {mean:.6f}")

    return results


def analyze_frequency_patterns(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    n_samples: int = 50
) -> Dict[str, Any]:
    """
    Analyze frequency patterns using Transformer scalogram gradients.
    """
    print("\n" + "=" * 60)
    print("Analyzing Frequency Patterns (Transformer)")
    print("=" * 60)

    all_freq_importance = []
    all_labels = []
    count = 0

    model.eval()

    for batch in tqdm(dataloader, desc="Analyzing frequency"):
        if count >= n_samples:
            break

        scalograms = batch['scalogram'].to(device).requires_grad_(True)
        raw_eeg = batch['raw_eeg'].to(device)
        wpd_features = batch['wpd_features'].to(device)
        labels = batch['label'].numpy()

        for i in range(scalograms.size(0)):
            if count >= n_samples:
                break

            try:
                scal = scalograms[i:i+1]
                if scal.dim() == 3:
                    scal = scal.unsqueeze(1)
                scal = scal.requires_grad_(True)
                eeg = raw_eeg[i:i+1]
                wpd = wpd_features[i:i+1]

                outputs = model(scal, eeg, wpd)
                logits = outputs['logits']

                # Gradient w.r.t. scalogram
                grad = torch.autograd.grad(
                    logits.sum(),
                    scal,
                    create_graph=False
                )[0]

                # Average across time dimension to get frequency importance
                # Scalogram shape: (1, 1, H, W) where H = frequency, W = time
                freq_importance = grad.abs().mean(dim=-1).squeeze().detach().cpu().numpy()
                all_freq_importance.append(freq_importance)
                all_labels.append(int(labels[i]))
                count += 1

            except Exception as e:
                continue

    results = {
        'method': 'Frequency Pattern Analysis',
        'n_samples': count
    }

    if all_freq_importance:
        freq_arr = np.array(all_freq_importance)
        labels_arr = np.array(all_labels)

        # Map frequency bins to EEG bands
        n_freq_bins = freq_arr.shape[1]

        # Approximate EEG band mapping (assuming 0-50Hz range mapped to 64 bins)
        band_ranges = {
            'delta': (0, int(n_freq_bins * 4 / 50)),      # 0-4 Hz
            'theta': (int(n_freq_bins * 4 / 50), int(n_freq_bins * 8 / 50)),   # 4-8 Hz
            'alpha': (int(n_freq_bins * 8 / 50), int(n_freq_bins * 13 / 50)),  # 8-13 Hz
            'beta': (int(n_freq_bins * 13 / 50), int(n_freq_bins * 30 / 50)),  # 13-30 Hz
            'gamma': (int(n_freq_bins * 30 / 50), n_freq_bins)  # 30-50 Hz
        }

        band_importance = {}
        for band, (start, end) in band_ranges.items():
            if start < end and end <= n_freq_bins:
                band_importance[band] = {
                    'mean': float(freq_arr[:, start:end].mean()),
                    'std': float(freq_arr[:, start:end].std())
                }

                # Per-class
                mdd_mask = labels_arr == 1
                healthy_mask = labels_arr == 0

                if mdd_mask.any():
                    band_importance[band]['mdd_mean'] = float(freq_arr[mdd_mask, start:end].mean())
                if healthy_mask.any():
                    band_importance[band]['healthy_mean'] = float(freq_arr[healthy_mask, start:end].mean())

        results['band_importance'] = band_importance
        results['raw_frequency_importance'] = freq_arr.mean(axis=0).tolist()

        print(f"\nFrequency Band Importance:")
        for band, values in band_importance.items():
            print(f"  {band:6s}: {values['mean']:.6f}")

    return results


def generate_summary(all_results: Dict) -> Dict[str, Any]:
    """Generate a human-readable summary of findings."""
    summary = {
        'key_findings': []
    }

    # Branch contributions
    if 'branch_contributions' in all_results['methods']:
        bc = all_results['methods']['branch_contributions']
        if 'overall' in bc:
            branches = [(k, v['mean']) for k, v in bc['overall'].items()]
            branches.sort(key=lambda x: x[1], reverse=True)
            summary['dominant_branch'] = branches[0][0]
            summary['key_findings'].append(
                f"The {branches[0][0]} branch contributes most ({branches[0][1]:.1%}) to predictions"
            )

    # Top electrodes
    if 'integrated_gradients' in all_results['methods']:
        ig = all_results['methods']['integrated_gradients']
        if 'top_electrodes' in ig:
            top_elec = [e['name'] for e in ig['top_electrodes'][:3]]
            summary['top_electrodes'] = top_elec
            summary['key_findings'].append(
                f"Most important electrodes: {', '.join(top_elec)}"
            )

    # Frequency bands
    if 'frequency_patterns' in all_results['methods']:
        fp = all_results['methods']['frequency_patterns']
        if 'band_importance' in fp:
            bands = [(k, v['mean']) for k, v in fp['band_importance'].items()]
            bands.sort(key=lambda x: x[1], reverse=True)
            summary['top_frequency_bands'] = [b[0] for b in bands[:2]]
            summary['key_findings'].append(
                f"Most important frequency bands: {bands[0][0]}, {bands[1][0]}"
            )

    return summary


def main():
    parser = argparse.ArgumentParser(description='V2 Model Explainability Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint.json or model weights')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='explainability_results_v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples for analysis')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['branch', 'ig', 'temporal', 'frequency'],
                        help='Methods: branch, ig, temporal, frequency')

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V2 Model Explainability Analysis")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Methods: {args.methods}")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    print("\nLoading model...")
    model, ckpt_info = load_model_from_checkpoint(args.checkpoint, args.device)
    print("Model loaded successfully!")

    # Load dataset
    print("\nLoading dataset...")
    from data.datasets.figshare_dataset import FigshareEEGDataset
    from torch.utils.data import DataLoader

    dataset = FigshareEEGDataset(
        data_dir=args.data_dir,
        condition='EC',
        precompute_features=True,
        cache_dir=Path(args.data_dir) / 'cache'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues
    )

    print(f"Loaded {len(dataset)} samples")

    # Run analyses
    all_results = {
        'timestamp': timestamp,
        'checkpoint': str(args.checkpoint),
        'n_samples': args.n_samples,
        'methods': {}
    }

    if 'branch' in args.methods:
        try:
            results = analyze_branch_contributions(
                model, dataloader, args.device, args.n_samples
            )
            all_results['methods']['branch_contributions'] = results
        except Exception as e:
            print(f"Branch analysis failed: {e}")
            all_results['methods']['branch_contributions'] = {'error': str(e)}

    if 'ig' in args.methods:
        try:
            results = compute_integrated_gradients(
                model, dataloader, args.device, min(args.n_samples, 50)
            )
            all_results['methods']['integrated_gradients'] = results
        except Exception as e:
            print(f"IG analysis failed: {e}")
            all_results['methods']['integrated_gradients'] = {'error': str(e)}

    if 'temporal' in args.methods:
        try:
            results = analyze_temporal_patterns(
                model, dataloader, args.device, min(args.n_samples, 50)
            )
            all_results['methods']['temporal_patterns'] = results
        except Exception as e:
            print(f"Temporal analysis failed: {e}")
            all_results['methods']['temporal_patterns'] = {'error': str(e)}

    if 'frequency' in args.methods:
        try:
            results = analyze_frequency_patterns(
                model, dataloader, args.device, min(args.n_samples, 50)
            )
            all_results['methods']['frequency_patterns'] = results
        except Exception as e:
            print(f"Frequency analysis failed: {e}")
            all_results['methods']['frequency_patterns'] = {'error': str(e)}

    # Generate summary
    all_results['summary'] = generate_summary(all_results)

    # Save results
    with open(output_dir / 'explainability_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    if all_results['summary']['key_findings']:
        print("\nKey Findings:")
        for finding in all_results['summary']['key_findings']:
            print(f"  • {finding}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
