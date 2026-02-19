#!/usr/bin/env python3
"""
Unified Explainability Analysis Runner

Runs all explainability methods:
1. Integrated Gradients (IG) - Feature attribution
2. Layer-wise Relevance Propagation (LRP) - Relevance decomposition
3. TCAV - Concept-based explanations

Usage:
------
python explainability/run_explainability.py \
    --model_path outputs/run_20260201_014750/checkpoints/fold_0.pt \
    --data_dir data/raw/figshare \
    --output_dir explainability_results
"""

import argparse
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_integrated_gradients(
    model,
    dataloader,
    device: str,
    n_samples: int = 50
) -> Dict[str, Any]:
    """Run Integrated Gradients analysis."""
    print("\n" + "=" * 60)
    print("Running Integrated Gradients Analysis")
    print("=" * 60)

    from explainability.integrated_gradients import IntegratedGradients

    explainer = IntegratedGradients(model, device=device)

    # Collect attributions
    all_scalogram_attr = []
    all_electrode_attr = []
    all_labels = []

    count = 0
    for batch in dataloader:
        if count >= n_samples:
            break

        scalograms = batch['scalogram']
        wpd_features = batch['wpd_features']
        labels = batch['label']

        for i in range(scalograms.size(0)):
            if count >= n_samples:
                break

            try:
                # Prepare inputs
                scal = scalograms[i:i+1]
                if scal.dim() == 3:
                    scal = scal.unsqueeze(1)
                scal = scal.to(device)
                wpd = wpd_features[i:i+1].to(device)

                # Create graph
                from models.branches.gnn_encoder import create_eeg_graph_batch
                x, edge_index, batch_idx = create_eeg_graph_batch(wpd.cpu())
                x = x.to(device)
                edge_index = edge_index.to(device)
                batch_idx = batch_idx.to(device)

                # Get attributions
                attr = explainer.attribute(
                    scal, x, edge_index, batch_idx,
                    target_class=int(labels[i])
                )

                if 'scalogram' in attr and attr['scalogram'] is not None:
                    all_scalogram_attr.append(attr['scalogram'].cpu().numpy())
                if 'wpd' in attr and attr['wpd'] is not None:
                    # Average across features to get electrode-level
                    electrode_attr = attr['wpd'].mean(dim=-1).cpu().numpy()
                    all_electrode_attr.append(electrode_attr)

                all_labels.append(int(labels[i]))
                count += 1

                if count % 10 == 0:
                    print(f"  Processed {count}/{n_samples} samples")

            except Exception as e:
                print(f"  Error on sample {count}: {e}")
                continue

    # Aggregate results
    electrode_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]

    results = {
        'method': 'Integrated Gradients',
        'n_samples': count,
    }

    if all_electrode_attr:
        electrode_attr = np.array(all_electrode_attr)
        labels_arr = np.array(all_labels)

        results['electrode_importance'] = {
            'names': electrode_names,
            'mean': electrode_attr.mean(axis=0).tolist(),
            'std': electrode_attr.std(axis=0).tolist(),
            'mdd_mean': electrode_attr[labels_arr == 1].mean(axis=0).tolist() if (labels_arr == 1).any() else [],
            'healthy_mean': electrode_attr[labels_arr == 0].mean(axis=0).tolist() if (labels_arr == 0).any() else []
        }

        # Top electrodes
        mean_importance = electrode_attr.mean(axis=0)
        top_indices = np.argsort(np.abs(mean_importance))[::-1][:5]
        results['top_electrodes'] = [
            {'name': electrode_names[i], 'importance': float(mean_importance[i])}
            for i in top_indices
        ]

        print(f"\n  Top 5 Electrodes by IG importance:")
        for item in results['top_electrodes']:
            print(f"    {item['name']}: {item['importance']:.4f}")

    if all_scalogram_attr:
        scalogram_attr = np.array(all_scalogram_attr)
        # Frequency importance (average across time)
        freq_importance = np.abs(scalogram_attr).mean(axis=(0, -1))
        results['frequency_importance'] = freq_importance.tolist()

    return results


def run_lrp_analysis(
    model,
    dataloader,
    device: str,
    n_samples: int = 50
) -> Dict[str, Any]:
    """Run LRP analysis."""
    print("\n" + "=" * 60)
    print("Running Layer-wise Relevance Propagation Analysis")
    print("=" * 60)

    from explainability.lrp import create_lrp_analyzer

    analyzer = create_lrp_analyzer(model, rule='epsilon', device=device)

    # Compute electrode importance
    print("  Computing electrode importance...")
    electrode_results = analyzer.compute_electrode_importance(dataloader, n_samples)

    # Compute frequency importance
    print("  Computing frequency importance...")
    freq_results = analyzer.compute_frequency_importance(dataloader, n_samples)

    results = {
        'method': 'Layer-wise Relevance Propagation',
        'rule': 'epsilon',
        'n_samples': n_samples,
        'electrode_importance': {
            'names': electrode_results['electrode_names'],
            'mean': electrode_results['mean_relevance'].tolist(),
            'std': electrode_results['std_relevance'].tolist(),
            'mdd_mean': electrode_results['mdd_relevance'].tolist(),
            'healthy_mean': electrode_results['healthy_relevance'].tolist(),
            'difference': electrode_results['relevance_difference'].tolist()
        },
        'frequency_importance': {
            'bands': {k: v for k, v in freq_results['band_importance'].items()},
            'raw': freq_results['raw_freq_relevance'].tolist() if len(freq_results['raw_freq_relevance']) > 0 else []
        }
    }

    # Top electrodes
    mean_relevance = electrode_results['mean_relevance']
    top_indices = np.argsort(np.abs(mean_relevance))[::-1][:5]
    results['top_electrodes'] = [
        {'name': electrode_results['electrode_names'][i], 'relevance': float(mean_relevance[i])}
        for i in top_indices
    ]

    print(f"\n  Top 5 Electrodes by LRP relevance:")
    for item in results['top_electrodes']:
        print(f"    {item['name']}: {item['relevance']:.4f}")

    print(f"\n  Frequency Band Importance:")
    for band, values in freq_results['band_importance'].items():
        print(f"    {band}: {values['mean']:.4f} (MDD: {values['mdd']:.4f}, Healthy: {values['healthy']:.4f})")

    return results


def run_tcav_analysis(
    model,
    dataloader,
    device: str,
    layer_name: str = 'fusion.fusion_layer',
    n_cav_examples: int = 30,
    n_tcav_samples: int = 50
) -> Dict[str, Any]:
    """Run TCAV analysis."""
    print("\n" + "=" * 60)
    print("Running TCAV (Concept Activation Vectors) Analysis")
    print("=" * 60)

    from explainability.tcav import create_tcav_analyzer

    # Try different layer names
    layer_options = [
        'fusion.fusion_layer',
        'fusion',
        'classifier.classifier.0',
        'classifier'
    ]

    analyzer = None
    for layer in layer_options:
        try:
            analyzer = create_tcav_analyzer(model, layer_name=layer, device=device)
            # Test if layer exists
            analyzer._register_hook()
            analyzer._remove_hook()
            print(f"  Using layer: {layer}")
            break
        except Exception as e:
            continue

    if analyzer is None:
        print("  ERROR: Could not find suitable layer for TCAV")
        return {'method': 'TCAV', 'error': 'No suitable layer found'}

    # Run analysis
    tcav_results = analyzer.run_full_analysis(
        dataloader,
        concepts=['alpha_asymmetry', 'theta_elevation', 'alpha_reduction'],
        n_cav_examples=n_cav_examples,
        n_tcav_samples=n_tcav_samples
    )

    results = {
        'method': 'TCAV',
        'layer': layer_name,
        'concepts': {}
    }

    for concept_name, concept_results in tcav_results.items():
        if 'error' not in concept_results:
            results['concepts'][concept_name] = {
                'description': concept_results.get('description', ''),
                'cav_accuracy': concept_results['cav_accuracy'],
                'mdd_tcav_score': concept_results['mdd']['tcav_score'],
                'mdd_p_value': concept_results['mdd']['p_value'],
                'mdd_significant': concept_results['mdd']['significant'],
                'healthy_tcav_score': concept_results['healthy']['tcav_score'],
                'healthy_p_value': concept_results['healthy']['p_value'],
                'healthy_significant': concept_results['healthy']['significant']
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='Run Explainability Analysis')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='explainability_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['ig', 'lrp', 'tcav'],
                        help='Methods to run: ig, lrp, tcav')
    parser.add_argument('--model_weights', type=str, default=None,
                        help='Path to trained model weights (optional)')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EEG Depression Detection - Explainability Analysis")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Methods: {args.methods}")

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
        num_workers=2
    )

    print(f"Loaded {len(dataset)} samples")

    # Create model
    print("\nCreating model...")
    from models.full_model import AdvancedEEGDepressionDetector, ModelConfig

    model = AdvancedEEGDepressionDetector(ModelConfig())
    model = model.to(args.device)

    # Load trained weights if provided
    if args.model_weights and Path(args.model_weights).exists():
        print(f"Loading weights from: {args.model_weights}")
        state_dict = torch.load(args.model_weights, map_location=args.device)
        model.load_state_dict(state_dict)
        print("Trained weights loaded successfully!")
    else:
        print("WARNING: Using untrained weights - results will be meaningless!")
        print("Use --model_weights to load trained model")

    model.eval()

    # Run analyses
    all_results = {
        'timestamp': timestamp,
        'n_samples': args.n_samples,
        'methods': {}
    }

    if 'ig' in args.methods:
        try:
            ig_results = run_integrated_gradients(
                model, dataloader, args.device, args.n_samples
            )
            all_results['methods']['integrated_gradients'] = ig_results
        except Exception as e:
            print(f"IG analysis failed: {e}")
            all_results['methods']['integrated_gradients'] = {'error': str(e)}

    if 'lrp' in args.methods:
        try:
            lrp_results = run_lrp_analysis(
                model, dataloader, args.device, args.n_samples
            )
            all_results['methods']['lrp'] = lrp_results
        except Exception as e:
            print(f"LRP analysis failed: {e}")
            all_results['methods']['lrp'] = {'error': str(e)}

    if 'tcav' in args.methods:
        try:
            tcav_results = run_tcav_analysis(
                model, dataloader, args.device
            )
            all_results['methods']['tcav'] = tcav_results
        except Exception as e:
            print(f"TCAV analysis failed: {e}")
            all_results['methods']['tcav'] = {'error': str(e)}

    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        return obj

    all_results = convert_to_json_serializable(all_results)

    # Save results
    with open(output_dir / 'explainability_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
