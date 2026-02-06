#!/usr/bin/env python3
"""
CLI Entry Point for Running Permutation Tests.

This script provides a command-line interface for running permutation tests
to validate whether model performance is statistically significant.

Usage:
    python -m app.statistical_tests.run_permutation_test --help
    
    # Run with observed metrics from a JSON file
    python -m app.statistical_tests.run_permutation_test \
        --observed-metrics metrics.json \
        --n-permutations 100 \
        --output-dir evaluations/permutation_tests
    
    # Run with manually specified observed metrics
    python -m app.statistical_tests.run_permutation_test \
        --observed-f1 0.85 \
        --observed-accuracy 0.82 \
        --observed-precision 0.88 \
        --observed-recall 0.82 \
        --n-permutations 100
    
    # Load observed metrics from MLflow run
    python -m app.statistical_tests.run_permutation_test \
        --mlflow-run-id <run_id> \
        --n-permutations 100
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.loader import load_config


def load_observed_metrics_from_json(filepath: str) -> dict:
    """Load observed metrics from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_observed_metrics_from_mlflow(run_id: str) -> dict:
    """Load observed metrics from an MLflow run."""
    try:
        import mlflow
        
        cfg = load_config()
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        metrics = {}
        metric_mapping = {
            'test_f1': 'f1',
            'test_precision': 'precision',
            'test_recall': 'recall',
            'best_test_f1': 'f1',
            'final_best_f1_value': 'f1'
        }
        
        for mlflow_key, our_key in metric_mapping.items():
            if mlflow_key in run.data.metrics:
                metrics[our_key] = run.data.metrics[mlflow_key]
        
        # Also check params for final values
        params = run.data.params
        if 'final_best_f1_value' in params:
            metrics['f1'] = float(params['final_best_f1_value'])
        
        return metrics
    except ImportError:
        raise ImportError("MLflow is required to load metrics from runs. Install with: pip install mlflow")
    except Exception as e:
        raise ValueError(f"Failed to load metrics from MLflow run {run_id}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run permutation tests to validate model performance significance.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Observed metrics source (mutually exclusive)
    metrics_group = parser.add_argument_group('Observed Metrics Source')
    metrics_source = metrics_group.add_mutually_exclusive_group()
    metrics_source.add_argument(
        '--observed-metrics',
        type=str,
        help='Path to JSON file containing observed metrics'
    )
    metrics_source.add_argument(
        '--mlflow-run-id',
        type=str,
        help='MLflow run ID to load observed metrics from'
    )
    
    # Manual metric specification
    manual_group = parser.add_argument_group('Manual Metrics (if not using --observed-metrics or --mlflow-run-id)')
    manual_group.add_argument('--observed-f1', type=float, help='Observed F1 score')
    manual_group.add_argument('--observed-accuracy', type=float, help='Observed accuracy')
    manual_group.add_argument('--observed-precision', type=float, help='Observed precision')
    manual_group.add_argument('--observed-recall', type=float, help='Observed recall')
    
    # Permutation test parameters
    test_group = parser.add_argument_group('Permutation Test Parameters')
    test_group.add_argument(
        '--n-permutations',
        type=int,
        default=None,
        help='Number of permutation iterations (default: from config)'
    )
    test_group.add_argument(
        '--permutation-epochs',
        type=int,
        default=None,
        help='Number of epochs per permutation iteration (default: from config)'
    )
    test_group.add_argument(
        '--significance-level',
        type=float,
        default=None,
        help='Significance level for hypothesis testing (default: from config)'
    )
    test_group.add_argument(
        '--mode',
        type=str,
        choices=['normal', 'optimized'],
        default='normal',
        help='Test mode: normal (full data) or optimized (sampled data, faster)'
    )
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument(
        '--bat-type',
        type=str,
        choices=['r', 'm'],
        default='r',
        help='Bat type: r=rousettus, m=mauritius (default: r)'
    )
    train_group.add_argument(
        '--augmented',
        action='store_true',
        help='Use augmented data'
    )
    train_group.add_argument(
        '--data-source',
        type=str,
        choices=['video', 'still'],
        default='video',
        help='Data source type (default: video)'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: evaluations/permutation_tests/{timestamp})'
    )
    output_group.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )
    output_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def get_config_defaults(mode: str = 'normal'):
    """Load default parameters from config.
    
    Args:
        mode: 'normal' for full data or 'optimized' for faster iteration
    """
    cfg = load_config()
    
    # Check if statistical_tests config exists
    full_config = cfg.get_full_config()
    stats_config = full_config.get('statistical_tests', {}).get('permutation_test', {})
    optimized_config = stats_config.get('optimized_mode', {})
    
    # Base defaults
    defaults = {
        'n_permutations': stats_config.get('n_permutations', 100),
        'permutation_epochs': stats_config.get('permutation_epochs', 10),
        'significance_level': stats_config.get('significance_level', 0.05),
        'metrics_to_test': stats_config.get('metrics_to_test', ['f1', 'accuracy', 'precision', 'recall']),
        'save_null_distribution': stats_config.get('save_null_distribution', True),
        'sample_fraction': 1.0,
        'shuffle_buffer_fraction': 1.0
    }
    
    # Override with optimized mode settings if requested
    if mode == 'optimized':
        defaults['n_permutations'] = optimized_config.get('n_permutations', 30)
        defaults['permutation_epochs'] = optimized_config.get('permutation_epochs', 5)
        defaults['sample_fraction'] = optimized_config.get('sample_fraction', 0.25)
        defaults['shuffle_buffer_fraction'] = optimized_config.get('shuffle_buffer_fraction', 0.5)
    
    return defaults


def main():
    args = parse_args()
    
    # Load config defaults based on mode
    defaults = get_config_defaults(mode=args.mode)
    
    # Determine observed metrics
    observed_metrics = {}
    
    if args.observed_metrics:
        print(f"Loading observed metrics from: {args.observed_metrics}")
        observed_metrics = load_observed_metrics_from_json(args.observed_metrics)
    elif args.mlflow_run_id:
        print(f"Loading observed metrics from MLflow run: {args.mlflow_run_id}")
        observed_metrics = load_observed_metrics_from_mlflow(args.mlflow_run_id)
    else:
        # Manual metrics
        if args.observed_f1 is not None:
            observed_metrics['f1'] = args.observed_f1
        if args.observed_accuracy is not None:
            observed_metrics['accuracy'] = args.observed_accuracy
        if args.observed_precision is not None:
            observed_metrics['precision'] = args.observed_precision
        if args.observed_recall is not None:
            observed_metrics['recall'] = args.observed_recall
    
    if not observed_metrics:
        print("Error: No observed metrics provided.")
        print("Use --observed-metrics, --mlflow-run-id, or manual metric flags.")
        sys.exit(1)
    
    print(f"\nObserved metrics: {observed_metrics}")
    
    # Set up parameters
    n_permutations = args.n_permutations or defaults['n_permutations']
    permutation_epochs = args.permutation_epochs or defaults['permutation_epochs']
    significance_level = args.significance_level or defaults['significance_level']
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'evaluations/permutation_tests/{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Import here to avoid TensorFlow import at module load
    from .permutation_test import PermutationTest
    from .permutation_trainer import create_permutation_trainer
    from .permutation_visualizer import PermutationVisualizer
    
    # Create permutation test
    perm_test = PermutationTest(
        n_permutations=n_permutations,
        permutation_epochs=permutation_epochs,
        significance_level=significance_level,
        metrics_to_test=defaults['metrics_to_test'],
        save_null_distribution=defaults['save_null_distribution'],
        verbose=args.verbose
    )
    
    # Run permutation test
    print(f"\n{'='*70}")
    print(f"Starting Permutation Test")
    print(f"{'='*70}")
    print(f"  Mode: {args.mode}")
    print(f"  Permutations: {n_permutations}")
    print(f"  Epochs per permutation: {permutation_epochs}")
    print(f"  Significance level: {significance_level}")
    if args.mode == 'optimized':
        print(f"  Sample fraction: {defaults['sample_fraction']} (using {defaults['sample_fraction']*100:.0f}% of data)")
        print(f"  Shuffle buffer fraction: {defaults['shuffle_buffer_fraction']}")
    print(f"  Bat type: {args.bat_type}")
    print(f"  Data source: {args.data_source}")
    print(f"  Augmented: {args.augmented}")
    print(f"{'='*70}\n")
    
    results = perm_test.run(
        observed_metrics=observed_metrics,
        trainer_factory=create_permutation_trainer,
        bat_type=args.bat_type,
        augmented_data=args.augmented,
        data_source=args.data_source,
        sample_fraction=defaults['sample_fraction'],
        shuffle_buffer_fraction=defaults['shuffle_buffer_fraction']
    )
    
    # Save results
    results_path = os.path.join(output_dir, 'permutation_results.json')
    results.save(results_path)
    print(f"\nResults saved to: {results_path}")
    
    # Generate visualizations
    if not args.no_plots:
        visualizer = PermutationVisualizer(results, output_dir=output_dir)
        visualizer.generate_full_report(show=False, export_csv=True, export_json=False)
    
    # Print summary
    results.print_summary()
    
    # Exit with appropriate code
    significant_metrics = [m for m, r in results.metrics.items() if r.significant]
    if significant_metrics:
        print(f"\nModel performance is statistically significant for: {significant_metrics}")
        sys.exit(0)
    else:
        print("\nWarning: Model performance is NOT statistically significant for any metric.")
        sys.exit(1)


if __name__ == '__main__':
    main()
