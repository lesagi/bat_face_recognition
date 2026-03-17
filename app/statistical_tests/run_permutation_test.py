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
            'test_accuracy': 'accuracy',
            'best_test_f1': 'f1',
            'final_best_f1_value': 'f1',
            'final_test_accuracy': 'accuracy'
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


REQUIRED_TRAINING_PARAMS = ['bat_type', 'data_source', 'augmented_data']


def load_training_config_from_mlflow(run_id: str) -> dict:
    """Load training configuration (bat_type, data_source, augmented_data) from an MLflow run.
    
    Raises ValueError if any required param is missing from the run,
    so the user can review the run and pass them explicitly via CLI.
    """
    try:
        import mlflow
        
        cfg = load_config()
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        
        missing = [p for p in REQUIRED_TRAINING_PARAMS if p not in params]
        if missing:
            raise ValueError(
                f"MLflow run {run_id} is missing training params: {missing}. "
                f"Please pass them explicitly via CLI flags (e.g. --bat-type, --data-source, --augmented)."
            )
        
        return {
            'bat_type': params['bat_type'],
            'data_source': params['data_source'],
            'augmented_data': params['augmented_data'].lower() == 'true',
        }
    except ImportError:
        raise ImportError("MLflow is required to load config from runs. Install with: pip install mlflow")


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
        default=None,
        help='Bat type: r=rousettus, m=mauritius (default: r, or from MLflow run)'
    )
    train_group.add_argument(
        '--augmented',
        action='store_true',
        default=None,
        help='Use augmented data'
    )
    train_group.add_argument(
        '--data-source',
        type=str,
        choices=['video', 'still'],
        default=None,
        help='Data source type (default: video, or from MLflow run)'
    )
    train_group.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU ID to use (e.g., 0 or 1). Sets CUDA_VISIBLE_DEVICES before TF init.'
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
    
    # Merge mode
    merge_group = parser.add_argument_group('Merge Mode (combine partial results from parallel GPU runs)')
    merge_group.add_argument(
        '--merge-results',
        nargs=2,
        type=str,
        metavar='RESULT_JSON',
        help='Merge two partial result JSON files instead of running a test'
    )
    
    return parser.parse_args()


def get_config_defaults(mode: str = 'normal'):
    cfg = load_config()
    
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
        'shuffle_buffer_fraction': 1.0,
        'optimizer': 'adam',
        'batch_size': 128,
        'eval_last_only': True
    }
    
    # Override with optimized mode settings if requested
    if mode == 'optimized':
        defaults['n_permutations'] = optimized_config.get('n_permutations', 100)
        defaults['permutation_epochs'] = optimized_config.get('permutation_epochs', 10)
        defaults['sample_fraction'] = optimized_config.get('sample_fraction', 0.5)
        defaults['shuffle_buffer_fraction'] = optimized_config.get('shuffle_buffer_fraction', 0.5)
        defaults['optimizer'] = optimized_config.get('optimizer', 'sgd')
        defaults['batch_size'] = optimized_config.get('batch_size', 128)
        defaults['eval_last_only'] = optimized_config.get('eval_last_only', True)
    
    return defaults


def merge_results(result_paths, output_dir, observed_metrics_source=None, mlflow_run_id=None,
                   significance_level=0.05, no_plots=False):
    from .permutation_test import PermutationTest, PermutationTestResults

    print(f"Merging results from {len(result_paths)} files...", flush=True)
    
    all_results = []
    for path in result_paths:
        print(f"  Loading: {path}", flush=True)
        all_results.append(PermutationTestResults.load(path))
    
    # Combine null distributions
    combined_null = {}
    all_metrics = set()
    for r in all_results:
        for metric_name, metric_result in r.metrics.items():
            all_metrics.add(metric_name)
            if metric_name not in combined_null:
                combined_null[metric_name] = []
            combined_null[metric_name].extend(metric_result.null_distribution)
    
    total_permutations = sum(r.n_permutations for r in all_results)
    print(f"  Total permutations: {total_permutations}", flush=True)
    for metric_name in sorted(combined_null.keys()):
        print(f"  {metric_name}: {len(combined_null[metric_name])} values in null distribution", flush=True)
    
    # Get observed metrics from the first result file
    observed_metrics = {}
    for metric_name in all_metrics:
        for r in all_results:
            if metric_name in r.metrics:
                observed_metrics[metric_name] = r.metrics[metric_name].observed
                break
    
    # Override observed metrics if provided
    if observed_metrics_source:
        loaded = load_observed_metrics_from_json(observed_metrics_source)
        observed_metrics.update(loaded)
    elif mlflow_run_id:
        loaded = load_observed_metrics_from_mlflow(mlflow_run_id)
        observed_metrics.update(loaded)
    
    print(f"  Observed metrics: {observed_metrics}", flush=True)
    
    # Recalculate p-values with the combined null distribution
    perm_test = PermutationTest(
        n_permutations=total_permutations,
        permutation_epochs=all_results[0].permutation_epochs,
        significance_level=significance_level,
        metrics_to_test=list(all_metrics),
        save_null_distribution=True
    )
    
    merged = perm_test.run_from_null_distributions(observed_metrics, combined_null)
    
    # Save merged results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'permutation_results.json')
    merged.save(results_path)
    print(f"\nMerged results saved to: {results_path}", flush=True)
    
    # Generate visualizations
    if not no_plots:
        from .permutation_visualizer import PermutationVisualizer
        visualizer = PermutationVisualizer(merged, output_dir=output_dir)
        visualizer.generate_full_report(show=False, export_csv=True, export_json=False)
    
    merged.print_summary()
    return merged


def main():
    args = parse_args()
    
    # Set GPU before any TensorFlow imports
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"GPU restricted to device {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})", flush=True)
    
    # Handle merge mode (no TF needed for this)
    if args.merge_results:
        output_dir = args.output_dir
        if not output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'evaluations/permutation_tests/merged_{timestamp}'
        
        significance_level = args.significance_level or 0.05
        merge_results(
            result_paths=args.merge_results,
            output_dir=output_dir,
            observed_metrics_source=args.observed_metrics if hasattr(args, 'observed_metrics') else None,
            mlflow_run_id=args.mlflow_run_id if hasattr(args, 'mlflow_run_id') else None,
            significance_level=significance_level,
            no_plots=args.no_plots
        )
        sys.exit(0)
    
    # Load config defaults based on mode
    defaults = get_config_defaults(mode=args.mode)
    
    # Determine observed metrics and training config
    observed_metrics = {}
    mlflow_config = {}
    
    if args.observed_metrics:
        print(f"Loading observed metrics from: {args.observed_metrics}", flush=True)
        observed_metrics = load_observed_metrics_from_json(args.observed_metrics)
    elif args.mlflow_run_id:
        print(f"Loading observed metrics from MLflow run: {args.mlflow_run_id}", flush=True)
        observed_metrics = load_observed_metrics_from_mlflow(args.mlflow_run_id)
        mlflow_config = load_training_config_from_mlflow(args.mlflow_run_id)
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
        print("Error: No observed metrics provided.", flush=True)
        print("Use --observed-metrics, --mlflow-run-id, or manual metric flags.", flush=True)
        sys.exit(1)
    
    print(f"\nObserved metrics: {observed_metrics}", flush=True)
    
    # Resolve training config: CLI override > MLflow > hardcoded default
    bat_type = args.bat_type if args.bat_type is not None else mlflow_config.get('bat_type', 'r')
    data_source = args.data_source if args.data_source is not None else mlflow_config.get('data_source', 'video')
    augmented = args.augmented if args.augmented is not None else mlflow_config.get('augmented_data', False)
    
    if mlflow_config:
        print(f"Loaded training config from MLflow run: "
              f"bat_type={mlflow_config['bat_type']}, "
              f"data_source={mlflow_config['data_source']}, "
              f"augmented={mlflow_config['augmented_data']}", flush=True)
        for param, cli_val, mlflow_val in [
            ('bat_type', args.bat_type, mlflow_config['bat_type']),
            ('data_source', args.data_source, mlflow_config['data_source']),
            ('augmented', args.augmented, mlflow_config['augmented_data']),
        ]:
            if cli_val is not None and cli_val != mlflow_val:
                print(f"  {param}={cli_val} (overridden by CLI, MLflow had: {mlflow_val})", flush=True)
    
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
    print(f"Output directory: {output_dir}", flush=True)
    
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
    print(f"\n{'='*70}", flush=True)
    print(f"Starting Permutation Test", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Mode: {args.mode}", flush=True)
    print(f"  Permutations: {n_permutations}", flush=True)
    print(f"  Epochs per permutation: {permutation_epochs}", flush=True)
    print(f"  Significance level: {significance_level}", flush=True)
    if args.mode == 'optimized':
        print(f"  Optimizer: {defaults['optimizer']} (SGD uses less memory than Adam)", flush=True)
        print(f"  Batch size: {defaults['batch_size']}", flush=True)
        print(f"  Sample fraction: {defaults['sample_fraction']}", flush=True)
        print(f"  Shuffle buffer fraction: {defaults['shuffle_buffer_fraction']}", flush=True)
        print(f"  Eval last only: {defaults['eval_last_only']}", flush=True)
    print(f"  Bat type: {bat_type}", flush=True)
    print(f"  Data source: {data_source}", flush=True)
    print(f"  Augmented: {augmented}", flush=True)
    if args.gpu is not None:
        print(f"  GPU: {args.gpu}", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    results = perm_test.run(
        observed_metrics=observed_metrics,
        trainer_factory=create_permutation_trainer,
        bat_type=bat_type,
        augmented_data=augmented,
        data_source=data_source,
        sample_fraction=defaults['sample_fraction'],
        shuffle_buffer_fraction=defaults['shuffle_buffer_fraction'],
        optimizer=defaults['optimizer'],
        batch_size=defaults['batch_size'],
        eval_last_only=defaults['eval_last_only']
    )
    
    # Save results
    results_path = os.path.join(output_dir, 'permutation_results.json')
    results.save(results_path)
    print(f"\nResults saved to: {results_path}", flush=True)
    
    # Generate visualizations
    if not args.no_plots:
        visualizer = PermutationVisualizer(results, output_dir=output_dir)
        visualizer.generate_full_report(show=False, export_csv=True, export_json=False)
    
    # Print summary
    results.print_summary()
    
    # Exit with appropriate code
    significant_metrics = [m for m, r in results.metrics.items() if r.significant]
    if significant_metrics:
        print(f"\nModel performance is statistically significant for: {significant_metrics}", flush=True)
        sys.exit(0)
    else:
        print("\nWarning: Model performance is NOT statistically significant for any metric.", flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
