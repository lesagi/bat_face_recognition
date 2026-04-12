"""
Permutation Test for Statistical Significance of Model Performance.

This module implements permutation testing to determine whether the model's
performance metrics (F1, accuracy, precision, recall) are statistically
significantly better than random chance.

The test works by:
1. Recording the observed metrics from the actual trained model
2. Running multiple training iterations with randomly permuted labels
3. Building a null distribution from permuted results
4. Calculating p-values by comparing observed metrics to null distribution
"""

import os
import sys
import json
import time
import gc
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class MetricResult:
    observed: float
    null_mean: float
    null_std: float
    p_value: float
    significant: bool
    null_distribution: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "p_value": self.p_value,
            "significant": self.significant,
            "null_distribution": self.null_distribution
        }


@dataclass
class PermutationTestResults:
    n_permutations: int
    significance_level: float
    metrics: Dict[str, MetricResult]
    total_time_seconds: float
    permutation_epochs: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_permutations": self.n_permutations,
            "significance_level": self.significance_level,
            "permutation_epochs": self.permutation_epochs,
            "total_time_seconds": self.total_time_seconds,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()}
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'PermutationTestResults':
        with open(path, 'r') as f:
            data = json.load(f)
        
        metrics = {}
        for name, metric_data in data["metrics"].items():
            metrics[name] = MetricResult(
                observed=metric_data["observed"],
                null_mean=metric_data["null_mean"],
                null_std=metric_data["null_std"],
                p_value=metric_data["p_value"],
                significant=metric_data["significant"],
                null_distribution=metric_data.get("null_distribution", [])
            )
        
        return cls(
            n_permutations=data["n_permutations"],
            significance_level=data["significance_level"],
            permutation_epochs=data["permutation_epochs"],
            total_time_seconds=data["total_time_seconds"],
            metrics=metrics
        )
    
    def print_summary(self):
        print(f"\n{'='*70}", flush=True)
        print(f"Permutation Test Results (n={self.n_permutations})", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"{'Metric':<12} {'Observed':>10} {'Mean(Null)':>12} {'Std(Null)':>10} {'P-value':>10} {'Significant':<12}", flush=True)
        print(f"{'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*12}", flush=True)
        
        for name, result in self.metrics.items():
            sig_str = f"Yes (p<{self.significance_level})" if result.significant else "No"
            print(f"{name:<12} {result.observed:>10.4f} {result.null_mean:>12.4f} {result.null_std:>10.4f} {result.p_value:>10.4f} {sig_str:<12}", flush=True)
        
        print(f"{'='*70}", flush=True)
        print(f"Total time: {self.total_time_seconds:.1f} seconds", flush=True)
        print(f"{'='*70}\n", flush=True)


class PermutationTest:
    def __init__(
        self,
        n_permutations: int = 100,
        permutation_epochs: int = 10,
        significance_level: float = 0.05,
        metrics_to_test: Optional[List[str]] = None,
        save_null_distribution: bool = True,
        verbose: bool = True
    ):
        self.n_permutations = n_permutations
        self.permutation_epochs = permutation_epochs
        self.significance_level = significance_level
        self.metrics_to_test = metrics_to_test or ["f1", "accuracy", "precision", "recall"]
        self.save_null_distribution = save_null_distribution
        self.verbose = verbose
        
        # Storage for null distributions
        self.null_distributions: Dict[str, List[float]] = {
            metric: [] for metric in self.metrics_to_test
        }
    
    def compute_p_value(self, observed: float, null_distribution: np.ndarray) -> float:
        """
        Compute p-value for observed metric against null distribution.
        
        Uses the formula: p = (count(null >= observed) + 1) / (n + 1)
        The +1 in numerator and denominator is for continuity correction.
        
        Args:
            observed: The observed metric value from the real model
            null_distribution: Array of metric values from permuted runs
            
        Returns:
            p-value between 0 and 1
        """
        n = len(null_distribution)
        # Count how many permuted values are >= observed (one-sided test)
        count_greater_equal = np.sum(null_distribution >= observed)
        # Continuity correction
        p_value = (count_greater_equal + 1) / (n + 1)
        return float(p_value)
    
    def _print_gpu_status(self):
        import tensorflow as tf
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpu_count = len(gpus) if gpus else 0
        
        print(f"GPU(s) available: {gpu_count}", flush=True)
        if gpu_count > 0:
            gpu_names = [gpu.name for gpu in gpus]
            print(f"  Using: {gpu_names}", flush=True)
        print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}", flush=True)
        print(flush=True)
    
    def run(
        self,
        observed_metrics: Dict[str, float],
        trainer_factory: callable,
        bat_type: str = 'r',
        augmented_data: bool = False,
        data_source: str = 'video',
        background: str = 'random',
        sample_fraction: float = 1.0,
        shuffle_buffer_fraction: float = 1.0,
        optimizer: str = 'adam',
        batch_size: int = 128,
        eval_last_only: bool = True
    ) -> PermutationTestResults:
        start_time = time.time()
        
        # Reset null distributions
        self.null_distributions = {metric: [] for metric in self.metrics_to_test}
        
        if self.verbose:
            print(f"\n{'='*70}", flush=True)
            print(f"Starting Permutation Test", flush=True)
            print(f"{'='*70}", flush=True)
            print(f"Number of permutations: {self.n_permutations}", flush=True)
            print(f"Epochs per permutation: {self.permutation_epochs}", flush=True)
            print(f"Significance level: {self.significance_level}", flush=True)
            print(f"Metrics to test: {self.metrics_to_test}", flush=True)
            if sample_fraction < 1.0:
                print(f"Sample fraction: {sample_fraction} (using {sample_fraction*100:.0f}% of data)", flush=True)
            print(f"Shuffle buffer fraction: {shuffle_buffer_fraction}", flush=True)
            print(f"Optimizer: {optimizer}", flush=True)
            print(f"Batch size: {batch_size}", flush=True)
            print(f"Eval last only: {eval_last_only}", flush=True)
            print(f"{'='*70}\n", flush=True)
            
            # Show GPU status once before starting iterations
            self._print_gpu_status()
        
        # Create trainer ONCE -- the model and optimizer stay alive across all
        # permutations to avoid TF BFC allocator memory leaks. Only the data
        # (with new permuted labels) and weights are reset each iteration.
        trainer = trainer_factory(
            bat_type=bat_type,
            augmented_data=augmented_data,
            data_source=data_source,
            background=background,
            permute_labels=True,
            num_epochs=self.permutation_epochs,
            mlflow_enabled=False,
            verbose=self.verbose,
            sample_fraction=sample_fraction,
            shuffle_buffer_fraction=shuffle_buffer_fraction,
            optimizer=optimizer,
            batch_size=batch_size,
            eval_last_only=eval_last_only
        )
        
        # Run permutation iterations
        permutation_times = []
        
        for i in range(self.n_permutations):
            perm_start = time.time()
            
            if self.verbose:
                if permutation_times:
                    avg_time = sum(permutation_times) / len(permutation_times)
                    remaining = avg_time * (self.n_permutations - i)
                    remaining_min = remaining / 60
                    eta_str = f" | ETA: {remaining_min:.1f} min" if remaining_min < 60 else f" | ETA: {remaining_min/60:.1f} hrs"
                else:
                    eta_str = ""
                elapsed_total = time.time() - start_time
                elapsed_min = elapsed_total / 60
                print(f"\n--- Permutation {i+1}/{self.n_permutations} (elapsed: {elapsed_min:.1f} min{eta_str}) ---", flush=True)
            
            # Reset weights and create new data with fresh label permutation.
            # Skip on the first iteration since the trainer was just created.
            if i > 0:
                trainer.reset_for_new_permutation()
            
            # Train and get metrics
            metrics = trainer.train_and_evaluate()
            
            # Store metrics in null distributions
            for metric in self.metrics_to_test:
                if metric in metrics:
                    self.null_distributions[metric].append(metrics[metric])
            
            perm_time = time.time() - perm_start
            permutation_times.append(perm_time)
            
            if self.verbose:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k in self.metrics_to_test])
                print(f"   Permutation {i+1} metrics: {metrics_str} (took {perm_time:.1f}s)", flush=True)
            
            # Light cleanup -- just free data references, not the model
            gc.collect()
        
        # Calculate results for each metric
        results_metrics: Dict[str, MetricResult] = {}
        
        for metric in self.metrics_to_test:
            if metric not in observed_metrics:
                if self.verbose:
                    print(f"Warning: Metric '{metric}' not found in observed_metrics, skipping", flush=True)
                continue
                
            observed = observed_metrics[metric]
            null_dist = np.array(self.null_distributions[metric])
            
            if len(null_dist) == 0:
                if self.verbose:
                    print(f"Warning: No null distribution data for metric '{metric}'", flush=True)
                continue
            
            p_value = self.compute_p_value(observed, null_dist)
            null_mean = float(np.mean(null_dist))
            null_std = float(np.std(null_dist))
            significant = p_value < self.significance_level
            
            results_metrics[metric] = MetricResult(
                observed=observed,
                null_mean=null_mean,
                null_std=null_std,
                p_value=p_value,
                significant=significant,
                null_distribution=self.null_distributions[metric] if self.save_null_distribution else []
            )
        
        total_time = time.time() - start_time
        
        results = PermutationTestResults(
            n_permutations=self.n_permutations,
            significance_level=self.significance_level,
            permutation_epochs=self.permutation_epochs,
            metrics=results_metrics,
            total_time_seconds=total_time
        )
        
        if self.verbose:
            results.print_summary()
        
        return results
    
    def run_from_null_distributions(
        self,
        observed_metrics: Dict[str, float],
        null_distributions: Dict[str, List[float]]
    ) -> PermutationTestResults:
        """
        Calculate results from pre-computed null distributions.
        
        Useful when null distributions were computed separately or loaded from file.
        
        Args:
            observed_metrics: Dictionary of observed metrics
            null_distributions: Pre-computed null distributions for each metric
            
        Returns:
            PermutationTestResults object
        """
        results_metrics: Dict[str, MetricResult] = {}
        
        for metric in self.metrics_to_test:
            if metric not in observed_metrics or metric not in null_distributions:
                continue
            
            observed = observed_metrics[metric]
            null_dist = np.array(null_distributions[metric])
            
            p_value = self.compute_p_value(observed, null_dist)
            null_mean = float(np.mean(null_dist))
            null_std = float(np.std(null_dist))
            significant = p_value < self.significance_level
            
            results_metrics[metric] = MetricResult(
                observed=observed,
                null_mean=null_mean,
                null_std=null_std,
                p_value=p_value,
                significant=significant,
                null_distribution=null_distributions[metric] if self.save_null_distribution else []
            )
        
        return PermutationTestResults(
            n_permutations=len(list(null_distributions.values())[0]) if null_distributions else 0,
            significance_level=self.significance_level,
            permutation_epochs=self.permutation_epochs,
            metrics=results_metrics,
            total_time_seconds=0.0  # Not applicable for pre-computed
        )
