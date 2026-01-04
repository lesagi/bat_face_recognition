"""
Global class distribution computation for Siamese networks.

This module provides three strategies for computing global class distributions
that are used for stable per-class weight balancing:
1. file_based: Fast approximation from file counts
2. sampled: Accurate via sampling a percentage of the dataset
3. full_scan: Most accurate via complete dataset iteration
"""

import time
from enum import Enum
from typing import Dict, Optional
from collections import Counter
import tensorflow as tf

from siamese_data.pair_class_info import PairClassInfo


class GlobalDistributionStrategy(str, Enum):
    """Strategies for computing global class distribution."""
    FILE_BASED = "file_based"
    SAMPLED = "sampled"
    FULL_SCAN = "full_scan"


def compute_global_class_distribution(
    dataset: tf.data.Dataset,
    strategy: str,
    data_splitter=None,
    sampling_percentage: float = 0.1
) -> Dict[str, int]:
    """
    Compute global class distribution using the specified strategy.
    
    Args:
        dataset: TensorFlow dataset containing (img1, img2, label, class_info) tuples
        strategy: Strategy to use ('file_based', 'sampled', or 'full_scan')
        data_splitter: Data splitter instance (required for 'file_based' strategy)
        sampling_percentage: Percentage of data to sample (for 'sampled' strategy)
        
    Returns:
        Dictionary mapping class names to sample counts
        
    Raises:
        ValueError: If strategy is invalid or required parameters are missing
        
    Example:
        >>> dist = compute_global_class_distribution(
        ...     dataset=train_data,
        ...     strategy='sampled',
        ...     sampling_percentage=0.1
        ... )
        >>> print(dist)
        {'class_a': 150, 'class_b': 200, 'class_c': 100}
    """
    # Validate strategy
    try:
        strategy_enum = GlobalDistributionStrategy(strategy.lower())
    except ValueError:
        valid_strategies = [s.value for s in GlobalDistributionStrategy]
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}"
        )
    
    # Dispatch to appropriate strategy
    if strategy_enum == GlobalDistributionStrategy.FILE_BASED:
        if data_splitter is None:
            raise ValueError("file_based strategy requires data_splitter parameter")
        return _compute_from_files(data_splitter)
    
    elif strategy_enum == GlobalDistributionStrategy.SAMPLED:
        if sampling_percentage <= 0 or sampling_percentage > 1:
            raise ValueError(
                f"sampling_percentage must be between 0 and 1, got {sampling_percentage}"
            )
        return _compute_from_sampling(dataset, sampling_percentage)
    
    elif strategy_enum == GlobalDistributionStrategy.FULL_SCAN:
        return _compute_from_full_scan(dataset)
    
    else:
        # Should never reach here due to enum validation
        raise ValueError(f"Unhandled strategy: {strategy}")


def _compute_from_files(data_splitter) -> Dict[str, int]:
    """
    Compute distribution from file counts using combinatorics.
    
    This is the fastest method but provides an approximation. It estimates
    the number of pairs that will be generated from each class based on
    the number of files per class.
    
    For anchor pairs (same class): C(n, 2) = n * (n-1) / 2
    For negative pairs (different classes): n_a * n_b
    
    Args:
        data_splitter: Data splitter with train_class_files attribute
        
    Returns:
        Dictionary mapping class names to estimated pair counts
    """
    class_distribution = {}
    
    # Get file counts per class
    class_files = data_splitter.train_class_files
    
    # Count anchor pairs for each class (combinations of same class)
    for class_name, files in class_files.items():
        n_files = len(files)
        if n_files >= 2:
            # Number of combinations: C(n, 2) = n * (n-1) / 2
            n_anchor_pairs = n_files * (n_files - 1) // 2
            class_distribution[class_name] = class_distribution.get(class_name, 0) + n_anchor_pairs
    
    # Count negative pairs (cross-class products)
    # Each class appears in negative pairs with all other classes
    class_names = list(class_files.keys())
    for i, class_a in enumerate(class_names):
        for class_b in class_names[i+1:]:
            n_files_a = len(class_files[class_a])
            n_files_b = len(class_files[class_b])
            
            # In permutation mode: both (a,b) and (b,a)
            # In combination mode: only (a,b)
            # We'll assume both classes get equal representation
            n_negative_pairs = n_files_a * n_files_b
            
            # Add to both classes
            class_distribution[class_a] = class_distribution.get(class_a, 0) + n_negative_pairs
            class_distribution[class_b] = class_distribution.get(class_b, 0) + n_negative_pairs
    
    if not class_distribution:
        raise ValueError("No class distribution could be computed from files")
    
    return class_distribution


def _compute_from_sampling(
    dataset: tf.data.Dataset,
    sampling_percentage: float
) -> Dict[str, int]:
    """
    Compute distribution by sampling a percentage of the dataset.
    
    This provides a good balance between speed and accuracy. By default,
    sampling 10% of the data gives a representative distribution while
    keeping startup time low.
    
    Args:
        dataset: TensorFlow dataset to sample from
        sampling_percentage: Fraction of dataset to sample (0.0 to 1.0)
        
    Returns:
        Dictionary mapping class names to sample counts
        
    Raises:
        ValueError: If dataset is empty or no samples found
    """
    class_counter = Counter()
    samples_processed = 0
    
    # Get dataset cardinality to determine sample size
    cardinality = dataset.cardinality().numpy()
    
    if cardinality == tf.data.UNKNOWN_CARDINALITY or cardinality < 0:
        # Unknown cardinality - iterate and count
        # Take every Nth sample where N = 1/sampling_percentage
        skip_rate = max(1, int(1.0 / sampling_percentage))
        
        for idx, (_, _, _, class_info) in enumerate(dataset):
            if idx % skip_rate == 0:
                class_info_str = class_info.numpy()
                if isinstance(class_info_str, bytes):
                    class_info_str = class_info_str.decode('utf-8')
                
                # Parse PairClassInfo to get individual classes
                pair_info = PairClassInfo.from_string(class_info_str)
                for cls in pair_info.classes:
                    class_counter[cls.name] += 1
                
                samples_processed += 1
                
                # Safety limit: stop after 10000 samples
                if samples_processed >= 10000:
                    break
    else:
        # Known cardinality - take exact percentage
        num_samples = max(1, int(cardinality * sampling_percentage))
        
        for _, _, _, class_info in dataset.take(num_samples):
            class_info_str = class_info.numpy()
            if isinstance(class_info_str, bytes):
                class_info_str = class_info_str.decode('utf-8')
            
            # Parse PairClassInfo to get individual classes
            pair_info = PairClassInfo.from_string(class_info_str)
            for cls in pair_info.classes:
                class_counter[cls.name] += 1
            
            samples_processed += 1
    
    if not class_counter:
        raise ValueError("No samples found in dataset during sampling")
    
    # Scale up counts to estimate full distribution
    if samples_processed > 0 and sampling_percentage < 1.0:
        scale_factor = 1.0 / sampling_percentage
        class_distribution = {
            cls: max(1, int(count * scale_factor))
            for cls, count in class_counter.items()
        }
    else:
        class_distribution = dict(class_counter)
    
    return class_distribution


def _compute_from_full_scan(dataset: tf.data.Dataset) -> Dict[str, int]:
    """
    Compute distribution by scanning the entire dataset.
    
    This is the most accurate method but also the slowest. It iterates
    through the entire dataset once to count class occurrences.
    
    Args:
        dataset: TensorFlow dataset to scan
        
    Returns:
        Dictionary mapping class names to sample counts
        
    Raises:
        ValueError: If dataset is empty
    """
    class_counter = Counter()
    samples_processed = 0
    
    # Iterate through entire dataset
    for _, _, _, class_info in dataset:
        class_info_str = class_info.numpy()
        if isinstance(class_info_str, bytes):
            class_info_str = class_info_str.decode('utf-8')
        
        # Parse PairClassInfo to get individual classes
        pair_info = PairClassInfo.from_string(class_info_str)
        for cls in pair_info.classes:
            class_counter[cls.name] += 1
        
        samples_processed += 1
    
    if not class_counter:
        raise ValueError("No samples found in dataset during full scan")
    
    return dict(class_counter)


def validate_distribution(distribution: Dict[str, int]) -> None:
    """
    Validate a class distribution dictionary.
    
    Args:
        distribution: Dictionary mapping class names to counts
        
    Raises:
        ValueError: If distribution is invalid
    """
    if not distribution:
        raise ValueError("Distribution cannot be empty")
    
    for class_name, count in distribution.items():
        if not isinstance(class_name, str):
            raise ValueError(f"Class name must be string, got {type(class_name)}")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"Count for class '{class_name}' must be positive integer, got {count}")


def print_distribution_summary(distribution: Dict[str, int], strategy: str) -> None:
    """
    Print a summary of the class distribution.
    
    Args:
        distribution: Dictionary mapping class names to counts
        strategy: Strategy used to compute the distribution
    """
    total = sum(distribution.values())
    num_classes = len(distribution)
    
    print(f"\n📊 Global Class Distribution (strategy: {strategy}):")
    print(f"   Total samples: {total}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes:")
    
    # Sort by count descending
    sorted_classes = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_classes:
        percentage = 100.0 * count / total
        print(f"      {class_name}: {count} ({percentage:.1f}%)")

