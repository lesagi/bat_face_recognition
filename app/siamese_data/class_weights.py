"""
Class weighting strategies for handling imbalanced Siamese network training.

Implements three weighting schemes:
- INS (Inverse Number of Samples): Best for moderate imbalance
- ISNS (Inverse Square Root of Number of Samples): Gentler weighting
- ENS (Effective Number of Samples): Best for extreme imbalance

References:
- https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
- Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
"""

import numpy as np
from typing import Dict, Tuple, List, Union, Optional
from collections import Counter
from enum import Enum

# Import anchor/negative weight calculation from dedicated module
from siamese_data.anchor_negative_weights import (
    calculate_anchor_negative_weights,
    compute_batch_anchor_negative_weights
)

# Import PairClassInfo for multi-class pair support
from siamese_data.pair_class_info import PairClassInfo


class NegativePairCombination(str, Enum):
    """
    Strategies for combining weights of multiple classes in negative pairs.
    
    - SUM: Most stable, recommended for general use
    - GEOMETRIC_MEAN: Balanced approach between sum and product
    - PRODUCT: Aggressive weighting for extreme imbalance
    """
    SUM = "sum"
    GEOMETRIC_MEAN = "geometric_mean"
    PRODUCT = "product"


def combine_class_weights(
    weights: List[float],
    combination: str
) -> float:
    """
    Combine multiple class weights into single pair weight.
    
    This function is used for negative pairs that involve multiple classes.
    Different combination strategies provide different sensitivity to class imbalance.
    
    Args:
        weights: List of individual class weights (from global distribution)
        combination: Combination strategy ('sum', 'geometric_mean', 'product')
        
    Returns:
        Combined weight value
        
    Raises:
        ValueError: If combination strategy is invalid or weights list is empty
        
    Example:
        >>> # Two classes with different frequencies
        >>> weights = [0.2, 0.5]  # Class A rarer than Class B
        >>> combine_class_weights(weights, 'sum')
        0.35  # (0.2 + 0.5) / 2
    """
    if not weights:
        raise ValueError("Weights list cannot be empty")
    
    # Single weight case - return as-is (positive pairs)
    if len(weights) == 1:
        return weights[0]
    
    combination = combination.lower()
    
    if combination == NegativePairCombination.SUM:
        # Arithmetic mean of per-class weights keeps the scale consistent
        # with single-class (positive pair) weights.
        return sum(weights) / len(weights)
    
    elif combination == NegativePairCombination.GEOMETRIC_MEAN:
        # Geometric mean of per-class weights -- slightly favours rare-class
        # pairs over arithmetic mean while staying on the same scale.
        product = np.prod(weights)
        return product ** (1.0 / len(weights))
    
    elif combination == NegativePairCombination.PRODUCT:
        # Product of per-class weights -- most aggressive towards rare pairs.
        return np.prod(weights)
    
    else:
        valid_strategies = [e.value for e in NegativePairCombination]
        raise ValueError(
            f"Invalid combination strategy: '{combination}'. "
            f"Must be one of: {valid_strategies}"
        )


def calculate_ins_weights(class_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate Inverse Number of Samples (INS) weights.
    
    Formula: weight_c = 1 / n_c
    Where n_c is the number of samples in class c.
    
    Weights are normalized to sum to 1.
    
    Args:
        class_counts: Dictionary mapping class names to sample counts
        
    Returns:
        Dictionary mapping class names to normalized weights
        
    Raises:
        ValueError: If class counts are invalid (empty, negative, or zero)
    """
    if not class_counts:
        raise ValueError("Class counts dictionary cannot be empty")
    
    if any(count <= 0 for count in class_counts.values()):
        raise ValueError("All class counts must be positive")
    
    # Calculate inverse weights
    inv_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # Normalize so weights sum to 1
    total = sum(inv_weights.values())
    normalized_weights = {cls: w / total for cls, w in inv_weights.items()}
    
    return normalized_weights


def calculate_isns_weights(class_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate Inverse Square Root of Number of Samples (ISNS) weights.
    
    Formula: weight_c = 1 / sqrt(n_c)
    Where n_c is the number of samples in class c.
    
    This is a gentler weighting scheme compared to INS.
    Weights are normalized to sum to 1.
    
    Args:
        class_counts: Dictionary mapping class names to sample counts
        
    Returns:
        Dictionary mapping class names to normalized weights
        
    Raises:
        ValueError: If class counts are invalid
    """
    if not class_counts:
        raise ValueError("Class counts dictionary cannot be empty")
    
    if any(count <= 0 for count in class_counts.values()):
        raise ValueError("All class counts must be positive")
    
    # Calculate inverse square root weights
    inv_sqrt_weights = {cls: 1.0 / np.sqrt(count) for cls, count in class_counts.items()}
    
    # Normalize so weights sum to 1
    total = sum(inv_sqrt_weights.values())
    normalized_weights = {cls: w / total for cls, w in inv_sqrt_weights.items()}
    
    return normalized_weights


def calculate_ens_weights(class_counts: Dict[str, int], beta: float = 0.9999) -> Dict[str, float]:
    """
    Calculate Effective Number of Samples (ENS) weights.
    
    Formula: weight_c = (1 - beta) / (1 - beta^n_c)
    Where:
    - n_c is the number of samples in class c
    - beta ∈ [0, 1) is a hyperparameter
    
    As described in "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019),
    the effective number accounts for data overlap. Higher beta values give more weight
    to minority classes.
    
    Recommended beta values: 0.9, 0.99, 0.999, 0.9999
    
    Weights are normalized to sum to 1.
    
    Args:
        class_counts: Dictionary mapping class names to sample counts
        beta: Hyperparameter controlling the weighting strength (default: 0.9999)
        
    Returns:
        Dictionary mapping class names to normalized weights
        
    Raises:
        ValueError: If class counts are invalid or beta is out of range
    """
    if not class_counts:
        raise ValueError("Class counts dictionary cannot be empty")
    
    if any(count <= 0 for count in class_counts.values()):
        raise ValueError("All class counts must be positive")
    
    if not (0 <= beta < 1):
        raise ValueError("Beta must be in range [0, 1)")
    
    # Calculate effective number weights
    if beta == 0:
        # Special case: beta=0 gives uniform weights
        ens_weights = {cls: 1.0 for cls in class_counts}
    else:
        ens_weights = {}
        for cls, count in class_counts.items():
            numerator = 1.0 - beta
            denominator = 1.0 - (beta ** count)
            # Handle numerical stability for very small denominators
            if denominator < 1e-10:
                denominator = 1e-10
            ens_weights[cls] = numerator / denominator
    
    # Normalize so weights sum to 1
    total = sum(ens_weights.values())
    normalized_weights = {cls: w / total for cls, w in ens_weights.items()}
    
    return normalized_weights


def calculate_per_class_weights(
    class_counts: Dict[str, int],
    scheme: str = 'ins',
    beta: float = 0.9999
) -> Dict[str, float]:
    """
    Calculate per-class weights using the specified weighting scheme.
    
    Args:
        class_counts: Dictionary mapping class names to sample counts
        scheme: Weighting scheme ('ins', 'isns', or 'ens')
        beta: Beta parameter for ENS scheme (ignored for other schemes)
        
    Returns:
        Dictionary mapping class names to weights
        
    Raises:
        ValueError: If scheme is invalid or class counts are empty
    """
    if not class_counts:
        raise ValueError("Class counts dictionary cannot be empty")
    
    scheme = scheme.lower()
    
    if scheme == 'ins':
        return calculate_ins_weights(class_counts)
    elif scheme == 'isns':
        return calculate_isns_weights(class_counts)
    elif scheme == 'ens':
        return calculate_ens_weights(class_counts, beta)
    else:
        raise ValueError(f"Invalid weighting scheme: {scheme}. Must be 'ins', 'isns', or 'ens'")


class ClassWeightCalculator:
    """
    Main class for computing sample weights in Siamese network training.
    
    Handles two types of imbalance:
    1. Anchor vs Negative imbalance (more negative pairs than positive)
    2. Per-class imbalance (some classes have more samples than others)
    
    Usage:
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
            'ens_beta': 0.9999
        }
        
        calculator = ClassWeightCalculator(config)
        weights = calculator.compute_sample_weights(labels, class_info)
    """
    
    def __init__(self, config: Dict, global_class_distribution: Optional[Dict[str, int]] = None):
        """
        Initialize the weight calculator.
        
        Args:
            config: Configuration dictionary with keys:
                - enabled (bool): Whether to apply class balancing
                - anchor_negative_balance (bool): Balance anchor vs negative pairs
                - per_class_balance (bool): Balance across classes
                - weighting_scheme (str): 'ins', 'isns', or 'ens'
                - ens_beta (float): Beta parameter for ENS scheme
                - negative_pair_combination (str): Strategy for combining multi-class weights
                    ('sum', 'geometric_mean', 'product')
            global_class_distribution: Global class distribution for per-class balancing.
                Required if per_class_balance is True.
                
        Raises:
            ValueError: If per_class_balance is True but global_class_distribution is None
            ValueError: If negative_pair_combination is invalid
        """
        self.enabled = config.get('enabled', False)
        self.anchor_negative_balance = config.get('anchor_negative_balance', True)
        self.per_class_balance = config.get('per_class_balance', True)
        self.weighting_scheme = config.get('weighting_scheme', 'ins')
        self.ens_beta = config.get('ens_beta', 0.9999)
        self.negative_pair_combination = config.get('negative_pair_combination', 'sum')
        
        # Validate combination strategy
        valid_strategies = [e.value for e in NegativePairCombination]
        if self.negative_pair_combination not in valid_strategies:
            raise ValueError(
                f"Invalid negative_pair_combination: '{self.negative_pair_combination}'. "
                f"Must be one of: {valid_strategies}"
            )
        
        # Store global class distribution
        self._global_class_distribution = global_class_distribution
        
        # Strict validation: per_class_balance requires global distribution (only when enabled)
        if self.enabled and self.per_class_balance and self._global_class_distribution is None:
            raise ValueError(
                "per_class_balance=True requires global_class_distribution. "
                "Use 'global_distribution_strategy' in config to compute it. "
                "Options: 'file_based', 'sampled', 'full_scan'"
            )
        
        # Cached weights
        self._anchor_weight = None
        self._negative_weight = None
        self._class_weights = None
    
    def compute_sample_weights(
        self,
        labels: np.ndarray,
        class_info: List[str]
    ) -> np.ndarray:
        """
        Compute per-sample weights for a batch of training samples.
        
        Now supports multi-class pairs (e.g., negative pairs with two classes).
        The class_info parameter contains serialized PairClassInfo strings.
        
        Args:
            labels: Array of labels (1.0 for anchors/positives, 0.0 for negatives)
            class_info: List of serialized PairClassInfo strings
                - Positive pairs: "class_a:1.0"
                - Negative pairs: "class_a:1.0|class_b:1.0"
            
        Returns:
            Array of weights (same length as labels)
            
        Raises:
            ValueError: If labels and class_info have different lengths
        """
        if len(labels) != len(class_info):
            raise ValueError(
                f"Labels length ({len(labels)}) must match class_info length ({len(class_info)})"
            )
        
        # If disabled, return uniform weights
        if not self.enabled:
            return np.ones(len(labels), dtype=np.float32)
        
        # Initialize weights to 1
        weights = np.ones(len(labels), dtype=np.float32)
        
        # Apply anchor/negative balancing (pair-level balance)
        if self.anchor_negative_balance:
            anchor_weight, negative_weight = self._compute_anchor_negative_weights(labels)
            
            # Apply to each sample based on its label
            weights[labels == 1.0] *= anchor_weight
            weights[labels == 0.0] *= negative_weight
        
        # Apply per-class balancing (class-level balance) with multi-class support
        if self.per_class_balance:
            global_class_weights = self._compute_class_weights()
            
            # Process each pair
            for i, class_info_str in enumerate(class_info):
                # Parse serialized PairClassInfo
                pair_info = PairClassInfo.from_string(class_info_str)
                
                # Get weights for all classes in this pair
                pair_class_weights = [
                    global_class_weights.get(cls.name, 1.0)
                    for cls in pair_info.classes
                ]
                
                # Apply weight based on number of classes
                if len(pair_class_weights) == 1:
                    # Positive pair: single class
                    weights[i] *= pair_class_weights[0]
                else:
                    # Negative pair: combine multiple class weights
                    combined_weight = combine_class_weights(
                        pair_class_weights,
                        self.negative_pair_combination
                    )
                    weights[i] *= combined_weight
        
        # Normalize weights to prevent loss scale issues
        # Keep mean at 1.0 so loss magnitude is comparable
        weights = weights / np.mean(weights)
        
        return weights
    
    def set_global_class_distribution(self, distribution: Dict[str, int]) -> None:
        """
        Set the global class distribution after initialization.
        
        Useful for lazy initialization patterns where the distribution
        is computed after the calculator is created.
        
        Args:
            distribution: Dictionary mapping class names to sample counts
            
        Raises:
            ValueError: If distribution is empty or invalid
        """
        if not distribution:
            raise ValueError("Global class distribution cannot be empty")
        
        # Validate all values are positive integers
        for class_name, count in distribution.items():
            if not isinstance(count, int) or count <= 0:
                raise ValueError(
                    f"Invalid count for class '{class_name}': {count}. "
                    "All counts must be positive integers."
                )
        
        self._global_class_distribution = distribution
    
    def _compute_anchor_negative_weights(
        self,
        labels: np.ndarray
    ) -> Tuple[float, float]:
        """Compute anchor and negative weights from labels."""
        num_anchors = np.sum(labels == 1.0)
        num_negatives = np.sum(labels == 0.0)
        
        # Handle edge cases where we have only one type
        if num_anchors == 0 and num_negatives > 0:
            # Only negatives in this batch
            return 0.0, 1.0
        elif num_negatives == 0 and num_anchors > 0:
            # Only anchors in this batch
            return 1.0, 0.0
        elif num_anchors == 0 and num_negatives == 0:
            # Empty batch (shouldn't happen)
            return 1.0, 1.0
        
        return calculate_anchor_negative_weights(int(num_anchors), int(num_negatives))
    
    def _compute_class_weights(self) -> Dict[str, float]:
        """
        Compute per-class weights from global class distribution.
        
        Uses the globally-computed class distribution (stable across batches)
        rather than per-batch counts (which can vary significantly).
        
        Returns:
            Dictionary mapping class names to weights
            
        Raises:
            ValueError: If global distribution is not set (defensive check)
        """
        if self._global_class_distribution is None:
            raise ValueError(
                "Global class distribution is not set. This should not happen "
                "if per_class_balance validation passed during initialization."
            )
        
        # Calculate weights using global distribution and specified scheme
        return calculate_per_class_weights(
            self._global_class_distribution,
            scheme=self.weighting_scheme,
            beta=self.ens_beta
        )
    
    def create_weight_lookup_table(self):
        """
        Create a TensorFlow hash table for O(1) weight lookups.
        
        Pre-computes weights for all possible class_info strings (both single-class
        for positive pairs and dual-class for negative pairs).
        
        Returns:
            tf.lookup.StaticHashTable or None if per_class_balance is disabled
        """
        import tensorflow as tf
        
        if not self.per_class_balance or self._global_class_distribution is None:
            return None
        
        class_weights = self._compute_class_weights()
        class_names = list(class_weights.keys())
        keys, values = [], []
        
        # Single-class entries (positive pairs): "class_a:1.0"
        for cls in class_names:
            keys.append(f"{cls}:1.0")
            values.append(class_weights[cls])
        
        # Dual-class entries (negative pairs): "class_a:1.0|class_b:1.0"
        for i, cls_a in enumerate(class_names):
            for cls_b in class_names[i+1:]:
                combined = combine_class_weights(
                    [class_weights[cls_a], class_weights[cls_b]], 
                    self.negative_pair_combination
                )
                # Add both orderings
                keys.append(f"{cls_a}:1.0|{cls_b}:1.0")
                values.append(combined)
                keys.append(f"{cls_b}:1.0|{cls_a}:1.0")
                values.append(combined)
        
        # Create TensorFlow hash table
        init = tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values, dtype=tf.float32)
        )
        return tf.lookup.StaticHashTable(init, default_value=1.0)
    
    def compute_weight_statistics(
        self,
        weights: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistics about the weight distribution.
        
        Args:
            weights: Array of sample weights
            labels: Array of labels (1.0 for anchors, 0.0 for negatives)
            
        Returns:
            Dictionary with weight statistics
        """
        stats = {
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
        }
        
        # Separate statistics for anchors and negatives
        if len(labels) > 0:
            anchor_weights = weights[labels == 1.0]
            negative_weights = weights[labels == 0.0]
            
            if len(anchor_weights) > 0:
                stats['anchor_mean'] = float(np.mean(anchor_weights))
                stats['anchor_std'] = float(np.std(anchor_weights))
            
            if len(negative_weights) > 0:
                stats['negative_mean'] = float(np.mean(negative_weights))
                stats['negative_std'] = float(np.std(negative_weights))
            
            # Effective contribution ratio
            if len(anchor_weights) > 0 and len(negative_weights) > 0:
                anchor_contribution = np.sum(anchor_weights)
                negative_contribution = np.sum(negative_weights)
                total_contribution = anchor_contribution + negative_contribution
                
                stats['anchor_contribution_pct'] = 100.0 * anchor_contribution / total_contribution
                stats['negative_contribution_pct'] = 100.0 * negative_contribution / total_contribution
        
        return stats

