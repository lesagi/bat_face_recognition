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


def calculate_anchor_negative_weights(
    num_anchors: int, 
    num_negatives: int
) -> Tuple[float, float]:
    """
    Calculate weights to balance anchor (positive) and negative pairs.
    
    The goal is to make both types contribute equally to the loss, so:
    - If there are more negatives, anchors get higher weight
    - If there are more anchors, negatives get higher weight
    
    Returns weights that sum to 1.0 and create a 50/50 contribution split.
    
    Args:
        num_anchors: Number of anchor (positive) pairs
        num_negatives: Number of negative pairs
        
    Returns:
        Tuple of (anchor_weight, negative_weight)
        
    Raises:
        ValueError: If either count is zero or negative
    """
    if num_anchors <= 0:
        raise ValueError("Number of anchors must be positive")
    
    if num_negatives <= 0:
        raise ValueError("Number of negatives must be positive")
    
    # Total pairs
    total = num_anchors + num_negatives
    
    # Weight inversely proportional to count
    # anchor_weight * num_anchors = negative_weight * num_negatives (for equal contribution)
    # anchor_weight + negative_weight = 1 (normalized)
    
    # Solving: anchor_weight = total / (2 * num_anchors)
    anchor_weight = 0.5 * total / num_anchors
    negative_weight = 0.5 * total / num_negatives
    
    # Normalize to sum to 1
    total_weight = anchor_weight + negative_weight
    anchor_weight /= total_weight
    negative_weight /= total_weight
    
    return anchor_weight, negative_weight


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
    
    def __init__(self, config: Dict):
        """
        Initialize the weight calculator.
        
        Args:
            config: Configuration dictionary with keys:
                - enabled (bool): Whether to apply class balancing
                - anchor_negative_balance (bool): Balance anchor vs negative pairs
                - per_class_balance (bool): Balance across classes
                - weighting_scheme (str): 'ins', 'isns', or 'ens'
                - ens_beta (float): Beta parameter for ENS scheme
        """
        self.enabled = config.get('enabled', False)
        self.anchor_negative_balance = config.get('anchor_negative_balance', True)
        self.per_class_balance = config.get('per_class_balance', True)
        self.weighting_scheme = config.get('weighting_scheme', 'ins')
        self.ens_beta = config.get('ens_beta', 0.9999)
        
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
        
        Args:
            labels: Array of labels (1.0 for anchors/positives, 0.0 for negatives)
            class_info: List of class identifiers for each sample
            
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
        
        # Apply anchor/negative balancing
        if self.anchor_negative_balance:
            anchor_weight, negative_weight = self._compute_anchor_negative_weights(labels)
            
            # Apply to each sample based on its label
            weights[labels == 1.0] *= anchor_weight
            weights[labels == 0.0] *= negative_weight
        
        # Apply per-class balancing
        if self.per_class_balance:
            class_weights = self._compute_class_weights(class_info)
            
            # Apply to each sample based on its class
            for i, cls in enumerate(class_info):
                weights[i] *= class_weights[cls]
        
        # Normalize weights to prevent loss scale issues
        # Keep mean at 1.0 so loss magnitude is comparable
        weights = weights / np.mean(weights)
        
        return weights
    
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
    
    def _compute_class_weights(
        self,
        class_info: List[str]
    ) -> Dict[str, float]:
        """Compute per-class weights from class information."""
        # Count samples per class
        class_counts = Counter(class_info)
        
        # Calculate weights using specified scheme
        return calculate_per_class_weights(
            class_counts,
            scheme=self.weighting_scheme,
            beta=self.ens_beta
        )
    
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

