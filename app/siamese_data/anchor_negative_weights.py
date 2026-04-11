"""
Anchor/Negative weight balancing for Siamese networks.

This module provides per-batch balancing of anchor (positive) and negative pairs
to ensure both types contribute equally to the loss function during training.
"""

import numpy as np
from typing import Tuple


def calculate_anchor_negative_weights(
    num_anchors: int, 
    num_negatives: int,
    target_ratio: float = 0.5,
) -> Tuple[float, float]:
    """
    Calculate weights to balance anchor (positive) and negative pairs.
    
    With target_ratio=0.5 (default), both types contribute equally.
    Higher values (e.g. 0.7) make anchors contribute 70% of total loss,
    penalizing false negatives more heavily.
    
    Args:
        num_anchors: Number of anchor (positive) pairs
        num_negatives: Number of negative pairs
        target_ratio: Desired fraction of total loss from anchor pairs.
            0.5 = equal contribution (default), 0.7 = 70/30 favoring anchors.
        
    Returns:
        Tuple of (anchor_weight, negative_weight)
        
    Raises:
        ValueError: If either count is zero or negative
        ValueError: If target_ratio is not in (0, 1)
        
    Example:
        >>> calculate_anchor_negative_weights(10, 90, target_ratio=0.5)
        (0.9, 0.1)  # Anchors get 9x weight since they're 1/10 of the data
    """
    if num_anchors <= 0:
        raise ValueError("Number of anchors must be positive")
    
    if num_negatives <= 0:
        raise ValueError("Number of negatives must be positive")
    
    if not (0 < target_ratio < 1):
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")
    
    total = num_anchors + num_negatives
    
    anchor_weight = target_ratio * total / num_anchors
    negative_weight = (1.0 - target_ratio) * total / num_negatives
    
    # Normalize to sum to 1
    total_weight = anchor_weight + negative_weight
    anchor_weight /= total_weight
    negative_weight /= total_weight
    
    return anchor_weight, negative_weight


def compute_batch_anchor_negative_weights(
    labels: np.ndarray,
    target_ratio: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute anchor/negative weights for a batch of samples.
    
    Args:
        labels: Array of labels (1.0 for anchors/positives, 0.0 for negatives)
        target_ratio: Desired fraction of total loss from anchor pairs (default 0.5).
        
    Returns:
        Tuple of (anchor_weight, negative_weight)
        
    Raises:
        ValueError: If labels array is empty
    """
    if len(labels) == 0:
        raise ValueError("Labels array cannot be empty")
    
    num_anchors = int(np.sum(labels == 1.0))
    num_negatives = int(np.sum(labels == 0.0))
    
    if num_anchors == 0 and num_negatives > 0:
        return 0.0, 1.0
    elif num_negatives == 0 and num_anchors > 0:
        return 1.0, 0.0
    elif num_anchors == 0 and num_negatives == 0:
        raise ValueError("Batch contains no valid labels (neither anchors nor negatives)")
    
    return calculate_anchor_negative_weights(num_anchors, num_negatives, target_ratio)

