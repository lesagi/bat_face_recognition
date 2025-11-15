"""
Anchor/Negative weight balancing for Siamese networks.

This module provides per-batch balancing of anchor (positive) and negative pairs
to ensure both types contribute equally to the loss function during training.
"""

import numpy as np
from typing import Tuple


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
        
    Example:
        >>> calculate_anchor_negative_weights(10, 90)
        (0.9, 0.1)  # Anchors get 9x weight since they're 1/10 of the data
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


def compute_batch_anchor_negative_weights(
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Compute anchor/negative weights for a batch of samples.
    
    This function counts the number of anchors (label=1.0) and negatives (label=0.0)
    in the batch and computes appropriate weights to balance their contribution.
    
    Args:
        labels: Array of labels (1.0 for anchors/positives, 0.0 for negatives)
        
    Returns:
        Tuple of (anchor_weight, negative_weight)
        
    Raises:
        ValueError: If labels array is empty
        ValueError: If batch contains only one type (handled with special cases)
        
    Example:
        >>> labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        >>> compute_batch_anchor_negative_weights(labels)
        (0.625, 0.375)  # 2 anchors, 3 negatives
    """
    if len(labels) == 0:
        raise ValueError("Labels array cannot be empty")
    
    # Count anchors and negatives
    num_anchors = int(np.sum(labels == 1.0))
    num_negatives = int(np.sum(labels == 0.0))
    
    # Handle edge cases
    if num_anchors == 0 and num_negatives > 0:
        # Only negatives in this batch - give them all the weight
        return 0.0, 1.0
    elif num_negatives == 0 and num_anchors > 0:
        # Only anchors in this batch - give them all the weight
        return 1.0, 0.0
    elif num_anchors == 0 and num_negatives == 0:
        # Empty batch (shouldn't happen, but handle gracefully)
        raise ValueError("Batch contains no valid labels (neither anchors nor negatives)")
    
    # Normal case: both types present
    return calculate_anchor_negative_weights(num_anchors, num_negatives)

