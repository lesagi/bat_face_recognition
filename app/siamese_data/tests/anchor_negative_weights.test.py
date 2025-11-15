#!/usr/bin/env python3
"""
Test suite for anchor/negative weight calculations in Siamese network training.
Tests per-batch balancing of positive and negative pairs.
"""

import os
import sys
import unittest
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from siamese_data.anchor_negative_weights import (
    calculate_anchor_negative_weights,
    compute_batch_anchor_negative_weights,
)


class TestCalculateWeights(unittest.TestCase):
    """Test the core weight calculation function."""

    def test_equal_distribution(self):
        """Equal anchors and negatives should give equal weights."""
        anchor_weight, negative_weight = calculate_anchor_negative_weights(50, 50)
        
        self.assertAlmostEqual(anchor_weight, 0.5, places=5)
        self.assertAlmostEqual(negative_weight, 0.5, places=5)
        
    def test_more_negatives(self):
        """More negatives should give anchors higher weight."""
        anchor_weight, negative_weight = calculate_anchor_negative_weights(10, 90)
        
        # Anchors should get much higher weight
        self.assertGreater(anchor_weight, negative_weight)
        # Weights should sum to 1
        self.assertAlmostEqual(anchor_weight + negative_weight, 1.0, places=5)
        
    def test_more_anchors(self):
        """More anchors should give negatives higher weight."""
        anchor_weight, negative_weight = calculate_anchor_negative_weights(90, 10)
        
        # Negatives should get much higher weight
        self.assertGreater(negative_weight, anchor_weight)
        # Weights should sum to 1
        self.assertAlmostEqual(anchor_weight + negative_weight, 1.0, places=5)
        
    def test_extreme_imbalance(self):
        """Extreme imbalance should still produce valid weights."""
        anchor_weight, negative_weight = calculate_anchor_negative_weights(1, 99)
        
        # Anchor should get ~99% of total weight
        self.assertGreater(anchor_weight, 0.9)
        self.assertLess(negative_weight, 0.1)
        # Weights should sum to 1
        self.assertAlmostEqual(anchor_weight + negative_weight, 1.0, places=5)
        
    def test_weights_sum_to_one(self):
        """Weights should always sum to 1.0."""
        test_cases = [(10, 90), (50, 50), (90, 10), (33, 67), (1, 999)]
        
        for num_anchors, num_negatives in test_cases:
            anchor_w, negative_w = calculate_anchor_negative_weights(num_anchors, num_negatives)
            self.assertAlmostEqual(anchor_w + negative_w, 1.0, places=5,
                                  msg=f"Failed for {num_anchors} anchors, {num_negatives} negatives")
    
    def test_equal_contribution(self):
        """Weights should create equal contribution from both types."""
        test_cases = [(10, 90), (50, 50), (90, 10), (25, 75)]
        
        for num_anchors, num_negatives in test_cases:
            anchor_w, negative_w = calculate_anchor_negative_weights(num_anchors, num_negatives)
            
            # Contribution = weight * count
            anchor_contribution = anchor_w * num_anchors
            negative_contribution = negative_w * num_negatives
            
            # Should be approximately equal (50/50 split)
            self.assertAlmostEqual(anchor_contribution, negative_contribution, places=5,
                                  msg=f"Failed for {num_anchors} anchors, {num_negatives} negatives")


class TestValidation(unittest.TestCase):
    """Test validation and error handling."""

    def test_zero_anchors_raises_error(self):
        """Zero anchors should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_anchor_negative_weights(0, 100)
        
        self.assertIn("anchors must be positive", str(context.exception))
    
    def test_zero_negatives_raises_error(self):
        """Zero negatives should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_anchor_negative_weights(100, 0)
        
        self.assertIn("negatives must be positive", str(context.exception))
    
    def test_negative_anchors_raises_error(self):
        """Negative anchor count should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_anchor_negative_weights(-10, 100)
    
    def test_negative_negatives_raises_error(self):
        """Negative negative count should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_anchor_negative_weights(100, -10)


class TestBatchComputation(unittest.TestCase):
    """Test batch-level weight computation."""

    def test_balanced_batch(self):
        """Balanced batch should produce equal weights."""
        labels = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        anchor_w, negative_w = compute_batch_anchor_negative_weights(labels)
        
        self.assertAlmostEqual(anchor_w, 0.5, places=5)
        self.assertAlmostEqual(negative_w, 0.5, places=5)
    
    def test_imbalanced_batch(self):
        """Imbalanced batch should produce appropriate weights."""
        # 2 anchors, 8 negatives
        labels = np.array([1.0, 1.0] + [0.0] * 8)
        anchor_w, negative_w = compute_batch_anchor_negative_weights(labels)
        
        # Anchors should get higher weight
        self.assertGreater(anchor_w, negative_w)
        self.assertAlmostEqual(anchor_w + negative_w, 1.0, places=5)
    
    def test_single_anchor(self):
        """Batch with single anchor should work."""
        labels = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        anchor_w, negative_w = compute_batch_anchor_negative_weights(labels)
        
        # Should produce valid weights
        self.assertGreater(anchor_w, 0.5)
        self.assertAlmostEqual(anchor_w + negative_w, 1.0, places=5)
    
    def test_single_negative(self):
        """Batch with single negative should work."""
        labels = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        anchor_w, negative_w = compute_batch_anchor_negative_weights(labels)
        
        # Should produce valid weights
        self.assertGreater(negative_w, 0.5)
        self.assertAlmostEqual(anchor_w + negative_w, 1.0, places=5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_only_anchors(self):
        """Batch with only anchors should return (1.0, 0.0)."""
        labels = np.array([1.0, 1.0, 1.0, 1.0])
        anchor_w, negative_w = compute_batch_anchor_negative_weights(labels)
        
        self.assertEqual(anchor_w, 1.0)
        self.assertEqual(negative_w, 0.0)
    
    def test_only_negatives(self):
        """Batch with only negatives should return (0.0, 1.0)."""
        labels = np.array([0.0, 0.0, 0.0, 0.0])
        anchor_w, negative_w = compute_batch_anchor_negative_weights(labels)
        
        self.assertEqual(anchor_w, 0.0)
        self.assertEqual(negative_w, 1.0)
    
    def test_empty_batch_raises_error(self):
        """Empty batch should raise ValueError."""
        labels = np.array([])
        
        with self.assertRaises(ValueError) as context:
            compute_batch_anchor_negative_weights(labels)
        
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_large_batch(self):
        """Large batch should compute efficiently."""
        # 100 anchors, 900 negatives
        labels = np.array([1.0] * 100 + [0.0] * 900)
        anchor_w, negative_w = compute_batch_anchor_negative_weights(labels)
        
        # Should still produce valid weights
        self.assertGreater(anchor_w, negative_w)
        self.assertAlmostEqual(anchor_w + negative_w, 1.0, places=5)
    
    def test_float_labels(self):
        """Should work with float labels (not just 0.0/1.0)."""
        # NumPy comparison should handle close-to-integer floats
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        anchor_w, negative_w = compute_batch_anchor_negative_weights(labels)
        
        self.assertAlmostEqual(anchor_w + negative_w, 1.0, places=5)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of weight calculations."""

    def test_very_large_counts(self):
        """Very large counts should not cause overflow."""
        anchor_w, negative_w = calculate_anchor_negative_weights(1000000, 9000000)
        
        self.assertAlmostEqual(anchor_w + negative_w, 1.0, places=5)
        self.assertGreater(anchor_w, negative_w)
    
    def test_weights_always_positive(self):
        """Weights should always be positive."""
        test_cases = [(1, 99), (50, 50), (99, 1), (10, 1000)]
        
        for num_anchors, num_negatives in test_cases:
            anchor_w, negative_w = calculate_anchor_negative_weights(num_anchors, num_negatives)
            
            self.assertGreater(anchor_w, 0.0,
                              msg=f"Anchor weight not positive for {num_anchors} anchors")
            self.assertGreater(negative_w, 0.0,
                              msg=f"Negative weight not positive for {num_negatives} negatives")
    
    def test_consistency_across_calls(self):
        """Same inputs should produce same outputs."""
        anchor_w1, negative_w1 = calculate_anchor_negative_weights(30, 70)
        anchor_w2, negative_w2 = calculate_anchor_negative_weights(30, 70)
        
        self.assertEqual(anchor_w1, anchor_w2)
        self.assertEqual(negative_w1, negative_w2)


if __name__ == '__main__':
    unittest.main()

