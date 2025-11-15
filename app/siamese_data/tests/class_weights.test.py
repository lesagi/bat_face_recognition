#!/usr/bin/env python3
"""
Test suite for class weight calculations in Siamese network training.
Following TDD approach - tests written before implementation.
"""

import os
import sys
import unittest
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from siamese_data.class_weights import (
    calculate_ins_weights,
    calculate_isns_weights,
    calculate_ens_weights,
    calculate_anchor_negative_weights,
    calculate_per_class_weights,
    ClassWeightCalculator,
)


class TestWeightingSchemes(unittest.TestCase):
    """Test individual weighting schemes (INS, ISNS, ENS)."""

    def test_ins_equal_distribution(self):
        """INS: Equal class distribution should give equal weights."""
        class_counts = {'A': 100, 'B': 100, 'C': 100}
        weights = calculate_ins_weights(class_counts)
        
        # All weights should be equal
        self.assertAlmostEqual(weights['A'], weights['B'], places=5)
        self.assertAlmostEqual(weights['B'], weights['C'], places=5)
        
        # Each should be approximately 1/3 (normalized)
        self.assertAlmostEqual(weights['A'], 1.0/3, places=5)

    def test_ins_imbalanced_distribution(self):
        """INS: Minority classes should get higher weights."""
        class_counts = {'A': 10, 'B': 100, 'C': 1000}
        weights = calculate_ins_weights(class_counts)
        
        # Minority class should have highest weight
        self.assertGreater(weights['A'], weights['B'])
        self.assertGreater(weights['B'], weights['C'])
        
        # Weights should sum to 1 (normalized)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

    def test_ins_single_class(self):
        """INS: Single class should get weight of 1."""
        class_counts = {'A': 100}
        weights = calculate_ins_weights(class_counts)
        
        self.assertAlmostEqual(weights['A'], 1.0, places=5)

    def test_isns_equal_distribution(self):
        """ISNS: Equal class distribution should give equal weights."""
        class_counts = {'A': 100, 'B': 100, 'C': 100}
        weights = calculate_isns_weights(class_counts)
        
        # All weights should be equal
        self.assertAlmostEqual(weights['A'], weights['B'], places=5)
        self.assertAlmostEqual(weights['B'], weights['C'], places=5)

    def test_isns_gentler_than_ins(self):
        """ISNS: Should be gentler (less extreme) than INS."""
        class_counts = {'A': 10, 'B': 1000}
        
        ins_weights = calculate_ins_weights(class_counts)
        isns_weights = calculate_isns_weights(class_counts)
        
        # ISNS ratio should be less extreme than INS ratio
        ins_ratio = ins_weights['A'] / ins_weights['B']
        isns_ratio = isns_weights['A'] / isns_weights['B']
        
        self.assertGreater(ins_ratio, isns_ratio)
        self.assertGreater(isns_ratio, 1.0)  # Still favors minority

    def test_ens_with_different_betas(self):
        """ENS: Different beta values should produce different weights."""
        class_counts = {'A': 10, 'B': 100, 'C': 1000}
        
        weights_09 = calculate_ens_weights(class_counts, beta=0.9)
        weights_099 = calculate_ens_weights(class_counts, beta=0.99)
        weights_0999 = calculate_ens_weights(class_counts, beta=0.999)
        weights_09999 = calculate_ens_weights(class_counts, beta=0.9999)
        
        # Higher beta should give more weight to minority classes
        self.assertGreater(weights_09999['A'], weights_0999['A'])
        self.assertGreater(weights_0999['A'], weights_099['A'])
        self.assertGreater(weights_099['A'], weights_09['A'])

    def test_ens_normalization(self):
        """ENS: Weights should sum to 1."""
        class_counts = {'A': 50, 'B': 200, 'C': 500}
        weights = calculate_ens_weights(class_counts, beta=0.9999)
        
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

    def test_ens_extreme_imbalance(self):
        """ENS: Should handle extreme imbalance well."""
        class_counts = {'A': 5, 'B': 5000}
        weights = calculate_ens_weights(class_counts, beta=0.9999)
        
        # Minority should get much higher weight
        self.assertGreater(weights['A'], weights['B'])
        
        # But not infinite
        self.assertLess(weights['A'], 1.0)


class TestAnchorNegativeWeights(unittest.TestCase):
    """Test anchor vs negative pair weighting."""

    def test_equal_anchors_negatives(self):
        """Equal number of anchors and negatives should give equal weights."""
        anchor_weight, negative_weight = calculate_anchor_negative_weights(1000, 1000)
        
        self.assertAlmostEqual(anchor_weight, negative_weight, places=5)
        self.assertAlmostEqual(anchor_weight, 0.5, places=5)

    def test_more_negatives(self):
        """More negatives should give anchors higher weight."""
        anchor_weight, negative_weight = calculate_anchor_negative_weights(1000, 9000)
        
        # Anchor weight should be higher
        self.assertGreater(anchor_weight, negative_weight)
        
        # With 1:9 ratio, anchor should get ~9x weight
        ratio = anchor_weight / negative_weight
        self.assertAlmostEqual(ratio, 9.0, places=1)

    def test_more_anchors(self):
        """More anchors should give negatives higher weight."""
        anchor_weight, negative_weight = calculate_anchor_negative_weights(9000, 1000)
        
        # Negative weight should be higher
        self.assertGreater(negative_weight, anchor_weight)

    def test_extreme_imbalance(self):
        """Extreme imbalance (e.g., 100:1) should be handled."""
        anchor_weight, negative_weight = calculate_anchor_negative_weights(100, 10000)
        
        self.assertGreater(anchor_weight, negative_weight)
        self.assertGreater(anchor_weight, 0.0)
        self.assertLess(anchor_weight, 1.0)

    def test_zero_anchors_raises_error(self):
        """Zero anchors should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_anchor_negative_weights(0, 1000)

    def test_zero_negatives_raises_error(self):
        """Zero negatives should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_anchor_negative_weights(1000, 0)


class TestPerClassWeights(unittest.TestCase):
    """Test per-class weight calculation with different schemes."""

    def test_per_class_weights_ins(self):
        """Per-class weights using INS scheme."""
        class_counts = {'A': 100, 'B': 200, 'C': 300}
        weights = calculate_per_class_weights(class_counts, scheme='ins')
        
        # Should use INS formula
        self.assertGreater(weights['A'], weights['B'])
        self.assertGreater(weights['B'], weights['C'])

    def test_per_class_weights_isns(self):
        """Per-class weights using ISNS scheme."""
        class_counts = {'A': 100, 'B': 400}
        weights = calculate_per_class_weights(class_counts, scheme='isns')
        
        # Should use ISNS formula (square root)
        # A has 100, B has 400 -> sqrt ratio is 2:1
        ratio = weights['A'] / weights['B']
        self.assertAlmostEqual(ratio, 2.0, places=1)

    def test_per_class_weights_ens(self):
        """Per-class weights using ENS scheme."""
        class_counts = {'A': 50, 'B': 500}
        weights = calculate_per_class_weights(class_counts, scheme='ens', beta=0.9999)
        
        # Should use ENS formula
        self.assertGreater(weights['A'], weights['B'])

    def test_invalid_scheme_raises_error(self):
        """Invalid scheme should raise ValueError."""
        class_counts = {'A': 100}
        with self.assertRaises(ValueError):
            calculate_per_class_weights(class_counts, scheme='invalid')

    def test_empty_class_counts_raises_error(self):
        """Empty class counts should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_per_class_weights({}, scheme='ins')


class TestClassWeightCalculator(unittest.TestCase):
    """Test the main ClassWeightCalculator class."""

    def test_calculator_initialization(self):
        """Calculator should initialize with config and global distribution."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
            'ens_beta': 0.9999,
        }
        
        # Provide global distribution when per_class_balance is True
        global_dist = {'A': 100, 'B': 200, 'C': 150}
        
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        self.assertTrue(calculator.enabled)
        self.assertEqual(calculator.weighting_scheme, 'ins')

    def test_calculator_disabled(self):
        """Disabled calculator should return uniform weights."""
        config = {
            'enabled': False,
        }
        
        calculator = ClassWeightCalculator(config)
        
        # Mock data: 3 samples, 2 anchors (label=1), 1 negative (label=0)
        labels = np.array([1.0, 1.0, 0.0])
        class_info = ['A', 'B', 'A']
        
        weights = calculator.compute_sample_weights(labels, class_info)
        
        # Should return all ones when disabled
        np.testing.assert_array_almost_equal(weights, np.ones(3))

    def test_calculator_anchor_negative_only(self):
        """Calculator with only anchor/negative balancing."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': False,
            'weighting_scheme': 'ins',
        }
        
        calculator = ClassWeightCalculator(config)
        
        # 2 anchors, 8 negatives (1:4 ratio)
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        class_info = ['A'] * 10
        
        weights = calculator.compute_sample_weights(labels, class_info)
        
        # Anchors should have higher weight
        anchor_weights = weights[labels == 1.0]
        negative_weights = weights[labels == 0.0]
        
        self.assertGreater(anchor_weights[0], negative_weights[0])

    def test_calculator_per_class_only(self):
        """Calculator with only per-class balancing."""
        config = {
            'enabled': True,
            'anchor_negative_balance': False,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        # Provide global distribution reflecting class imbalance
        global_dist = {'A': 500, 'B': 100}  # A has 5x more samples
        
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        
        # Class A: 5 samples, Class B: 1 sample
        labels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        class_info = ['A', 'A', 'A', 'A', 'A', 'B']
        
        weights = calculator.compute_sample_weights(labels, class_info)
        
        # Class B (minority) should have higher weight
        self.assertGreater(weights[5], weights[0])

    def test_calculator_combined_weighting(self):
        """Calculator with both anchor/negative and per-class balancing."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        # Provide global distribution
        global_dist = {'A': 200, 'B': 300, 'C': 100}
        
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        
        # Mixed scenario
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        class_info = ['A', 'A', 'B', 'B', 'C']
        
        weights = calculator.compute_sample_weights(labels, class_info)
        
        # All weights should be positive
        self.assertTrue(np.all(weights > 0))
        
        # Weights should not all be equal (due to balancing)
        self.assertGreater(np.std(weights), 0.01)

    def test_calculator_compute_statistics(self):
        """Calculator should compute weight statistics."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        # Provide global distribution
        global_dist = {'A': 150, 'B': 200, 'C': 100}
        
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        class_info = ['A', 'A', 'B', 'B', 'C']
        
        weights = calculator.compute_sample_weights(labels, class_info)
        stats = calculator.compute_weight_statistics(weights, labels)
        
        # Should return dictionary with statistics
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('anchor_mean', stats)
        self.assertIn('negative_mean', stats)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_negative_class_counts(self):
        """Negative class counts should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_ins_weights({'A': -10, 'B': 100})

    def test_zero_class_counts(self):
        """Zero class counts should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_ins_weights({'A': 0, 'B': 100})

    def test_very_small_counts(self):
        """Very small counts (e.g., 1 sample) should work."""
        class_counts = {'A': 1, 'B': 100}
        weights = calculate_ins_weights(class_counts)
        
        # Should not crash and should heavily favor minority
        self.assertGreater(weights['A'], weights['B'])

    def test_very_large_counts(self):
        """Very large counts should work without overflow."""
        class_counts = {'A': 1000000, 'B': 10000000}
        weights = calculate_ins_weights(class_counts)
        
        # Should not crash
        self.assertGreater(weights['A'], weights['B'])

    def test_ens_beta_boundary_values(self):
        """ENS with beta at boundaries (0, 1) should handle gracefully."""
        class_counts = {'A': 10, 'B': 100}
        
        # Beta = 0 should work (becomes uniform)
        weights_0 = calculate_ens_weights(class_counts, beta=0.0)
        self.assertIsNotNone(weights_0)
        
        # Beta = 0.99999 (very close to 1) should work
        weights_high = calculate_ens_weights(class_counts, beta=0.99999)
        self.assertIsNotNone(weights_high)

    def test_mismatched_labels_class_info_length(self):
        """Mismatched lengths should raise ValueError."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        # Provide global distribution
        global_dist = {'A': 100, 'B': 200, 'C': 150}
        
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        
        labels = np.array([1.0, 0.0])
        class_info = ['A', 'B', 'C']  # Wrong length
        
        with self.assertRaises(ValueError):
            calculator.compute_sample_weights(labels, class_info)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests with realistic scenarios."""

    def test_realistic_scenario_moderate_imbalance(self):
        """Realistic scenario: 5 classes, moderate imbalance."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        # Provide global distribution reflecting moderate imbalance
        global_dist = {'A': 200, 'B': 300, 'C': 100, 'D': 250, 'E': 150}
        
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        
        # Simulate real data: different class sizes
        # Class A: 20 pairs, Class B: 30 pairs, Class C: 10 pairs
        # 30 anchors, 30 negatives
        labels = np.concatenate([
            np.ones(30),  # 30 anchor pairs
            np.zeros(30),  # 30 negative pairs
        ])
        
        class_info = (
            ['A'] * 10 + ['B'] * 15 + ['C'] * 5 +  # Anchors
            ['A'] * 10 + ['B'] * 15 + ['C'] * 5   # Negatives
        )
        
        weights = calculator.compute_sample_weights(labels, class_info)
        
        # Verify properties
        self.assertEqual(len(weights), 60)
        self.assertTrue(np.all(weights > 0))
        
        # Minority class C should get higher weights
        class_c_weights = weights[np.array([i for i, c in enumerate(class_info) if c == 'C'])]
        class_b_weights = weights[np.array([i for i, c in enumerate(class_info) if c == 'B'])]
        
        self.assertGreater(np.mean(class_c_weights), np.mean(class_b_weights))

    def test_realistic_scenario_extreme_imbalance(self):
        """Realistic scenario: extreme anchor/negative imbalance (10:1)."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': False,
            'weighting_scheme': 'ins',
        }
        
        calculator = ClassWeightCalculator(config)
        
        # 10 anchors, 100 negatives
        labels = np.concatenate([
            np.ones(10),
            np.zeros(100),
        ])
        
        class_info = ['A'] * 110
        
        weights = calculator.compute_sample_weights(labels, class_info)
        
        # Check effective contribution
        anchor_contribution = np.sum(weights[labels == 1.0])
        negative_contribution = np.sum(weights[labels == 0.0])
        
        # Should be roughly equal (balanced)
        ratio = anchor_contribution / negative_contribution
        self.assertAlmostEqual(ratio, 1.0, places=1)


class TestStrictValidation(unittest.TestCase):
    """Test strict validation requirements for global class distribution."""

    def test_per_class_balance_without_global_dist_raises_error(self):
        """per_class_balance=True without global distribution should raise ValueError."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        with self.assertRaises(ValueError) as context:
            ClassWeightCalculator(config, global_class_distribution=None)
        
        # Check error message is informative
        error_msg = str(context.exception)
        self.assertIn("per_class_balance", error_msg)
        self.assertIn("global_class_distribution", error_msg)
        self.assertIn("global_distribution_strategy", error_msg)
    
    def test_anchor_negative_balance_works_without_global_dist(self):
        """anchor_negative_balance should work without global distribution."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': False,  # Disabled
            'weighting_scheme': 'ins',
        }
        
        # Should not raise error
        calculator = ClassWeightCalculator(config, global_class_distribution=None)
        
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        class_info = ['A', 'A', 'B', 'B', 'C']
        
        weights = calculator.compute_sample_weights(labels, class_info)
        self.assertEqual(len(weights), 5)
    
    def test_disabled_calculator_works_without_global_dist(self):
        """Disabled calculator should work without global distribution."""
        config = {
            'enabled': False,
        }
        
        # Should not raise error
        calculator = ClassWeightCalculator(config, global_class_distribution=None)
        
        labels = np.array([1.0, 0.0])
        class_info = ['A', 'B']
        
        weights = calculator.compute_sample_weights(labels, class_info)
        np.testing.assert_array_equal(weights, np.ones(2))
    
    def test_error_message_suggests_solutions(self):
        """Error message should suggest configuration options."""
        config = {
            'enabled': True,
            'per_class_balance': True,
        }
        
        with self.assertRaises(ValueError) as context:
            ClassWeightCalculator(config)
        
        error_msg = str(context.exception)
        # Should mention all three strategies
        self.assertIn("file_based", error_msg)
        self.assertIn("sampled", error_msg)
        self.assertIn("full_scan", error_msg)


class TestGlobalClassWeights(unittest.TestCase):
    """Test global class weight behavior."""

    def test_weights_stable_across_batches(self):
        """Global weights should remain stable across different batches."""
        config = {
            'enabled': True,
            'anchor_negative_balance': False,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        # Global distribution: A=100, B=300, C=600
        global_dist = {'A': 100, 'B': 300, 'C': 600}
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        
        # Batch 1: Mostly class A
        labels1 = np.array([1.0] * 5)
        class_info1 = ['A', 'A', 'A', 'A', 'A']
        weights1 = calculator.compute_sample_weights(labels1, class_info1)
        
        # Batch 2: Mostly class C
        labels2 = np.array([1.0] * 5)
        class_info2 = ['C', 'C', 'C', 'C', 'C']
        weights2 = calculator.compute_sample_weights(labels2, class_info2)
        
        # Class A samples should always get higher weight than class C
        # regardless of batch composition
        self.assertGreater(weights1[0], weights2[0])
    
    def test_class_info_used_for_weight_application(self):
        """class_info should be used for weight application, not counting."""
        config = {
            'enabled': True,
            'anchor_negative_balance': False,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        # Global distribution: A=100, B=200
        global_dist = {'A': 100, 'B': 200}
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        
        # Batch with uneven class distribution (different from global)
        labels = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        class_info = ['A', 'B', 'B', 'B', 'B']  # 1 A, 4 B in batch
        
        weights = calculator.compute_sample_weights(labels, class_info)
        
        # A sample should get higher weight (based on global dist, not batch dist)
        self.assertGreater(weights[0], weights[1])
    
    def test_hybrid_approach_combines_both(self):
        """Hybrid approach should combine per-batch anchor/negative + global per-class."""
        config = {
            'enabled': True,
            'anchor_negative_balance': True,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        # Global distribution: A=100, B=300
        global_dist = {'A': 100, 'B': 300}
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        
        # Batch: 2 anchors from A, 8 negatives from B
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        class_info = ['A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
        
        weights = calculator.compute_sample_weights(labels, class_info)
        
        # Anchors should get higher weight (per-batch balancing)
        anchor_weight = weights[0]
        negative_weight = weights[2]
        self.assertGreater(anchor_weight, negative_weight)
        
        # Class A should also get higher weight (global per-class balancing)
        # Both effects should be present
        self.assertGreater(weights[0], weights[2])
    
    def test_set_global_distribution_after_init(self):
        """set_global_class_distribution should allow lazy initialization."""
        config = {
            'enabled': True,
            'anchor_negative_balance': False,
            'per_class_balance': False,  # Start with it disabled
        }
        
        calculator = ClassWeightCalculator(config)
        
        # Now set global distribution
        global_dist = {'A': 100, 'B': 200}
        calculator.set_global_class_distribution(global_dist)
        
        # Should be stored
        self.assertIsNotNone(calculator._global_class_distribution)
        self.assertEqual(calculator._global_class_distribution, global_dist)
    
    def test_set_global_distribution_validates_input(self):
        """set_global_class_distribution should validate input."""
        config = {'enabled': False}
        calculator = ClassWeightCalculator(config)
        
        # Empty distribution should raise error
        with self.assertRaises(ValueError):
            calculator.set_global_class_distribution({})
        
        # Invalid count should raise error
        with self.assertRaises(ValueError):
            calculator.set_global_class_distribution({'A': 0})
        
        with self.assertRaises(ValueError):
            calculator.set_global_class_distribution({'A': -10})
    
    def test_weights_use_global_not_batch_counts(self):
        """Weights should be computed from global distribution, not batch counts."""
        config = {
            'enabled': True,
            'anchor_negative_balance': False,
            'per_class_balance': True,
            'weighting_scheme': 'ins',
        }
        
        # Global distribution: A=100, B=300 (B has 3x more)
        global_dist = {'A': 100, 'B': 300}
        calculator = ClassWeightCalculator(config, global_class_distribution=global_dist)
        
        # Batch 1: Mostly A (opposite of global)
        labels1 = np.array([1.0] * 4)
        class_info1 = ['A', 'A', 'A', 'B']
        weights1 = calculator.compute_sample_weights(labels1, class_info1)
        
        # Batch 2: Mostly B (matches global)
        labels2 = np.array([1.0] * 4)
        class_info2 = ['A', 'B', 'B', 'B']
        weights2 = calculator.compute_sample_weights(labels2, class_info2)
        
        # A should always get higher weight regardless of batch composition
        # Batch 1: A weight
        a_weight_batch1 = weights1[0]
        # Batch 2: A weight
        a_weight_batch2 = weights2[0]
        
        # Weights for class A should be similar across batches
        # (they're normalized differently but relative to B should be consistent)
        b_weight_batch1 = weights1[3]
        b_weight_batch2 = weights2[1]
        
        ratio1 = a_weight_batch1 / b_weight_batch1
        ratio2 = a_weight_batch2 / b_weight_batch2
        
        # Ratios should be similar (stable based on global dist)
        self.assertAlmostEqual(ratio1, ratio2, places=1)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

