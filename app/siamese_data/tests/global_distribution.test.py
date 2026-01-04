#!/usr/bin/env python3
"""
Test suite for global class distribution computation in Siamese network training.
Tests all three strategies: file_based, sampled, and full_scan.
"""

import os
import sys
import unittest
from unittest.mock import Mock, MagicMock
import tensorflow as tf

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from siamese_data.global_distribution import (
    GlobalDistributionStrategy,
    compute_global_class_distribution,
    _compute_from_files,
    _compute_from_sampling,
    _compute_from_full_scan,
    validate_distribution,
)


class TestStrategyEnum(unittest.TestCase):
    """Test the GlobalDistributionStrategy enum."""

    def test_enum_values(self):
        """Enum should have all three strategies."""
        self.assertEqual(GlobalDistributionStrategy.FILE_BASED, "file_based")
        self.assertEqual(GlobalDistributionStrategy.SAMPLED, "sampled")
        self.assertEqual(GlobalDistributionStrategy.FULL_SCAN, "full_scan")
    
    def test_enum_from_string(self):
        """Should be able to create enum from string."""
        self.assertEqual(GlobalDistributionStrategy("file_based"), GlobalDistributionStrategy.FILE_BASED)
        self.assertEqual(GlobalDistributionStrategy("sampled"), GlobalDistributionStrategy.SAMPLED)
        self.assertEqual(GlobalDistributionStrategy("full_scan"), GlobalDistributionStrategy.FULL_SCAN)


class TestStrategySelection(unittest.TestCase):
    """Test strategy selection and dispatch."""

    def test_invalid_strategy_raises_error(self):
        """Invalid strategy name should raise ValueError."""
        dataset = Mock()
        
        with self.assertRaises(ValueError) as context:
            compute_global_class_distribution(dataset, "invalid_strategy")
        
        self.assertIn("Invalid strategy", str(context.exception))
        self.assertIn("invalid_strategy", str(context.exception))
    
    def test_case_insensitive_strategy(self):
        """Strategy names should be case-insensitive."""
        # This will fail on missing params, but should pass strategy validation
        dataset = Mock()
        
        with self.assertRaises(ValueError) as context:
            compute_global_class_distribution(dataset, "FILE_BASED")
        
        # Should fail on missing data_splitter, not invalid strategy
        self.assertIn("data_splitter", str(context.exception))
    
    def test_file_based_requires_data_splitter(self):
        """file_based strategy should require data_splitter."""
        dataset = Mock()
        
        with self.assertRaises(ValueError) as context:
            compute_global_class_distribution(dataset, "file_based", data_splitter=None)
        
        self.assertIn("data_splitter", str(context.exception))
    
    def test_sampled_validates_percentage(self):
        """sampled strategy should validate sampling_percentage."""
        dataset = self._create_mock_dataset([])
        
        # Zero percentage
        with self.assertRaises(ValueError) as context:
            compute_global_class_distribution(dataset, "sampled", sampling_percentage=0.0)
        self.assertIn("sampling_percentage", str(context.exception))
        
        # Negative percentage
        with self.assertRaises(ValueError):
            compute_global_class_distribution(dataset, "sampled", sampling_percentage=-0.1)
        
        # Over 100%
        with self.assertRaises(ValueError):
            compute_global_class_distribution(dataset, "sampled", sampling_percentage=1.5)
    
    def _create_mock_dataset(self, class_data):
        """Helper to create a mock TensorFlow dataset."""
        if not class_data:
            class_data = [('A', 'A', 1.0, 'classA:1.0')]
        
        def generator():
            for item in class_data:
                yield item
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )
        return dataset


class TestFileBasedStrategy(unittest.TestCase):
    """Test file-based distribution computation."""

    def test_balanced_classes(self):
        """Balanced classes should produce similar counts."""
        data_splitter = Mock()
        data_splitter.train_class_files = {
            'classA': ['file1', 'file2', 'file3'],
            'classB': ['file4', 'file5', 'file6'],
            'classC': ['file7', 'file8', 'file9']
        }
        
        dist = _compute_from_files(data_splitter)
        
        # All classes should be present
        self.assertEqual(set(dist.keys()), {'classA', 'classB', 'classC'})
        
        # All counts should be positive
        for count in dist.values():
            self.assertGreater(count, 0)
    
    def test_imbalanced_classes(self):
        """Imbalanced classes should produce different counts."""
        data_splitter = Mock()
        data_splitter.train_class_files = {
            'classA': ['f1', 'f2'],  # 2 files
            'classB': ['f3', 'f4', 'f5', 'f6', 'f7']  # 5 files
        }
        
        dist = _compute_from_files(data_splitter)
        
        # Class with more files should have higher count
        self.assertGreater(dist['classB'], dist['classA'])
    
    def test_single_class(self):
        """Single class with multiple files should work."""
        data_splitter = Mock()
        data_splitter.train_class_files = {
            'classA': ['f1', 'f2', 'f3', 'f4']
        }
        
        dist = _compute_from_files(data_splitter)
        
        self.assertEqual(set(dist.keys()), {'classA'})
        self.assertGreater(dist['classA'], 0)
    
    def test_class_with_single_file(self):
        """Class with single file produces no anchor pairs."""
        data_splitter = Mock()
        data_splitter.train_class_files = {
            'classA': ['f1'],  # Only 1 file - no anchor pairs
            'classB': ['f2', 'f3']  # 2 files - can make pairs
        }
        
        dist = _compute_from_files(data_splitter)
        
        # ClassA should still appear due to negative pairs with classB
        self.assertIn('classA', dist)
        self.assertIn('classB', dist)
    
    def test_empty_classes_raises_error(self):
        """Empty class list should raise error."""
        data_splitter = Mock()
        data_splitter.train_class_files = {}
        
        with self.assertRaises(ValueError) as context:
            _compute_from_files(data_splitter)
        
        self.assertIn("No class distribution", str(context.exception))


class TestSampledStrategy(unittest.TestCase):
    """Test sampling-based distribution computation."""

    def _create_dataset_with_classes(self, class_counts):
        """Helper to create dataset with specified class distribution."""
        data = []
        for class_name, count in class_counts.items():
            for i in range(count):
                # Use PairClassInfo serialized format: "class_name:1.0"
                class_info_str = f'{class_name}:1.0'
                data.append((f'img1_{i}', f'img2_{i}', 1.0, class_info_str))
        
        def generator():
            for item in data:
                yield item
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )
        return dataset
    
    def test_balanced_sampling(self):
        """Sampling should preserve relative proportions."""
        class_counts = {'classA': 100, 'classB': 100, 'classC': 100}
        dataset = self._create_dataset_with_classes(class_counts)
        
        dist = _compute_from_sampling(dataset, sampling_percentage=0.2)
        
        # All classes should be present
        self.assertEqual(set(dist.keys()), {'classA', 'classB', 'classC'})
        
        # Counts should be similar (within reasonable variance)
        counts = list(dist.values())
        max_count = max(counts)
        min_count = min(counts)
        self.assertLess(max_count / min_count, 2.0)  # Less than 2x difference
    
    def test_imbalanced_sampling(self):
        """Sampling should reflect imbalanced distribution."""
        class_counts = {'classA': 30, 'classB': 300}
        dataset = self._create_dataset_with_classes(class_counts)
        
        dist = _compute_from_sampling(dataset, sampling_percentage=0.3)
        
        # ClassB should have more samples
        self.assertGreater(dist['classB'], dist['classA'])
    
    def test_small_percentage(self):
        """Small sampling percentage should still work."""
        class_counts = {'classA': 100, 'classB': 200}
        dataset = self._create_dataset_with_classes(class_counts)
        
        dist = _compute_from_sampling(dataset, sampling_percentage=0.05)
        
        # Should still capture both classes
        self.assertEqual(set(dist.keys()), {'classA', 'classB'})
    
    def test_large_percentage(self):
        """Large sampling percentage should be more accurate."""
        class_counts = {'classA': 100, 'classB': 200}
        dataset = self._create_dataset_with_classes(class_counts)
        
        dist = _compute_from_sampling(dataset, sampling_percentage=0.5)
        
        # Ratio should be close to 1:2
        ratio = dist['classB'] / dist['classA']
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.5)
    
    def test_small_dataset(self):
        """Small dataset should work without errors."""
        class_counts = {'classA': 5, 'classB': 5}
        dataset = self._create_dataset_with_classes(class_counts)
        
        dist = _compute_from_sampling(dataset, sampling_percentage=0.5)
        
        self.assertIn('classA', dist)
        self.assertIn('classB', dist)
    
    def test_empty_dataset_raises_error(self):
        """Empty dataset should raise error."""
        dataset = self._create_dataset_with_classes({})
        
        with self.assertRaises(ValueError) as context:
            _compute_from_sampling(dataset, sampling_percentage=0.1)
        
        self.assertIn("No samples found", str(context.exception))


class TestFullScanStrategy(unittest.TestCase):
    """Test full scan distribution computation."""

    def _create_dataset_with_classes(self, class_counts):
        """Helper to create dataset with specified class distribution."""
        data = []
        for class_name, count in class_counts.items():
            for i in range(count):
                # Use PairClassInfo serialized format: "class_name:1.0"
                class_info_str = f'{class_name}:1.0'
                data.append((f'img1_{i}', f'img2_{i}', 1.0, class_info_str))
        
        def generator():
            for item in data:
                yield item
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )
        return dataset
    
    def test_accurate_counts(self):
        """Full scan should produce exact counts."""
        class_counts = {'classA': 50, 'classB': 150, 'classC': 100}
        dataset = self._create_dataset_with_classes(class_counts)
        
        dist = _compute_from_full_scan(dataset)
        
        # Counts should be exact
        self.assertEqual(dist['classA'], 50)
        self.assertEqual(dist['classB'], 150)
        self.assertEqual(dist['classC'], 100)
    
    def test_single_class(self):
        """Single class should work."""
        class_counts = {'classA': 100}
        dataset = self._create_dataset_with_classes(class_counts)
        
        dist = _compute_from_full_scan(dataset)
        
        self.assertEqual(dist['classA'], 100)
    
    def test_many_classes(self):
        """Many classes should all be counted."""
        class_counts = {f'class{i}': 10 for i in range(20)}
        dataset = self._create_dataset_with_classes(class_counts)
        
        dist = _compute_from_full_scan(dataset)
        
        # All 20 classes should be present
        self.assertEqual(len(dist), 20)
        
        # All should have count of 10
        for count in dist.values():
            self.assertEqual(count, 10)
    
    def test_empty_dataset_raises_error(self):
        """Empty dataset should raise error."""
        dataset = self._create_dataset_with_classes({})
        
        with self.assertRaises(ValueError) as context:
            _compute_from_full_scan(dataset)
        
        self.assertIn("No samples found", str(context.exception))


class TestDistributionValidation(unittest.TestCase):
    """Test distribution validation."""

    def test_valid_distribution(self):
        """Valid distribution should not raise error."""
        dist = {'classA': 100, 'classB': 200}
        validate_distribution(dist)  # Should not raise
    
    def test_empty_distribution_raises_error(self):
        """Empty distribution should raise error."""
        with self.assertRaises(ValueError) as context:
            validate_distribution({})
        
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_non_string_class_name_raises_error(self):
        """Non-string class name should raise error."""
        dist = {123: 100}
        
        with self.assertRaises(ValueError) as context:
            validate_distribution(dist)
        
        self.assertIn("must be string", str(context.exception))
    
    def test_zero_count_raises_error(self):
        """Zero count should raise error."""
        dist = {'classA': 0}
        
        with self.assertRaises(ValueError) as context:
            validate_distribution(dist)
        
        self.assertIn("must be positive", str(context.exception))
    
    def test_negative_count_raises_error(self):
        """Negative count should raise error."""
        dist = {'classA': -10}
        
        with self.assertRaises(ValueError):
            validate_distribution(dist)


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def _create_dataset(self, class_counts):
        """Helper to create a realistic dataset."""
        data = []
        for class_name, count in class_counts.items():
            for i in range(count):
                # Use PairClassInfo serialized format: "class_name:1.0"
                class_info_str = f'{class_name}:1.0'
                data.append((f'img1_{class_name}_{i}', f'img2_{class_name}_{i}', 1.0, class_info_str))
        
        def generator():
            for item in data:
                yield item
        
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )
    
    def test_sampled_strategy_e2e(self):
        """End-to-end test of sampled strategy."""
        class_counts = {'dog': 100, 'cat': 200, 'bird': 150}
        dataset = self._create_dataset(class_counts)
        
        dist = compute_global_class_distribution(
            dataset=dataset,
            strategy='sampled',
            sampling_percentage=0.2
        )
        
        # All classes should be present
        self.assertEqual(set(dist.keys()), {'dog', 'cat', 'bird'})
        
        # All counts should be positive
        for count in dist.values():
            self.assertGreater(count, 0)
    
    def test_full_scan_strategy_e2e(self):
        """End-to-end test of full_scan strategy."""
        class_counts = {'A': 50, 'B': 75, 'C': 25}
        dataset = self._create_dataset(class_counts)
        
        dist = compute_global_class_distribution(
            dataset=dataset,
            strategy='full_scan'
        )
        
        # Should match exact counts
        self.assertEqual(dist['A'], 50)
        self.assertEqual(dist['B'], 75)
        self.assertEqual(dist['C'], 25)
    
    def test_file_based_strategy_e2e(self):
        """End-to-end test of file_based strategy."""
        data_splitter = Mock()
        data_splitter.train_class_files = {
            'red': ['r1', 'r2', 'r3'],
            'blue': ['b1', 'b2', 'b3', 'b4']
        }
        
        dataset = Mock()  # Not used for file_based
        
        dist = compute_global_class_distribution(
            dataset=dataset,
            strategy='file_based',
            data_splitter=data_splitter
        )
        
        # Both classes should be present
        self.assertIn('red', dist)
        self.assertIn('blue', dist)
        
        # Blue has more files, should have higher count
        self.assertGreater(dist['blue'], dist['red'])


if __name__ == '__main__':
    unittest.main()

