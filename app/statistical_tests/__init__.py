"""
Statistical Tests Module for Bat Face Recognition Project.

This module provides statistical testing capabilities for validating
model performance, including permutation tests for significance testing.
"""

from .permutation_test import PermutationTest, PermutationTestResults
from .permutation_trainer import PermutationTrainer, create_permutation_trainer
from .permutation_visualizer import PermutationVisualizer

__all__ = [
    "PermutationTest", 
    "PermutationTestResults",
    "PermutationTrainer",
    "create_permutation_trainer",
    "PermutationVisualizer"
]
