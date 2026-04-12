"""
Statistical Tests Module for Bat Face Recognition Project.

This module provides statistical testing capabilities for validating
model performance, including permutation tests for significance testing.
"""

from .permutation_test import PermutationTest, PermutationTestResults
from .permutation_trainer import PermutationTrainer, create_permutation_trainer
from .permutation_visualizer import PermutationVisualizer
from .inference_permutation_test import run_inference_permutation_test

__all__ = [
    "PermutationTest",
    "PermutationTestResults",
    "PermutationTrainer",
    "create_permutation_trainer",
    "PermutationVisualizer",
    "run_inference_permutation_test",
]
