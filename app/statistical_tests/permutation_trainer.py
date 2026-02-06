"""
Lightweight Trainer for Permutation Test Iterations.

This module provides a streamlined version of SiameseNetworkTrainer optimized
for running many permutation iterations efficiently. Key optimizations:
- Disabled MLflow logging
- Disabled checkpoint saving
- Disabled post-training automation (saliency maps, predictions)
- Reduced console output
- Returns only final metrics
"""

import os
import sys
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Store for later verbose output
        _GPU_COUNT = len(gpus)
        _GPU_NAMES = [gpu.name for gpu in gpus]
    except RuntimeError:
        _GPU_COUNT = 0
        _GPU_NAMES = []
else:
    _GPU_COUNT = 0
    _GPU_NAMES = []

from siamese_core.network import SiameseNetwork, L1Dist
from siamese_data.data_splitter import SiameseNetworkTrainingDataSplitter
from siamese_data.class_weights import ClassWeightCalculator
from siamese_data.global_distribution import compute_global_class_distribution
from config.loader import load_config


class PermutationTrainer:
    def __init__(
        self,
        bat_type: str = 'r',
        augmented_data: bool = False,
        data_source: str = 'video',
        permute_labels: bool = True,
        num_epochs: int = 10,
        mlflow_enabled: bool = False,
        verbose: bool = False,
        sample_fraction: float = 1.0,
        shuffle_buffer_fraction: float = 1.0
    ):
        cfg = load_config()
        sn_train = cfg.siamese_network.training
        
        self.permute_labels = permute_labels
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.mlflow_enabled = mlflow_enabled  # Typically False for permutation tests
        self.sample_fraction = sample_fraction
        self.shuffle_buffer_fraction = shuffle_buffer_fraction
        
        # GPU status logging
        if self.verbose:
            print(f"GPU(s) available: {_GPU_COUNT}")
            if _GPU_COUNT > 0:
                print(f"  Using: {_GPU_NAMES}")
            print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        
        # Mixed Precision Setup (silent)
        self.mixed_precision_enabled = cfg.siamese_network.advanced.get("mixed_precision", False)
        if self.mixed_precision_enabled:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        if self.verbose:
            print(f"Mixed precision: {self.mixed_precision_enabled}")
        
        # Get bat-type-specific input paths
        bat_key = 'mauritius' if bat_type == 'm' else 'rousettus'
        input_paths = cfg.siamese_network.input_paths[bat_key]
        self.input_dir = input_paths.get("random_bg_input")
        
        if not self.input_dir or not os.path.exists(self.input_dir):
            raise ValueError(f"Invalid or missing training input_dir: {self.input_dir}")
        
        # Training hyperparameters
        self.batch_size = sn_train.get("batch_size", 16)
        
        # Model, optimizer, loss
        learning_rate = sn_train.get("learning_rate", 1e-4)
        self.siamese_model = SiameseNetwork(L1Dist()).model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        if self.mixed_precision_enabled:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
        
        self.loss_function = tf.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.test_loss_function = tf.losses.BinaryCrossentropy()
        
        # Data loading
        training_portion = sn_train.get("train_val_split", 0.7)
        pair_mode = sn_train.get("pair_mode", "permutation")
        
        self.data_splitter = SiameseNetworkTrainingDataSplitter(
            [self.input_dir], 
            training_portion=training_portion, 
            mode=pair_mode, 
            permute_labels=self.permute_labels
        )
        
        train_data = self.data_splitter.train_data
        test_data = self.data_splitter.test_data
        
        if train_data is None or test_data is None:
            raise ValueError("Data splitter returned no train/test data")
        
        # Class balancing setup
        class_balancing_config = sn_train.get("class_balancing", {})
        
        global_dist = None
        if class_balancing_config.get("per_class_balance", False):
            strategy = class_balancing_config.get("global_distribution_strategy")
            sampling_pct = class_balancing_config.get("sampling_percentage", 0.1)
            
            global_dist = compute_global_class_distribution(
                dataset=train_data,
                strategy=strategy,
                data_splitter=self.data_splitter,
                sampling_percentage=sampling_pct
            )
        
        self.weight_calculator = ClassWeightCalculator(
            class_balancing_config,
            global_class_distribution=global_dist
        )
        
        if self.weight_calculator.enabled and self.weight_calculator.per_class_balance:
            self.class_weight_table = self.weight_calculator.create_weight_lookup_table()
        else:
            self.class_weight_table = None
        
        # Apply sampling if in optimized mode (sample_fraction < 1.0)
        if self.sample_fraction < 1.0:
            # Get approximate dataset sizes for sampling
            train_size = self.data_splitter.train_size if hasattr(self.data_splitter, 'train_size') else 89700
            test_size = self.data_splitter.test_size if hasattr(self.data_splitter, 'test_size') else 61328
            
            train_take = int(train_size * self.sample_fraction)
            test_take = int(test_size * self.sample_fraction)
            
            # Sample data (shuffle first to get random sample)
            train_data = train_data.shuffle(buffer_size=min(10000, train_size)).take(train_take)
            test_data = test_data.shuffle(buffer_size=min(10000, test_size)).take(test_take)
            
            if self.verbose:
                print(f"Optimized mode: using ~{train_take} train and ~{test_take} test samples")
        
        # Calculate shuffle buffer size
        base_buffer_size = 89700  # Default full buffer
        shuffle_buffer = int(base_buffer_size * self.shuffle_buffer_fraction)
        if self.sample_fraction < 1.0:
            # For sampled data, use smaller buffer proportional to sample size
            shuffle_buffer = min(shuffle_buffer, int(89700 * self.sample_fraction * self.shuffle_buffer_fraction))
        
        # Create data batches with appropriate shuffle buffer
        self.train_batches = (
            train_data
            .shuffle(buffer_size=shuffle_buffer)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self.test_batches = test_data.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Best tracking
        self.best_f1_value: float = 0.0
        self.best_loss_value: float = float('inf')
    
    def _compute_anchor_negative_weights_tf(self, labels):
        """
        Compute anchor/negative weights using pure TensorFlow ops.
        
        This replicates the logic from the main trainer but runs entirely on GPU.
        """
        anchor_count = tf.reduce_sum(tf.cast(labels == 1.0, tf.float32))
        negative_count = tf.reduce_sum(tf.cast(labels == 0.0, tf.float32))
        total_count = anchor_count + negative_count
        
        anchor_weight = tf.cond(
            anchor_count > 0,
            lambda: total_count / (2.0 * anchor_count),
            lambda: 1.0
        )
        negative_weight = tf.cond(
            negative_count > 0,
            lambda: total_count / (2.0 * negative_count),
            lambda: 1.0
        )
        
        anchor_mask = tf.cast(labels == 1.0, tf.float32)
        negative_mask = tf.cast(labels == 0.0, tf.float32)
        return anchor_mask * anchor_weight + negative_mask * negative_weight
    
    @tf.function
    def train_step(self, batch):
        """
        Perform a single training step with optional sample weighting.
        
        Uses pure TensorFlow operations for weight computation to maximize GPU utilization.
        """
        img1, img2, labels, class_info = batch
        
        with tf.GradientTape() as tape:
            yhat = self.siamese_model([img1, img2], training=True)
            per_sample_loss = self.loss_function(labels, yhat)
            
            # Apply sample weights if enabled (all TensorFlow ops)
            if self.weight_calculator.enabled:
                weights = tf.ones_like(per_sample_loss)
                
                # Anchor/Negative balance (pure TensorFlow)
                if self.weight_calculator.anchor_negative_balance:
                    an_weights = self._compute_anchor_negative_weights_tf(labels)
                    weights = weights * an_weights
                
                # Per-class balance (TensorFlow lookup table)
                if self.class_weight_table is not None:
                    class_weights = self.class_weight_table.lookup(class_info)
                    weights = weights * class_weights
                
                # Normalize weights to mean=1.0
                weights = weights / tf.reduce_mean(weights)
                
                # Cast weights to match loss dtype and apply
                weights_casted = tf.cast(weights, dtype=per_sample_loss.dtype)
                weighted_loss = per_sample_loss * weights_casted
                loss = tf.reduce_mean(weighted_loss)
            else:
                loss = tf.reduce_mean(per_sample_loss)
            
            if self.mixed_precision_enabled:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        
        if self.mixed_precision_enabled:
            scaled_gradients = tape.gradient(scaled_loss, self.siamese_model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.siamese_model.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.siamese_model.trainable_variables))
        
        return loss, yhat
    
    def train_and_evaluate(self) -> Dict[str, float]:
        """
        Train the model and return final test metrics.
        
        Returns:
            Dictionary with keys: 'f1', 'accuracy', 'precision', 'recall', 'loss'
        """
        for epoch in range(1, self.num_epochs + 1):
            if self.verbose:
                print(f"    Epoch {epoch}/{self.num_epochs}", end="", flush=True)
            
            r = Recall()
            p = Precision()
            
            for batch in self.train_batches:
                loss, yhat = self.train_step(batch)
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
            
            # Test evaluation at end of each epoch
            test_loss, test_recall, test_precision, test_f1 = self._test()
            
            # Print epoch results on same line
            if self.verbose:
                print(f" - loss: {test_loss:.4f}, f1: {test_f1:.4f}")
            
            # Track best F1
            if test_f1 > self.best_f1_value:
                self.best_f1_value = test_f1
            
            if test_loss < self.best_loss_value:
                self.best_loss_value = test_loss
        
        # Final evaluation
        final_loss, final_recall, final_precision, final_f1 = self._test()
        
        # Calculate accuracy from confusion matrix
        accuracy = self._calculate_accuracy()
        
        return {
            "f1": final_f1,
            "accuracy": accuracy,
            "precision": final_precision,
            "recall": final_recall,
            "loss": final_loss,
            "best_f1": self.best_f1_value,
            "best_loss": self.best_loss_value
        }
    
    def _test(self):
        r = Recall()
        p = Precision()
        total_loss = tf.constant(0.0, dtype=tf.float32)
        num_batches = 0
        
        for test_input, test_val, y_true, class_info in self.test_batches:
            yhat = self.siamese_model.predict([test_input, test_val], verbose=0)
            r.update_state(y_true, yhat)
            p.update_state(y_true, yhat)
            batch_loss = self.test_loss_function(y_true, yhat)
            total_loss = total_loss + tf.cast(batch_loss, tf.float32)
            num_batches += 1
        
        avg_loss = float(total_loss.numpy() / num_batches) if num_batches > 0 else 0.0
        recall_val = float(r.result().numpy())
        precision_val = float(p.result().numpy())
        f1_val = self._calculate_f1(precision_val, recall_val)
        
        return avg_loss, recall_val, precision_val, f1_val
    
    def _calculate_accuracy(self) -> float:
        """Calculate accuracy on test set."""
        total_correct = 0
        total_samples = 0
        
        for test_input, test_val, y_true, class_info in self.test_batches:
            yhat = self.siamese_model.predict([test_input, test_val], verbose=0)
            predictions = (yhat > 0.5).astype(float)
            correct = (predictions.flatten() == y_true.numpy().flatten()).sum()
            total_correct += correct
            total_samples += len(y_true)
        
        return total_correct / total_samples if total_samples > 0 else 0.0
    
    @staticmethod
    def _calculate_f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


def create_permutation_trainer(
    bat_type: str = 'r',
    augmented_data: bool = False,
    data_source: str = 'video',
    permute_labels: bool = True,
    num_epochs: int = 10,
    mlflow_enabled: bool = False,
    verbose: bool = False,
    sample_fraction: float = 1.0,
    shuffle_buffer_fraction: float = 1.0
) -> PermutationTrainer:
    """
    Factory function to create a PermutationTrainer instance.
    
    This is the recommended way to create trainers for permutation tests,
    as it provides a clean interface for the PermutationTest class.
    
    Args:
        bat_type: Bat type ('r' for rousettus, 'm' for mauritius)
        augmented_data: Whether to use augmented data
        data_source: Data source type ('video' or 'still')
        permute_labels: Whether to permute labels (True for null hypothesis)
        num_epochs: Number of training epochs
        mlflow_enabled: Whether to enable MLflow logging
        verbose: Whether to print progress
        sample_fraction: Fraction of data to use (1.0 = all, 0.25 = 25%)
        shuffle_buffer_fraction: Fraction of shuffle buffer size (1.0 = full)
    """
    return PermutationTrainer(
        bat_type=bat_type,
        augmented_data=augmented_data,
        data_source=data_source,
        permute_labels=permute_labels,
        num_epochs=num_epochs,
        mlflow_enabled=mlflow_enabled,
        verbose=verbose,
        sample_fraction=sample_fraction,
        shuffle_buffer_fraction=shuffle_buffer_fraction
    )
