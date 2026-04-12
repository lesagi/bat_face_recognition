"""
Lightweight Trainer for Permutation Test Iterations.

This module provides a streamlined version of SiameseNetworkTrainer optimized
for running many permutation iterations efficiently. Key optimizations:
- Disabled MLflow logging
- Disabled checkpoint saving
- Disabled post-training automation (saliency maps, predictions)
- Large batch size for GPU utilization
- Model reuse across permutations to prevent GPU memory leaks
- Optional eval-last-only mode to skip intermediate evaluations
- Returns only final metrics
"""

import os
import random
import sys
import time
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
    _BG_CONFIG_KEY = {
        "green": "green_bg_input",
        "random": "random_bg_input",
        "original": "original_bg_input",
    }

    def __init__(
        self,
        bat_type: str = 'r',
        augmented_data: bool = False,
        data_source: str = 'video',
        background: str = 'random',
        split_mode: str = 'image_split',
        permute_labels: bool = True,
        num_epochs: int = 10,
        mlflow_enabled: bool = False,
        verbose: bool = False,
        sample_fraction: float = 1.0,
        shuffle_buffer_fraction: float = 1.0,
        optimizer: str = 'adam',
        batch_size: int = 128,
        eval_last_only: bool = True
    ):
        cfg = load_config()
        sn_train = cfg.siamese_network.training
        
        self.permute_labels = permute_labels
        self.split_mode = split_mode
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.mlflow_enabled = mlflow_enabled
        self.sample_fraction = sample_fraction
        self.shuffle_buffer_fraction = shuffle_buffer_fraction
        self.optimizer_type = optimizer
        self.batch_size = batch_size
        self.eval_last_only = eval_last_only
        
        # GPU status logging
        if self.verbose:
            print(f"GPU(s) available: {_GPU_COUNT}", flush=True)
            if _GPU_COUNT > 0:
                print(f"  Using: {_GPU_NAMES}", flush=True)
            print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}", flush=True)
        
        # Mixed Precision Setup (silent)
        self.mixed_precision_enabled = cfg.siamese_network.advanced.get("mixed_precision", False)
        if self.mixed_precision_enabled:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        if self.verbose:
            print(f"Mixed precision: {self.mixed_precision_enabled}", flush=True)
            print(f"Batch size: {self.batch_size}", flush=True)
            print(f"Eval last only: {self.eval_last_only}", flush=True)
        
        # Store config values needed for data setup (per-permutation)
        bat_key = 'mauritius' if bat_type == 'm' else 'rousettus'
        input_paths = cfg.siamese_network.input_paths[bat_key]
        bg_key = self._BG_CONFIG_KEY[background]
        self.input_dir = input_paths.get(bg_key)
        
        if not self.input_dir or not os.path.exists(self.input_dir):
            raise ValueError(
                f"Invalid or missing training input_dir for background '{background}': "
                f"{self.input_dir}. "
                f"Please set input_paths.{bat_key}.{bg_key} in config.yml"
            )
        
        self.training_portion = sn_train.get("train_val_split", 0.7)
        self.pair_mode = sn_train.get("pair_mode", "permutation")
        self.class_balancing_config = sn_train.get("class_balancing", {})
        self.split_seed = random.randint(0, 2**31)
        
        # ============================================================
        # Phase 1: Model, optimizer, loss (GPU allocation -- ONCE)
        # These objects persist across all permutations to avoid
        # TF BFC allocator memory leaks.
        # ============================================================
        self.learning_rate = sn_train.get("learning_rate", 1e-4)
        self.siamese_model = SiameseNetwork(L1Dist()).model
        
        if self.optimizer_type == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(self.learning_rate, momentum=0.9)
            if self.verbose:
                print(f"Using SGD optimizer (memory-efficient)", flush=True)
        else:
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            if self.verbose:
                print(f"Using Adam optimizer", flush=True)
        
        if self.mixed_precision_enabled:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
        
        self.loss_function = tf.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.test_loss_function = tf.losses.BinaryCrossentropy()
        
        # ============================================================
        # Phase 2: Data setup (per-permutation)
        # ============================================================
        self._setup_data()
    
    def _setup_data(self):
        self.data_splitter = SiameseNetworkTrainingDataSplitter(
            [self.input_dir], 
            training_portion=self.training_portion, 
            mode=self.pair_mode, 
            permute_labels=self.permute_labels,
            split_seed=self.split_seed,
            split_mode=self.split_mode,
        )
        
        train_data = self.data_splitter.train_data
        test_data = self.data_splitter.test_data
        
        if train_data is None or test_data is None:
            raise ValueError("Data splitter returned no train/test data")
        
        # Class balancing setup
        global_dist = None
        if self.class_balancing_config.get("per_class_balance", False):
            strategy = self.class_balancing_config.get("global_distribution_strategy")
            sampling_pct = self.class_balancing_config.get("sampling_percentage", 0.1)
            
            global_dist = compute_global_class_distribution(
                dataset=train_data,
                strategy=strategy,
                data_splitter=self.data_splitter,
                sampling_percentage=sampling_pct
            )
        
        self.weight_calculator = ClassWeightCalculator(
            self.class_balancing_config,
            global_class_distribution=global_dist
        )
        
        if self.weight_calculator.enabled and self.weight_calculator.per_class_balance:
            self.class_weight_table = self.weight_calculator.create_weight_lookup_table()
        else:
            self.class_weight_table = None
        
        # Apply sampling if sample_fraction < 1.0
        train_size = self.data_splitter.train_size
        test_size = self.data_splitter.test_size
        
        if self.sample_fraction < 1.0:
            train_take = int(train_size * self.sample_fraction)
            test_take = int(test_size * self.sample_fraction)
            
            train_data = train_data.shuffle(buffer_size=min(10000, train_size)).take(train_take)
            test_data = test_data.shuffle(buffer_size=min(10000, test_size)).take(test_take)
            
            if self.verbose:
                print(f"Sampled data: ~{train_take} train and ~{test_take} test pairs", flush=True)
        
        # Calculate shuffle buffer size
        shuffle_buffer = int(train_size * self.shuffle_buffer_fraction)
        if self.sample_fraction < 1.0:
            shuffle_buffer = min(shuffle_buffer, int(train_size * self.sample_fraction * self.shuffle_buffer_fraction))
        
        # Create data batches (no .cache() -- bottleneck is GPU, not I/O)
        self.train_batches = (
            train_data
            .shuffle(buffer_size=shuffle_buffer)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self.test_batches = test_data.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Reset best tracking for this permutation
        self.best_f1_value: float = 0.0
        self.best_loss_value: float = float('inf')
    
    def _reinit_layer(self, layer):
        if hasattr(layer, 'kernel') and hasattr(layer, 'kernel_initializer'):
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        if hasattr(layer, 'bias') and hasattr(layer, 'bias_initializer'):
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))
    
    def reinitialize_weights(self):
        for layer in self.siamese_model.layers:
            if isinstance(layer, tf.keras.Model):
                for sublayer in layer.layers:
                    self._reinit_layer(sublayer)
            else:
                self._reinit_layer(layer)
        
        # Reset optimizer slot variables (momentum, velocity, etc.) in-place.
        # This avoids creating a new optimizer object, which would cause
        # tf.function retracing and potential new GPU memory allocations.
        for var in self.optimizer.variables():
            # Preserve loss scale variables for mixed precision
            if 'loss_scale' not in var.name.lower():
                var.assign(tf.zeros_like(var))
        
        if self.verbose:
            print(f"  Model weights re-randomized, optimizer state reset", flush=True)
    
    def reset_for_new_permutation(self):
        self.reinitialize_weights()
        self._setup_data()
    
    def _compute_anchor_negative_weights_tf(self, labels):
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
        img1, img2, labels, class_info = batch
        
        with tf.GradientTape() as tape:
            yhat = self.siamese_model([img1, img2], training=True)
            per_sample_loss = self.loss_function(labels, yhat)
            
            if self.weight_calculator.enabled:
                weights = tf.ones_like(per_sample_loss)
                
                if self.weight_calculator.anchor_negative_balance:
                    an_weights = self._compute_anchor_negative_weights_tf(labels)
                    weights = weights * an_weights
                
                if self.class_weight_table is not None:
                    class_weights = self.class_weight_table.lookup(class_info)
                    weights = weights * class_weights
                
                weights = weights / tf.reduce_mean(weights)
                weights_casted = tf.cast(weights, dtype=per_sample_loss.dtype)
                weighted_loss = per_sample_loss * weights_casted
                loss = tf.reduce_mean(weighted_loss)
            else:
                loss = tf.reduce_mean(per_sample_loss)
            
            if self.mixed_precision_enabled:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        
        if self.mixed_precision_enabled:
            scaled_grad = tape.gradient(scaled_loss, self.siamese_model.trainable_variables)
            grad = self.optimizer.get_unscaled_gradients(scaled_grad)
        else:
            grad = tape.gradient(loss, self.siamese_model.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grad, self.siamese_model.trainable_variables))
        return loss, yhat
    
    def train_and_evaluate(self) -> Dict[str, float]:
        import logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        logging.getLogger('tensorflow').setLevel(logging.WARNING)
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            
            r = Recall()
            p = Precision()
            batch_count = 0
            epoch_loss = 0.0
            
            for batch in self.train_batches:
                loss, yhat = self.train_step(batch)
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                epoch_loss += float(loss)
                batch_count += 1
                if self.verbose and batch_count % 50 == 0:
                    elapsed = time.time() - epoch_start
                    print(f"      [Train] Epoch {epoch}/{self.num_epochs} - batch {batch_count}, elapsed: {elapsed:.1f}s", flush=True)
            
            train_time = time.time() - epoch_start
            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            
            is_last_epoch = (epoch == self.num_epochs)
            last_eval = None
            
            if self.eval_last_only and not is_last_epoch:
                if self.verbose:
                    train_recall = float(r.result().numpy())
                    train_precision = float(p.result().numpy())
                    train_f1 = self._calculate_f1(train_precision, train_recall)
                    print(f"    Epoch {epoch}/{self.num_epochs} - train_loss: {avg_train_loss:.4f}, train_f1: {train_f1:.4f} ({batch_count} batches in {train_time:.1f}s)", flush=True)
            else:
                eval_start = time.time()
                test_loss, test_recall, test_precision, test_f1, test_accuracy = self._test()
                eval_time = time.time() - eval_start
                last_eval = (test_loss, test_recall, test_precision, test_f1, test_accuracy)
                
                if self.verbose:
                    epoch_time = time.time() - epoch_start
                    print(f"    Epoch {epoch}/{self.num_epochs} - loss: {test_loss:.4f}, f1: {test_f1:.4f}, acc: {test_accuracy:.4f} (train: {train_time:.1f}s, eval: {eval_time:.1f}s, total: {epoch_time:.1f}s)", flush=True)
                
                if test_f1 > self.best_f1_value:
                    self.best_f1_value = test_f1
                
                if test_loss < self.best_loss_value:
                    self.best_loss_value = test_loss
        
        if last_eval is not None:
            final_loss, final_recall, final_precision, final_f1, final_accuracy = last_eval
        else:
            final_loss, final_recall, final_precision, final_f1, final_accuracy = self._test()
        
        return {
            "f1": final_f1,
            "accuracy": final_accuracy,
            "precision": final_precision,
            "recall": final_recall,
            "loss": final_loss,
            "best_f1": self.best_f1_value,
            "best_loss": self.best_loss_value
        }
    
    @tf.function
    def _predict_batch(self, img1, img2):
        return self.siamese_model([img1, img2], training=False)

    def _test(self):
        r = Recall()
        p = Precision()
        total_loss = tf.constant(0.0, dtype=tf.float32)
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        for test_input, test_val, y_true, class_info in self.test_batches:
            yhat = self._predict_batch(test_input, test_val)
            r.update_state(y_true, yhat)
            p.update_state(y_true, yhat)
            batch_loss = self.test_loss_function(y_true, yhat)
            total_loss = total_loss + tf.cast(batch_loss, tf.float32)
            
            predictions = tf.cast(yhat > 0.5, tf.float32)
            correct = tf.reduce_sum(tf.cast(
                tf.equal(tf.reshape(predictions, [-1]),
                         tf.cast(tf.reshape(y_true, [-1]), tf.float32)),
                tf.float32
            ))
            total_correct += int(correct.numpy())
            total_samples += len(y_true)
            num_batches += 1
        
        avg_loss = float(total_loss.numpy() / num_batches) if num_batches > 0 else 0.0
        recall_val = float(r.result().numpy())
        precision_val = float(p.result().numpy())
        f1_val = self._calculate_f1(precision_val, recall_val)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, recall_val, precision_val, f1_val, accuracy
    
    @staticmethod
    def _calculate_f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


def create_permutation_trainer(
    bat_type: str = 'r',
    augmented_data: bool = False,
    data_source: str = 'video',
    background: str = 'random',
    split_mode: str = 'image_split',
    permute_labels: bool = True,
    num_epochs: int = 10,
    mlflow_enabled: bool = False,
    verbose: bool = False,
    sample_fraction: float = 1.0,
    shuffle_buffer_fraction: float = 1.0,
    optimizer: str = 'adam',
    batch_size: int = 128,
    eval_last_only: bool = True
) -> PermutationTrainer:
    return PermutationTrainer(
        bat_type=bat_type,
        augmented_data=augmented_data,
        data_source=data_source,
        background=background,
        split_mode=split_mode,
        permute_labels=permute_labels,
        num_epochs=num_epochs,
        mlflow_enabled=mlflow_enabled,
        verbose=verbose,
        sample_fraction=sample_fraction,
        shuffle_buffer_fraction=shuffle_buffer_fraction,
        optimizer=optimizer,
        batch_size=batch_size,
        eval_last_only=eval_last_only
    )
