"""
Siamese network trainer module.
"""

import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import random
import time
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall  # type: ignore[reportMissingImports]
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}")
else:
    print("❌ No GPU found, using CPU")

from siamese_core.network import SiameseNetwork, L1Dist, SIAMESE_INPUT_EDGE_LENGTH
from siamese_data.data_splitter import SiameseNetworkTrainingDataSplitter
from siamese_data.class_weights import ClassWeightCalculator
from siamese_data.global_distribution import compute_global_class_distribution
from siamese_training.focal_loss import BinaryFocalLoss
from config.loader import load_config


class TeeStream:
    """Writes to two streams simultaneously (tee for stdout capture)."""

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.stream2.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

    def fileno(self):
        return self.stream1.fileno()


class SiameseNetworkTrainer:
    _BG_CONFIG_KEY = {
        "green": "green_bg_input",
        "random": "random_bg_input",
        "original": "original_bg_input",
    }

    def __init__(
        self,
        bat_type: str,
        augmented_data: bool,
        data_source: str,
        background: str,
        split_mode: str,
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_function=tf.losses.BinaryCrossentropy(),
        permute_labels: Optional[bool] = None,
    ):
        cfg = load_config()
        self._config = cfg
        sn_train = cfg.siamese_network.training
        
        if permute_labels is None:
            permute_labels = sn_train.get("permute_labels", False)
        
        # Mixed Precision Setup
        self.mixed_precision_enabled = cfg.siamese_network.advanced.get("mixed_precision", False)
        if self.mixed_precision_enabled:
            print("🚀 Enabling Mixed Precision training...")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"   - Compute dtype: {policy.compute_dtype}")
            print(f"   - Variable dtype: {policy.variable_dtype}")

        self.permute_labels = permute_labels
        self.background = background
        self.split_mode = split_mode

        # Get bat-type-specific input paths
        bat_key = 'mauritius' if bat_type == 'm' else 'rousettus'
        input_paths = cfg.siamese_network.input_paths[bat_key]

        # Resolve input directory from --background choice
        bg_key = self._BG_CONFIG_KEY[background]
        self.input_dir = input_paths.get(bg_key)
        
        if not self.input_dir or not os.path.exists(self.input_dir):
            raise ValueError(f"Invalid or missing training input_dir for background '{background}': "
                           f"{self.input_dir}. "
                           f"Please set input_paths.{bat_key}.{bg_key} in config.yml")

        # Training hyperparameters
        self.num_epochs = sn_train.get("epochs", 80)
        self.batch_size = sn_train.get("batch_size", 16)

        # Output paths - per-species directory from config
        output_dirs = sn_train.get("output_dir")
        if not isinstance(output_dirs, dict) or bat_key not in output_dirs:
            raise ValueError(f"Missing siamese_network.training.output_dir.{bat_key} in config")
        base_output_dir = output_dirs[bat_key]
        if not base_output_dir:
            raise ValueError(f"Empty siamese_network.training.output_dir.{bat_key} in config")
        
        # Store training parameters for post-training automation
        self.bat_type = bat_type
        self.augmented_data = augmented_data
        self.data_source = data_source
        
        # Create output directory with format: {date}_{experiment_id}_{run_id}
        # We'll create the actual directory name after MLflow run starts (to get run_id)
        self.base_output_dir = base_output_dir
        self.model_output_dir = None  # Will be set in _create_output_directory()
        self.checkpoint_dir = None  # Set only when save_tf_checkpoints is True
        self._checkpoint_manager = None  # tf.train.CheckpointManager, optional

        _default_retention: Dict[str, bool] = {
            "save_best_f1": True,
            "save_best_recall": True,
            "save_best_precision": True,
            "save_best_loss": True,
            "save_final_model": False,
            "save_tf_checkpoints": False,
        }
        self._artifact_retention: Dict[str, bool] = {
            **_default_retention,
            **(sn_train.get("artifact_retention") or {}),
        }
        # First SavedModel written at an epoch wins the physical directory; other roles symlink.
        self._epoch_canonical_subdir: Dict[int, str] = {}

        # MLflow setup
        self.mlflow_enabled = cfg.mlflow.enabled
        self.mlflow_tracking_uri = self._resolve_mlflow_tracking_uri(cfg.mlflow.tracking_uri)
        self.mlflow_experiment_name = self._build_experiment_name(bat_type, augmented_data, data_source)
        self.parent_run = None
        self.metric_history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_recall": [],
            "train_precision": [],
            "train_f1": [],
            "test_loss": [],
            "test_recall": [],
            "test_precision": [],
            "test_f1": [],
        }
        self.current_epoch: int = 0
        
        # Best model tracking (test set)
        self.best_loss_value: float = float('inf')
        self.best_loss_epoch: int = 0
        self.best_f1_value: float = 0.0
        self.best_f1_epoch: int = 0
        self.best_recall_value: float = 0.0
        self.best_recall_epoch: int = 0
        self.best_precision_value: float = 0.0
        self.best_precision_epoch: int = 0
        
        # Early stopping tracking
        self.early_stopping_enabled = sn_train.get("early_stopping", {}).get("enabled", False)
        self.early_stopping_patience = sn_train.get("early_stopping", {}).get("patience", 10)
        self.early_stopping_min_delta = sn_train.get("early_stopping", {}).get("min_delta", 0.001)
        self.early_stopping_restore_best = sn_train.get("early_stopping", {}).get("restore_best_weights", True)
        self.early_stopping_counter = 0
        self.early_stopping_best_value = 0.0  # Track best F1 for early stopping

        # Model, optimizer, loss
        self.siamese_model = SiameseNetwork(L1Dist()).model
        self._initial_lr = sn_train.get("learning_rate", 1e-4)
        self.optimizer = None  # Will be created after batches are available
        
        # Wrap optimizer for mixed precision if enabled
        self._needs_mixed_precision_wrap = self.mixed_precision_enabled
            
        # Loss function setup (configurable via config.yml)
        loss_cfg = sn_train.get("loss", {})
        loss_type = loss_cfg.get("type", "BinaryCrossentropy")
        from_logits = loss_cfg.get("from_logits", False)

        if loss_type == "BinaryFocalLoss":
            focal_alpha = loss_cfg.get("focal_alpha", 0.75)
            focal_gamma = loss_cfg.get("focal_gamma", 2.0)
            self.loss_function = BinaryFocalLoss(
                alpha=focal_alpha, gamma=focal_gamma,
                from_logits=from_logits,
                reduction=tf.keras.losses.Reduction.NONE,
            )
            self.test_loss_function = BinaryFocalLoss(
                alpha=focal_alpha, gamma=focal_gamma,
                from_logits=from_logits,
                reduction=tf.keras.losses.Reduction.AUTO,
            )
            print(f"🔧 Loss: BinaryFocalLoss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            self.loss_function = tf.losses.BinaryCrossentropy(
                from_logits=from_logits,
                reduction=tf.keras.losses.Reduction.NONE,
            )
            self.test_loss_function = tf.losses.BinaryCrossentropy(
                from_logits=from_logits,
            )
            print(f"🔧 Loss: BinaryCrossentropy")

        # Data loading (must come BEFORE class balancing for global distribution)
        print(f"🔧 Loading data from: {self.input_dir}")
        
        # Check GPU availability
        print(f"🔧 TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
        print(f"🔧 TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

        # Get training portion from config (split_mode is a constructor argument)
        training_portion = sn_train.get("train_val_split", 0.7)
        self.training_portion = training_portion
        pair_mode = sn_train.get("pair_mode", "permutation")

        self.data_splitter = SiameseNetworkTrainingDataSplitter(
            [self.input_dir], training_portion=training_portion, mode=pair_mode,
            permute_labels=self.permute_labels, split_mode=self.split_mode,
        )
        train_data = self.data_splitter.train_data
        test_data = self.data_splitter.test_data
        if train_data is None or test_data is None:
            raise ValueError("Data splitter returned no train/test data")

        # Class balancing setup (after data loading)
        class_balancing_config = sn_train.get("class_balancing", {})
        
        # Store for MLflow logging
        self.class_balancing_config = class_balancing_config
        
        # Compute global distribution if needed (BEFORE batching and calculator creation)
        global_dist = None
        if class_balancing_config.get("per_class_balance", False):
            strategy = class_balancing_config.get("global_distribution_strategy")
            if not strategy:
                raise ValueError(
                    "per_class_balance=true requires 'global_distribution_strategy' in config. "
                    "Options: 'file_based', 'sampled', 'full_scan'"
                )
            
            sampling_pct = class_balancing_config.get("sampling_percentage", 0.1)
            print(f"🔧 Computing global class distribution using '{strategy}' strategy...")
            start_time = time.time()
            
            # Compute before batching
            global_dist = compute_global_class_distribution(
                dataset=train_data,
                strategy=strategy,
                data_splitter=self.data_splitter,
                sampling_percentage=sampling_pct
            )
            
            elapsed = time.time() - start_time
            print(f"   ✅ Global distribution computed in {elapsed:.2f}s")
            print(f"   📊 Classes: {list(global_dist.keys())}")
            total_samples = sum(global_dist.values())
            print(f"   📊 Total samples: {total_samples}")
        
        # Initialize calculator with global distribution
        self.weight_calculator = ClassWeightCalculator(
            class_balancing_config,
            global_class_distribution=global_dist
        )
        
        if self.weight_calculator.enabled:
            print(f"🔧 Class balancing enabled:")
            print(f"   - Anchor/Negative balance: {self.weight_calculator.anchor_negative_balance}")
            print(f"   - Per-class balance: {self.weight_calculator.per_class_balance}")
            print(f"   - Weighting scheme: {self.weight_calculator.weighting_scheme}")
            if self.weight_calculator.weighting_scheme == 'ens':
                print(f"   - ENS beta: {self.weight_calculator.ens_beta}")
        
        # Create TensorFlow lookup table for per-class weights (GPU-optimized)
        if self.weight_calculator.enabled and self.weight_calculator.per_class_balance:
            self.class_weight_table = self.weight_calculator.create_weight_lookup_table()
            if self.class_weight_table is not None:
                print(f"   ✅ Created TensorFlow weight lookup table")
        else:
            self.class_weight_table = None
        
        # Create data batches (after class balancing setup)
        # Optimized pipeline: use AUTOTUNE for prefetch (no .cache() to avoid RAM bloat on large datasets)
        print(f"🔧 Creating data batches...")
        self.train_batches = (
            train_data
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self.test_batches = test_data.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        print(f"✅ Data loading completed")
        
        # Print class distribution if class balancing is enabled
        if self.weight_calculator.enabled:
            train_dist = self.data_splitter.get_class_distribution('train')
            print(f"\n📊 Training class distribution:")
            for cls, count in sorted(train_dist.items()):
                print(f"   {cls}: {count} samples")

        # Create optimizer with per-epoch LR decay (paper: 0.99x per epoch)
        steps_per_epoch = len(self.train_batches)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self._initial_lr,
            decay_steps=steps_per_epoch,
            decay_rate=0.99,
            staircase=True,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        if self._needs_mixed_precision_wrap:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

        # TF checkpoint
        self.checkpoint = tf.train.Checkpoint(opt=self.optimizer, siamese_model=self.siamese_model)

    def _resolve_mlflow_tracking_uri(self, uri: str) -> str:
        """
        Resolve MLflow tracking URI to be path-agnostic.
        If URI is relative (e.g., './mlruns'), resolve it to absolute path based on project root.
        """
        # If URI already has a file:// scheme with absolute path, return as-is
        if uri.startswith("file:///"):
            return uri
        
        # If URI starts with file:./ (relative), strip the file: prefix
        if uri.startswith("file:./"):
            uri = uri[5:]  # Remove 'file:'
        
        # Check if URI is relative (starts with ./ or doesn't start with /)
        if uri.startswith("./") or (not uri.startswith("/") and not uri.startswith("file://")):
            # Get project root (bat_face_rec directory)
            # Navigate from app/siamese_training/trainer.py -> bat_face_rec
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            
            # Remove leading ./ if present
            relative_path = uri[2:] if uri.startswith("./") else uri
            
            # Resolve to absolute path
            absolute_path = os.path.join(project_root, relative_path)
            
            # Return with file:// prefix
            return f"file://{absolute_path}"
        
        # URI is already absolute, add file:// prefix if not present
        if not uri.startswith("file://"):
            return f"file://{uri}"
        
        return uri

    def _build_experiment_name(self, bat_type: str, augmented_data: bool, data_source: str) -> str:
        """Build experiment name from template using provided arguments."""
        if bat_type not in ['m', 'r']:
            raise ValueError("bat_type must be 'm' (mauritius) or 'r' (rousettus)")
        if data_source not in ['video', 'still']:
            raise ValueError("data_source must be 'video' or 'still'")
        
        bat_name = "mauritius" if bat_type == 'm' else "rousettus"
        aug_str = "augmented" if augmented_data else "no_aug"
        bg_str = f"{self.background}_bg"
        
        return f"siamese_{bat_name}_{data_source}_{aug_str}_{bg_str}"

    def _create_output_directory(self):
        """
        Create output directory with format: {date}_{experiment_id}_{run_id}
        """
        from datetime import datetime
        
        # Get date in YYYYMMDD format
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Get experiment ID from MLflow experiment name
        experiment_id = self.mlflow_experiment_name if self.mlflow_enabled else "siamese_unknown"
        
        # Get run ID from MLflow if enabled, otherwise use timestamp
        if self.mlflow_enabled and self.parent_run:
            run_id = self.parent_run.info.run_id[:8]  # Use first 8 chars of run_id
        else:
            run_id = datetime.now().strftime("%H%M%S")
        
        # Create directory name
        dir_name = f"{date_str}_{experiment_id}_{run_id}"
        self.model_output_dir = os.path.join(self.base_output_dir, dir_name)
        os.makedirs(self.model_output_dir, exist_ok=True)

        if self._artifact_retention.get("save_tf_checkpoints"):
            self.checkpoint_dir = os.path.join(self.model_output_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Save config snapshot locally (and to MLflow later)
        import yaml as _yaml
        config_snapshot_path = os.path.join(self.model_output_dir, "config_snapshot.yml")
        with open(config_snapshot_path, "w") as f:
            _yaml.dump(self._config.get_full_config(), f, default_flow_style=False)

        print(f"📁 Created output directory: {self.model_output_dir}")
    
    def _start_parent_run(self, bat_type: str, augmented_data: bool, data_source: str):
        if not self.mlflow_enabled:
            # Create output directory even if MLflow is disabled
            self._create_output_directory()
            # Start capturing stdout to training log file
            self._log_file = open(os.path.join(self.model_output_dir, "training.log"), "w")
            self._original_stdout = sys.stdout
            sys.stdout = TeeStream(sys.stdout, self._log_file)
            return
        
        print(f"🔧 Starting MLflow run...")
        print(f"📊 Experiment name: {self.mlflow_experiment_name}")
        print(f"📊 Tracking URI: {self.mlflow_tracking_uri}")
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        
        # Build descriptive run name from parameters
        bat_name = "mauritius" if bat_type == 'm' else "rousettus"
        aug_str = "augmented" if augmented_data else "no_aug"
        run_name = f"training_{bat_name}_{data_source}_{aug_str}_{self.background}_bg"
        
        print(f"📊 Run name: {run_name}")
        self.parent_run = mlflow.start_run(run_name=run_name)

        # Create output directory now that we have run_id
        self._create_output_directory()

        # Set searchable tags for filtering in MLflow UI
        mlflow.set_tag("bat_type", bat_type)
        mlflow.set_tag("bat_species", bat_name)
        mlflow.set_tag("data_source", data_source)
        mlflow.set_tag("augmented", str(augmented_data))
        mlflow.set_tag("background", self.background)
        mlflow.set_tag("model_output_dir", self.model_output_dir)

        # Log config snapshot as MLflow artifact
        config_snapshot_path = os.path.join(self.model_output_dir, "config_snapshot.yml")
        if os.path.exists(config_snapshot_path):
            mlflow.log_artifact(config_snapshot_path, artifact_path="config")

        # Start capturing stdout to training log file
        self._log_file = open(os.path.join(self.model_output_dir, "training.log"), "w")
        self._original_stdout = sys.stdout
        sys.stdout = TeeStream(sys.stdout, self._log_file)

        self._log_hyperparameters(bat_type, augmented_data, data_source)
        self._log_class_balancing_to_mlflow()

    def _end_parent_run(self):
        # Stop stdout capture and close log file
        if hasattr(self, '_original_stdout') and self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if hasattr(self, '_log_file') and self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        if not self.mlflow_enabled:
            return
        if self.parent_run is not None:
            # Log training log as artifact
            log_path = os.path.join(self.model_output_dir, "training.log")
            if os.path.exists(log_path):
                mlflow.log_artifact(log_path, artifact_path="logs")
            mlflow.end_run()
            self.parent_run = None

    def _log_hyperparameters(self, bat_type: str, augmented_data: bool, data_source: str):
        if not self.mlflow_enabled:
            return
        cfg = load_config()
        sn_train = cfg.siamese_network.training
        params: Dict[str, Any] = {
            "epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": sn_train.get("learning_rate", 1e-4),
            "optimizer": sn_train.get("optimizer", {}).get("type", "Adam"),
            "loss": sn_train.get("loss", {}).get("type", "BinaryCrossentropy"),
            "loss_from_logits": sn_train.get("loss", {}).get("from_logits", False),
            "focal_alpha": sn_train.get("loss", {}).get("focal_alpha", "N/A"),
            "focal_gamma": sn_train.get("loss", {}).get("focal_gamma", "N/A"),
            "train_val_split": self.training_portion,
            "pair_mode": sn_train.get("pair_mode", "permutation"),
            "split_mode": self.split_mode,
            "max_samples_per_class": sn_train.get("max_samples_per_class", 0),
            "input_dir": self.input_dir,
            "output_dir": self.model_output_dir,
            "bat_type": bat_type,
            "augmented_data": augmented_data,
            "data_source": data_source,
            "background": self.background,
            "experiment_name": self.mlflow_experiment_name,
            # Weight balancing config
            "class_balancing_enabled": self.weight_calculator.enabled,
            "anchor_negative_balance": self.weight_calculator.anchor_negative_balance,
            "anchor_target_ratio": self.weight_calculator.anchor_target_ratio,
            "per_class_balance": self.weight_calculator.per_class_balance,
            "weighting_scheme": self.weight_calculator.weighting_scheme,
            "ens_beta": self.weight_calculator.ens_beta,
            "negative_pair_combination": self.weight_calculator.negative_pair_combination,
            "permute_labels": self.permute_labels,
        }
        mlflow.log_params(params)
        self._log_sample_images()

    def _log_epoch_metrics(self, epoch: int, train: Dict[str, float], test: Dict[str, float]):
        if not self.mlflow_enabled:
            return
        
        # Log metrics to parent run with step=epoch for UI graphs
        mlflow.log_metrics(
            {
                "train_loss": float(train["loss"]),
                "train_recall": float(train["recall"]),
                "train_precision": float(train["precision"]),
                "train_f1": float(train["f1"]),
                "test_loss": float(test["loss"]),
                "test_recall": float(test["recall"]),
                "test_precision": float(test["precision"]),
                "test_f1": float(test["f1"]),
            },
            step=epoch,
        )

    def _update_metric_history(self, train: Dict[str, float], test: Dict[str, float]):
        self.metric_history["train_loss"].append(float(train["loss"]))
        self.metric_history["train_recall"].append(float(train["recall"]))
        self.metric_history["train_precision"].append(float(train["precision"]))
        self.metric_history["train_f1"].append(float(train["f1"]))
        self.metric_history["test_loss"].append(float(test["loss"]))
        self.metric_history["test_recall"].append(float(test["recall"]))
        self.metric_history["test_precision"].append(float(test["precision"]))
        self.metric_history["test_f1"].append(float(test["f1"]))

    def _plot_and_log_artifacts(self, epoch: int):
        if not self.mlflow_enabled:
            return
        epochs = list(range(1, self.current_epoch + 1))

        def plot_metric(train_vals: List[float], test_vals: List[float], title: str, ylabel: str):
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, train_vals, label="train")
            plt.plot(epochs, test_vals, label="test")
            plt.xlabel("epoch")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.savefig(tmp_file.name)
            plt.close()
            artifact_path = f"metrics/{title.lower().replace(' ', '_')}.png"
            # Parent run is already active, just log directly
            mlflow.log_artifact(tmp_file.name, artifact_path=os.path.dirname(artifact_path))
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass

        plot_metric(self.metric_history["train_loss"], self.metric_history["test_loss"], "Loss", "loss")
        plot_metric(self.metric_history["train_recall"], self.metric_history["test_recall"], "Recall", "recall")
        plot_metric(self.metric_history["train_precision"], self.metric_history["test_precision"], "Precision", "precision")
        plot_metric(self.metric_history["train_f1"], self.metric_history["test_f1"], "F1 Score", "f1")

    def _log_sample_images(self):
        """Log 5 random sample images with their names and dimensions as MLflow artifacts."""
        if not self.mlflow_enabled:
            return
        
        # Get all image files from input directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            mlflow.log_text("No image files found in input directory", "sample_images/no_images.txt")
            return
        
        # Select 5 random images
        sample_files = random.sample(image_files, min(5, len(image_files)))
        
        # Log each sample image
        for i, image_path in enumerate(sample_files):
            try:
                # Load image and get dimensions
                img = Image.open(image_path)
                width, height = img.size
                img_array = np.array(img)
                
                # Log image dimensions as text
                dimensions_text = f"File: {os.path.basename(image_path)}\nDimensions: {width}x{height}\nChannels: {img_array.shape[2] if len(img_array.shape) == 3 else 1}"
                mlflow.log_text(dimensions_text, f"sample_images/sample_{i+1}/info.txt")
                
                # Log the actual image using mlflow.log_image() for path-agnostic logging
                # This logs the pixel data directly instead of a file reference
                mlflow.log_image(img, f"sample_images/sample_{i+1}.png")
                
            except Exception as e:
                error_text = f"Error processing {os.path.basename(image_path)}: {str(e)}"
                mlflow.log_text(error_text, f"sample_images/sample_{i+1}/error.txt")

    def _log_weight_statistics(self, batch):
        """
        Log weight statistics for monitoring class balancing.
        
        Args:
            batch: Training batch to analyze
        """
        if not self.weight_calculator.enabled:
            return
        
        try:
            # Extract batch components
            y = batch[2]
            class_info = batch[3]
            
            # Convert class_info to list of strings
            class_info_list = [c.decode('utf-8') if isinstance(c, bytes) else c.numpy().decode('utf-8') 
                               for c in class_info.numpy()]
            
            # Compute sample weights
            weights = self.weight_calculator.compute_sample_weights(
                labels=y.numpy(),
                class_info=class_info_list
            )
            
            # Compute statistics
            stats = self.weight_calculator.compute_weight_statistics(weights, y.numpy())
            
            # Print statistics
            print(f"\n📊 Sample Weight Statistics:")
            print(f"   Min weight: {stats['min']:.4f}")
            print(f"   Max weight: {stats['max']:.4f}")
            print(f"   Mean weight: {stats['mean']:.4f}")
            print(f"   Std weight: {stats['std']:.4f}")
            
            if 'anchor_mean' in stats and 'negative_mean' in stats:
                print(f"   Anchor mean weight: {stats['anchor_mean']:.4f}")
                print(f"   Negative mean weight: {stats['negative_mean']:.4f}")
            
            if 'anchor_contribution_pct' in stats:
                print(f"   Anchor contribution: {stats['anchor_contribution_pct']:.1f}%")
                print(f"   Negative contribution: {stats['negative_contribution_pct']:.1f}%")
            
            # Log to MLflow
            if self.mlflow_enabled:
                # Parent run is already active, just log directly
                for key, value in stats.items():
                    mlflow.log_metric(f"weight_stats/{key}", value, step=0)
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not log weight statistics: {e}")
    
    def _log_class_balancing_to_mlflow(self):
        """
        Log class balancing configuration and statistics to MLflow.
        
        Should be called after starting the parent MLflow run.
        """
        if not self.mlflow_enabled or not self.weight_calculator.enabled:
            return
        
        try:
            # Log basic configuration
            mlflow.log_param("class_balancing/enabled", True)
            mlflow.log_param("class_balancing/anchor_negative_balance", 
                             self.weight_calculator.anchor_negative_balance)
            mlflow.log_param("class_balancing/per_class_balance", 
                             self.weight_calculator.per_class_balance)
            mlflow.log_param("class_balancing/weighting_scheme", 
                             self.weight_calculator.weighting_scheme)
            
            # Log ENS beta if using ENS scheme
            if self.weight_calculator.weighting_scheme == 'ens':
                mlflow.log_param("class_balancing/ens_beta", 
                                self.weight_calculator.ens_beta)
            
            # Log global distribution parameters and values
            if self.weight_calculator.per_class_balance:
                strategy = self.class_balancing_config.get("global_distribution_strategy")
                sampling_pct = self.class_balancing_config.get("sampling_percentage", 0.1)
                
                mlflow.log_param("class_balancing/global_distribution_strategy", strategy)
                mlflow.log_param("class_balancing/sampling_percentage", sampling_pct)
                
                # Log actual global distribution as metrics
                if self.weight_calculator._global_class_distribution:
                    for cls, count in self.weight_calculator._global_class_distribution.items():
                        mlflow.log_metric(f"global_distribution/{cls}", count, step=0)
                    
                    # Log distribution statistics
                    counts = list(self.weight_calculator._global_class_distribution.values())
                    mlflow.log_metric("global_distribution/total_samples", sum(counts), step=0)
                    mlflow.log_metric("global_distribution/num_classes", len(counts), step=0)
                    mlflow.log_metric("global_distribution/max_class_size", max(counts), step=0)
                    mlflow.log_metric("global_distribution/min_class_size", min(counts), step=0)
                    
            print(f"   ✅ Class balancing configuration logged to MLflow")
            
        except Exception as e:
            print(f"   ⚠️ Warning: Could not log class balancing to MLflow: {e}")
    
    def _check_early_stopping(self, current_f1: float, epoch: int) -> bool:
        """
        Check if training should stop early based on F1-score improvement.
        
        Args:
            current_f1: Current epoch's F1-score
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if not self.early_stopping_enabled:
            return False
        
        # Check if F1 improved by more than min_delta
        if current_f1 > self.early_stopping_best_value + self.early_stopping_min_delta:
            # Improvement detected
            self.early_stopping_best_value = current_f1
            self.early_stopping_counter = 0
            return False
        else:
            # No improvement
            self.early_stopping_counter += 1
            print(f"⚠️  Early stopping: {self.early_stopping_counter}/{self.early_stopping_patience} epochs without improvement")
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\n{'='*70}")
                print(f"🛑 Early stopping triggered!")
                print(f"   No improvement in F1-score for {self.early_stopping_patience} epochs")
                print(f"   Best F1: {self.early_stopping_best_value:.6f}")
                print(f"   Stopping at epoch {epoch}")
                print(f"{'='*70}\n")
                
                # Log to MLflow
                if self.mlflow_enabled:
                    mlflow.log_param("early_stopped", True)
                    mlflow.log_param("early_stop_epoch", epoch)
                    mlflow.log_metric("early_stop_best_f1", self.early_stopping_best_value)
                
                return True
        
        return False
    
    def _compute_anchor_negative_weights_tf(self, labels: tf.Tensor) -> tf.Tensor:
        """
        Compute anchor/negative weights using pure TensorFlow ops.
        
        This replicates the logic from anchor_negative_weights.py but runs entirely on GPU.
        Uses self.weight_calculator.anchor_target_ratio to control the contribution split.
        
        Args:
            labels: Tensor of labels (1.0 for anchors/positives, 0.0 for negatives)
            
        Returns:
            Tensor of per-sample weights
        """
        ratio = self.weight_calculator.anchor_target_ratio
        num_anchors = tf.reduce_sum(tf.cast(labels == 1.0, tf.float32))
        num_negatives = tf.reduce_sum(tf.cast(labels == 0.0, tf.float32))
        total = num_anchors + num_negatives
        
        anchor_weight = ratio * total / tf.maximum(num_anchors, 1e-6)
        negative_weight = (1.0 - ratio) * total / tf.maximum(num_negatives, 1e-6)
        
        # Normalize so weights sum to 1
        total_weight = anchor_weight + negative_weight
        anchor_weight = anchor_weight / total_weight
        negative_weight = negative_weight / total_weight
        
        # Apply to each sample based on label
        anchor_mask = tf.cast(labels == 1.0, tf.float32)
        negative_mask = tf.cast(labels == 0.0, tf.float32)
        return anchor_mask * anchor_weight + negative_mask * negative_weight
    
    @tf.function
    def train_step(self, batch):
        """
        Perform a single training step with optional sample weighting.
        
        Uses pure TensorFlow operations for weight computation to maximize GPU utilization.
        
        Args:
            batch: Tuple of (img1, img2, label, class_info)
            
        Returns:
            Tuple of (Weighted loss value, predictions)
        """
        if self.siamese_model is None:
            raise ValueError(
                "No siamese model available for training. Initialize trainer with a siamese_model."
            )

        # All computations inside tape using pure TensorFlow ops for GPU efficiency
        with tf.GradientTape() as tape:
            # Extract batch components
            x = [batch[0], batch[1]]
            y = batch[2]
            class_info = batch[3]
            
            # Forward pass
            yhat = self.siamese_model(x, training=True)
            
            # Compute per-sample loss
            per_sample_loss = self.loss_function(y, yhat)
            
            # Apply sample weights if enabled (all TensorFlow ops)
            if self.weight_calculator.enabled:
                weights = tf.ones_like(per_sample_loss)
                
                # Anchor/Negative balance (pure TensorFlow)
                if self.weight_calculator.anchor_negative_balance:
                    an_weights = self._compute_anchor_negative_weights_tf(y)
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
                # No weighting, just reduce mean
                loss = tf.reduce_mean(per_sample_loss)

            # Scale loss if using mixed precision
            if self.mixed_precision_enabled:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if self.mixed_precision_enabled:
            scaled_grad = tape.gradient(scaled_loss, self.siamese_model.trainable_variables)
            grad = self.optimizer.get_unscaled_gradients(scaled_grad)
        else:
            grad = tape.gradient(loss, self.siamese_model.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grad, self.siamese_model.trainable_variables))
        return loss, yhat

    def fit(self, bat_type: str, augmented_data: bool, data_source: str):
        self._start_parent_run(bat_type, augmented_data, data_source)
        try:
            self.train()
            self.test()
            if self._artifact_retention.get("save_final_model"):
                self.save_model()
        finally:
            self._end_parent_run()

    def train(self):
        if self.siamese_model is None:
            raise ValueError(
                "No siamese model available for training. Initialize trainer with a siamese_model."
            )
        self._epoch_canonical_subdir = {}
        if (
            self.model_output_dir
            and self._artifact_retention.get("save_tf_checkpoints")
            and self.checkpoint_dir
        ):
            self._checkpoint_manager = tf.train.CheckpointManager(
                self.checkpoint, self.checkpoint_dir, max_to_keep=1
            )
        else:
            self._checkpoint_manager = None

        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            print("\n Epoch {}/{}".format(epoch, self.num_epochs))
            progbar = tf.keras.utils.Progbar(len(self.train_batches))

            r = Recall()
            p = Precision()
            total_train_loss = 0.0
            num_train_batches = 0
            
            # Log weight statistics for the first batch if class balancing is enabled
            weight_stats_logged = False

            for idx, batch in enumerate(self.train_batches):
                # Log weight statistics for first batch of first epoch
                if idx == 0 and epoch == 1 and self.weight_calculator.enabled and not weight_stats_logged:
                    self._log_weight_statistics(batch)
                    weight_stats_logged = True
                
                loss, yhat = self.train_step(batch)
                total_train_loss += self._to_float(loss)
                num_train_batches += 1
                
                # Log batch loss to MLflow every 100 batches (reduces GPU sync overhead)
                if self.mlflow_enabled and (idx + 1) % 100 == 0:
                    step = (epoch - 1) * len(self.train_batches) + idx
                    mlflow.log_metric("batch_loss", float(loss), step=step)
                
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                
                # Update progress bar every 100 batches or at the end (reduces output in nohup)
                if (idx + 1) % 100 == 0 or (idx + 1) == len(self.train_batches):
                    progbar.update(idx + 1)
            train_loss = total_train_loss / max(num_train_batches, 1)
            train_recall = self._to_float(r.result())
            train_precision = self._to_float(p.result())
            # Calculate F1 score
            train_f1 = self._calculate_f1(train_precision, train_recall)

            # periodic testing and checkpoints
            print(f"  Running test evaluation...")
            test_loss, test_recall, test_precision, test_f1 = self.test()
            print(f"  Test results - Loss: {test_loss:.6f}, Recall: {test_recall:.6f}, Precision: {test_precision:.6f}, F1: {test_f1:.6f}")

            self._update_metric_history(
                {"loss": train_loss, "recall": train_recall, "precision": train_precision, "f1": train_f1},
                {"loss": test_loss, "recall": test_recall, "precision": test_precision, "f1": test_f1},
            )
            self._log_epoch_metrics(
                epoch,
                {"loss": train_loss, "recall": train_recall, "precision": train_precision, "f1": train_f1},
                {"loss": test_loss, "recall": test_recall, "precision": test_precision, "f1": test_f1},
            )
            self._plot_and_log_artifacts(epoch)
            
            # Print current MLflow run info for monitoring
            if self.mlflow_enabled and self.parent_run:
                print(f"📊 MLflow Run ID: {self.parent_run.info.run_id}")
                print(f"📊 Experiment: {self.mlflow_experiment_name}")
                print(f"📊 View at: http://localhost:5000")

            # Track and save best models (SavedModel only; optional TF checkpoints via config)
            if test_loss < self.best_loss_value:
                self.best_loss_value = test_loss
                self.best_loss_epoch = epoch
                print(f"✅ New best test_loss! Loss: {test_loss:.6f} at epoch {epoch}")
                if self._artifact_retention.get("save_best_loss"):
                    self._save_best_metric_artifact("best_model_loss", epoch, save_format="tf")
                if self.mlflow_enabled:
                    mlflow.log_metric("best_loss_epoch", epoch)
                    mlflow.log_metric("best_test_loss", test_loss)

            if test_f1 > self.best_f1_value:
                self.best_f1_value = test_f1
                self.best_f1_epoch = epoch
                print(f"✅ New best test_f1! F1: {test_f1:.6f} at epoch {epoch}")
                if self._artifact_retention.get("save_best_f1"):
                    self._save_best_metric_artifact("best_model_f1", epoch, save_format="tf")
                if self.mlflow_enabled:
                    mlflow.log_metric("best_f1_epoch", epoch)
                    mlflow.log_metric("best_test_f1", test_f1)

            if test_recall > self.best_recall_value:
                self.best_recall_value = test_recall
                self.best_recall_epoch = epoch
                print(f"✅ New best test_recall! Recall: {test_recall:.6f} at epoch {epoch}")
                if self._artifact_retention.get("save_best_recall"):
                    self._save_best_metric_artifact("best_model_recall", epoch, save_format="tf")
                if self.mlflow_enabled:
                    mlflow.log_metric("best_recall_epoch", epoch)
                    mlflow.log_metric("best_test_recall", test_recall)

            if test_precision > self.best_precision_value:
                self.best_precision_value = test_precision
                self.best_precision_epoch = epoch
                print(f"✅ New best test_precision! Precision: {test_precision:.6f} at epoch {epoch}")
                if self._artifact_retention.get("save_best_precision"):
                    self._save_best_metric_artifact("best_model_precision", epoch, save_format="tf")
                if self.mlflow_enabled:
                    mlflow.log_metric("best_precision_epoch", epoch)
                    mlflow.log_metric("best_test_precision", test_precision)

            # Check early stopping (after tracking best F1)
            if self._check_early_stopping(test_f1, epoch):
                if self.early_stopping_restore_best and self._artifact_retention.get("save_best_f1"):
                    print(f"🔄 Restoring best model weights from epoch {self.best_f1_epoch}")
                    best_model_path = os.path.join(self.model_output_dir, "best_model_f1")
                    if os.path.exists(best_model_path):
                        self.siamese_model = tf.keras.models.load_model(
                            best_model_path,
                            custom_objects={
                                "L1Dist": L1Dist,
                                "BinaryFocalLoss": BinaryFocalLoss,
                            }
                        )
                        print(f"✅ Best weights restored (F1: {self.best_f1_value:.6f})")
                    else:
                        print(f"⚠️  best_model_f1 not on disk; skipping restore (save_best_f1 disabled?)")
                elif self.early_stopping_restore_best and not self._artifact_retention.get("save_best_f1"):
                    print("⚠️  restore_best_weights requested but save_best_f1 is false; skipping restore")
                break  # Exit training loop

            if self._checkpoint_manager is not None:
                self._checkpoint_manager.save()
        
        # Training completed - log final summary
        print(f"\n{'='*70}")
        if self.early_stopping_enabled and self.early_stopping_counter >= self.early_stopping_patience:
            print(f"🎯 Training stopped early at epoch {epoch}/{self.num_epochs}")
        else:
            print(f"🎯 Training completed! ({self.num_epochs} epochs)")
        print(f"{'='*70}")
        print(f"🏆 Best test_loss: {self.best_loss_value:.6f} at epoch {self.best_loss_epoch}")
        print(f"🏆 Best test_f1: {self.best_f1_value:.6f} at epoch {self.best_f1_epoch}")
        print(f"🏆 Best test_recall: {self.best_recall_value:.6f} at epoch {self.best_recall_epoch}")
        print(f"🏆 Best test_precision: {self.best_precision_value:.6f} at epoch {self.best_precision_epoch}")
        print(f"{'='*70}\n")

        self._write_training_summary_json()

        # Log final best epochs to MLflow as parameters (persisted)
        if self.mlflow_enabled:
            mlflow.log_param("final_best_loss_epoch", self.best_loss_epoch)
            mlflow.log_param("final_best_loss_value", round(self.best_loss_value, 6))
            mlflow.log_param("final_best_f1_epoch", self.best_f1_epoch)
            mlflow.log_param("final_best_f1_value", round(self.best_f1_value, 6))
            mlflow.log_param("final_best_recall_epoch", self.best_recall_epoch)
            mlflow.log_param("final_best_recall_value", round(self.best_recall_value, 6))
            mlflow.log_param("final_best_precision_epoch", self.best_precision_epoch)
            mlflow.log_param("final_best_precision_value", round(self.best_precision_value, 6))
        
        # Post-training automation: generate predictions and saliency maps
        print(f"\n{'='*70}")
        print(f"🚀 Starting post-training automation...")
        print(f"{'='*70}\n")
        self._run_post_training_automation()

    def test(self):
        if self.siamese_model is None:
            raise ValueError(
                "No siamese model available for testing. Initialize trainer with a siamese_model."
            )

        r = Recall()
        p = Precision()
        total_loss = tf.constant(0.0, dtype=tf.float32)
        num_batches = 0

        print(f"    Testing on {len(self.test_batches)} batches...")
        # Updated to unpack 4 values: (img1, img2, label, class_info)
        for batch_idx, (test_input, test_val, y_true, class_info) in enumerate(self.test_batches):
            yhat = self.siamese_model([test_input, test_val], training=False)
            
            # Use raw probabilities for metrics (TensorFlow metrics can handle probabilities)
            r.update_state(y_true, yhat)
            p.update_state(y_true, yhat)
            # Use unweighted test loss (default reduction)
            batch_loss = self.test_loss_function(y_true, yhat)
            total_loss = total_loss + tf.cast(batch_loss, tf.float32)
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"      Processed {batch_idx + 1}/{len(self.test_batches)} test batches")

        avg_loss = total_loss / tf.constant(float(num_batches), dtype=tf.float32)
        recall_val = self._to_float(r.result())
        precision_val = self._to_float(p.result())
        avg_loss_val = self._to_float(avg_loss)
        # Calculate F1 score
        f1_val = self._calculate_f1(precision_val, recall_val)
        
        print(f"    Test metrics - Batches: {num_batches}, Recall: {recall_val:.6f}, Precision: {precision_val:.6f}, F1: {f1_val:.6f}")
        return avg_loss_val, recall_val, precision_val, f1_val

    @staticmethod
    def _to_float(x):
        try:
            return float(x.numpy())  # type: ignore[attr-defined]
        except Exception:
            return float(x)
    
    @staticmethod
    def _calculate_f1(precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def _remove_path_for_resave(path: str) -> None:
        if not os.path.lexists(path):
            return
        if os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    def _save_best_metric_artifact(self, subdir_name: str, epoch: int, save_format: str = "tf") -> None:
        """
        Write a SavedModel under model_output_dir/subdir_name, or symlink to the first
        physical save for this epoch when multiple metrics improve on the same epoch.
        """
        if self.model_output_dir is None:
            raise RuntimeError("model_output_dir not set")
        dest = os.path.join(self.model_output_dir, subdir_name)

        if epoch not in self._epoch_canonical_subdir:
            self._remove_path_for_resave(dest)
            self.siamese_model.save(dest, save_format=save_format)
            self._epoch_canonical_subdir[epoch] = subdir_name
            return

        canon_name = self._epoch_canonical_subdir[epoch]
        if canon_name == subdir_name:
            return
        self._remove_path_for_resave(dest)
        os.symlink(canon_name, dest, target_is_directory=True)

    def _write_training_summary_json(self) -> None:
        """Small manifest of best-metric epochs and on-disk layout (paths relative to run dir)."""
        if self.model_output_dir is None:
            return

        def describe(subdir: str) -> Dict[str, Any]:
            p = os.path.join(self.model_output_dir, subdir)
            out: Dict[str, Any] = {"relative_path": subdir, "exists": os.path.lexists(p)}
            if os.path.lexists(p):
                out["is_symlink"] = os.path.islink(p)
                if out["is_symlink"]:
                    out["symlink_target"] = os.readlink(p)
            return out

        summary = {
            "model_output_dir": self.model_output_dir,
            "best_loss": {
                "epoch": self.best_loss_epoch,
                "value": self.best_loss_value,
                **describe("best_model_loss"),
            },
            "best_f1": {
                "epoch": self.best_f1_epoch,
                "value": self.best_f1_value,
                **describe("best_model_f1"),
            },
            "best_recall": {
                "epoch": self.best_recall_epoch,
                "value": self.best_recall_value,
                **describe("best_model_recall"),
            },
            "best_precision": {
                "epoch": self.best_precision_epoch,
                "value": self.best_precision_value,
                **describe("best_model_precision"),
            },
            "artifact_retention": dict(self._artifact_retention),
        }
        path = os.path.join(self.model_output_dir, "training_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def save_model(self, name="siamesemodelv2", version=None, save_format="tf"):
        if self.siamese_model is None:
            raise ValueError(
                "No siamese model available for saving. Initialize trainer with a siamese_model."
            )

        save_dir = self.model_output_dir
        os.makedirs(save_dir, exist_ok=True)

        if version:
            name = f"{name}_v{version}"
        self.siamese_model.save(os.path.join(save_dir, name), save_format=save_format)
    
    def _run_post_training_automation(self):
        """
        Run post-training automation: generate predictions and saliency maps.
        """
        try:
            # Import post-training modules
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            app_dir = os.path.dirname(current_dir)
            sys.path.insert(0, app_dir)
            
            from app.generate_predictions import generate_predictions_from_config
            
            # Get best model path (best_model_f1)
            best_model_path = os.path.join(self.model_output_dir, "best_model_f1")
            if not os.path.exists(best_model_path):
                print(f"⚠️  Best model not found at {best_model_path}, skipping post-training automation")
                return
            
            print(f"📦 Using best model: {best_model_path}")
            
            # Get config for input paths
            cfg = load_config()
            
            # Get bat-type-specific input paths
            bat_key = 'mauritius' if self.bat_type == 'm' else 'rousettus'
            input_paths = cfg.siamese_network.input_paths[bat_key]
            
            pred_config = cfg.siamese_network.generate_predictions
            saliency_config = cfg.siamese_network.saliency_maps
            
            # 1. Generate predictions
            print(f"\n📊 Generating predictions...")
            try:
                # Use best_f1_epoch as model_version
                model_version = self.best_f1_epoch
                print(f"   Using model version (best F1 epoch): {model_version}")
                
                result = generate_predictions_from_config(
                    model_path=best_model_path,
                    output_dir=self.model_output_dir,
                    bat_type=self.bat_type,
                    source=self.data_source,
                    background="original",  # Always use original background for evaluation
                    model_version=model_version,
                    include_subdirs=pred_config.get("subdirs"),
                    verbose=pred_config.get("verbose", False),
                    max_pairs=pred_config.get("max_pairs"),
                )
                if isinstance(result, tuple):
                    csv_path, plot_path = result
                    print(f"✅ Predictions saved to: {csv_path}")
                    if plot_path:
                        print(f"✅ Confusion matrix saved to: {plot_path}")
                else:
                    print(f"✅ Predictions saved to: {result}")
            except Exception as e:
                print(f"❌ Error generating predictions: {e}")
                import traceback
                traceback.print_exc()
            
            # 2. Generate saliency maps
            print(f"\n🎨 Generating saliency maps...")
            try:
                # Create saliency subdirectory
                saliency_output_dir = os.path.join(self.model_output_dir, "saliency")
                os.makedirs(saliency_output_dir, exist_ok=True)
                
                # Get original background input directory
                original_bg_input = input_paths.get("original_bg_input")
                if not original_bg_input or not os.path.exists(original_bg_input):
                    print(f"⚠️  Original background input directory not found: {original_bg_input}")
                    print(f"   Skipping saliency map generation")
                    return
                
                # Generate saliency maps using per-bat method
                from visualization.saliency import SiameseModelSaliencyMapCreator
                import tensorflow as tf
                from siamese_core.network import L1Dist
                
                # Load model for saliency
                from siamese_training.focal_loss import BinaryFocalLoss as _BFL
                model = tf.keras.models.load_model(
                    best_model_path,
                    custom_objects={
                        "L1Dist": L1Dist,
                        "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
                        "BinaryFocalLoss": _BFL,
                    },
                )
                
                # Detect input size
                input_shape = model.input_shape[0]
                input_size = input_shape[1] if len(input_shape) > 1 else SIAMESE_INPUT_EDGE_LENGTH
                
                # Create saliency creator
                saliency_creator = SiameseModelSaliencyMapCreator(
                    model=model,
                    input_dir_path=original_bg_input,
                    output_dir_path=saliency_output_dir,
                    nesting=None,
                    sample_size=saliency_config.get("sample_size", 25),
                    fast_mode=saliency_config.get("fast_mode", False),
                    input_size=input_size,
                    integration_steps=saliency_config.get("integration_steps"),
                    smoothing_samples=saliency_config.get("smoothing_samples"),
                )
                
                # Generate per-bat saliency images
                method = saliency_config.get("method", "integrated_gradients")
                smoothing = True  # Default to smoothing unless explicitly disabled
                samples_per_bat = saliency_config.get("samples_per_bat")
                
                print(f"   Using method: {method}")
                if samples_per_bat is not None:
                    print(f"   Sampling {samples_per_bat} images per bat class")
                output_files = saliency_creator.generate_per_bat_saliency_images(
                    method=method,
                    smoothing=smoothing,
                    samples_per_bat=samples_per_bat,
                )
                
                print(f"✅ Generated {len(output_files)} saliency images in: {saliency_output_dir}")
                
            except Exception as e:
                print(f"❌ Error generating saliency maps: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*70}")
            print(f"✅ Post-training automation completed!")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"❌ Error in post-training automation: {e}")
            import traceback
            traceback.print_exc()
