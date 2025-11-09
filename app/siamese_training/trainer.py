"""
Siamese network trainer module.
"""

import os
import tempfile
import random
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall  # type: ignore[reportMissingImports]
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from PIL import Image

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

from app.siamese_core.network import SiameseNetwork, L1Dist
from app.siamese_data.data_splitter import SiameseNetworkTrainingDataSplitter
from app.siamese_data.class_weights import ClassWeightCalculator
from app.config.loader import load_config


class SiameseNetworkTrainer:
    def __init__(
        self,
        bat_type: str,
        augmented_data: bool,
        data_source: str,
        input_dir: Optional[str] = None,
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_function=tf.losses.BinaryCrossentropy(),
    ):
        cfg = load_config()
        sn_train = cfg.siamese_network.training

        # Data dirs
        self.input_dir = input_dir or sn_train.get("input_dir")
        if not self.input_dir or not os.path.exists(self.input_dir):
            raise ValueError(f"Invalid or missing training input_dir: {self.input_dir}")

        # Training hyperparameters
        self.num_epochs = sn_train.get("epochs", 80)
        self.batch_size = sn_train.get("batch_size", 16)

        # Output paths
        model_output_dir = sn_train.get("output_dir")
        if not model_output_dir:
            raise ValueError("Missing siamese_network.training.output_dir in config")
        os.makedirs(model_output_dir, exist_ok=True)
        self.model_output_dir = model_output_dir

        # Checkpoints path (inside model output root)
        self.checkpoint_dir = os.path.join(self.model_output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # MLflow setup
        self.mlflow_enabled = cfg.mlflow.enabled
        self.mlflow_tracking_uri = self._resolve_mlflow_tracking_uri(cfg.mlflow.tracking_uri)
        self.mlflow_experiment_name = self._build_experiment_name(bat_type, augmented_data, data_source)
        self.parent_run = None
        self.metric_history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_recall": [],
            "train_precision": [],
            "test_loss": [],
            "test_recall": [],
            "test_precision": [],
        }
        self.current_epoch: int = 0

        # Model, optimizer, loss
        self.siamese_model = SiameseNetwork(L1Dist()).model
        self.optimizer = optimizer
        # Use reduction='none' to get per-sample loss for weighted loss calculation
        self.loss_function = tf.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # Separate loss for testing (unweighted, default reduction)
        self.test_loss_function = tf.losses.BinaryCrossentropy()

        # Class balancing setup
        class_balancing_config = sn_train.get("class_balancing", {})
        self.weight_calculator = ClassWeightCalculator(class_balancing_config)
        
        if self.weight_calculator.enabled:
            print(f"🔧 Class balancing enabled:")
            print(f"   - Anchor/Negative balance: {self.weight_calculator.anchor_negative_balance}")
            print(f"   - Per-class balance: {self.weight_calculator.per_class_balance}")
            print(f"   - Weighting scheme: {self.weight_calculator.weighting_scheme}")
            if self.weight_calculator.weighting_scheme == 'ens':
                print(f"   - ENS beta: {self.weight_calculator.ens_beta}")

        # Data
        print(f"🔧 Loading data from: {self.input_dir}")
        
        # Check GPU availability
        print(f"🔧 TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
        print(f"🔧 TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        
        self.data_splitter = SiameseNetworkTrainingDataSplitter(
            [self.input_dir], training_portion=0.7, mode="permutation"
        )
        train_data = self.data_splitter.train_data
        test_data = self.data_splitter.test_data
        if train_data is None or test_data is None:
            raise ValueError("Data splitter returned no train/test data")
        
        print(f"🔧 Creating data batches...")
        self.train_batches = train_data.batch(self.batch_size).prefetch(8)
        self.test_batches = test_data.batch(self.batch_size).prefetch(8)
        print(f"✅ Data loading completed")
        
        # Print class distribution if class balancing is enabled
        if self.weight_calculator.enabled:
            train_dist = self.data_splitter.get_class_distribution('train')
            print(f"\n📊 Training class distribution:")
            for cls, count in sorted(train_dist.items()):
                print(f"   {cls}: {count} samples")

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
        # Validate inputs
        if bat_type not in ['m', 'r']:
            raise ValueError("bat_type must be 'm' (mauritius) or 'r' (rousettus)")
        if data_source not in ['video', 'still']:
            raise ValueError("data_source must be 'video' or 'still'")
        
        # Map bat_type to full name
        bat_name = "mauritius" if bat_type == 'm' else "rousettus"
        
        # Map augmented_data to string
        aug_str = "augmented" if augmented_data else "no_aug"
        
        # Build experiment name from template
        experiment_name = f"siamese_{bat_name}_{data_source}_{aug_str}"
        return experiment_name

    def _start_parent_run(self, bat_type: str, augmented_data: bool, data_source: str):
        if not self.mlflow_enabled:
            return
        print(f"🔧 Starting MLflow run...")
        print(f"📊 Experiment name: {self.mlflow_experiment_name}")
        print(f"📊 Tracking URI: {self.mlflow_tracking_uri}")
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        
        # Build descriptive run name from parameters
        bat_name = "mauritius" if bat_type == 'm' else "rousettus"
        aug_str = "augmented" if augmented_data else "no_aug"
        run_name = f"training_{bat_name}_{data_source}_{aug_str}"
        
        print(f"📊 Run name: {run_name}")
        self.parent_run = mlflow.start_run(run_name=run_name)
        self._log_hyperparameters(bat_type, augmented_data, data_source)

    def _end_parent_run(self):
        if not self.mlflow_enabled:
            return
        if self.parent_run is not None:
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
            "train_val_split": sn_train.get("train_val_split", 0.7),
            "pair_mode": sn_train.get("pair_mode", "permutation"),
            "max_samples_per_class": sn_train.get("max_samples_per_class", 0),
            "input_dir": self.input_dir,
            "output_dir": self.model_output_dir,
            "bat_type": bat_type,
            "augmented_data": augmented_data,
            "data_source": data_source,
            "experiment_name": self.mlflow_experiment_name,
        }
        mlflow.log_params(params)
        self._log_sample_images()

    def _log_epoch_metrics(self, epoch: int, train: Dict[str, float], test: Dict[str, float]):
        if not self.mlflow_enabled:
            return
        
        # Log metrics to parent run for step-wise tracking and easy comparison
        # Parent run is already active from _start_parent_run, so log directly
        mlflow.log_metrics(
            {
                "train_loss": float(train["loss"]),
                "train_recall": float(train["recall"]),
                "train_precision": float(train["precision"]),
                "test_loss": float(test["loss"]),
                "test_recall": float(test["recall"]),
                "test_precision": float(test["precision"]),
            },
            step=epoch,
        )
        
        # Also log detailed metrics in nested run for this epoch
        with mlflow.start_run(run_name=f"epoch_{epoch}", nested=True):
            mlflow.log_param("epoch", int(epoch))
            mlflow.log_metrics(
                {
                    "loss": float(train["loss"]),
                    "recall": float(train["recall"]),
                    "precision": float(train["precision"]),
                }
            )
            mlflow.log_metrics(
                {
                    "loss_test": float(test["loss"]),
                    "recall_test": float(test["recall"]),
                    "precision_test": float(test["precision"]),
                }
            )

    def _update_metric_history(self, train: Dict[str, float], test: Dict[str, float]):
        self.metric_history["train_loss"].append(float(train["loss"]))
        self.metric_history["train_recall"].append(float(train["recall"]))
        self.metric_history["train_precision"].append(float(train["precision"]))
        self.metric_history["test_loss"].append(float(test["loss"]))
        self.metric_history["test_recall"].append(float(test["recall"]))
        self.metric_history["test_precision"].append(float(test["precision"]))

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
    
    def train_step(self, batch):
        """
        Perform a single training step with optional sample weighting.
        
        Args:
            batch: Tuple of (img1, img2, label, class_info)
            
        Returns:
            Weighted loss value
        """
        if self.siamese_model is None:
            raise ValueError(
                "No siamese model available for training. Initialize trainer with a siamese_model."
            )

        with tf.GradientTape() as tape:
            # Extract batch components
            x = [batch[0], batch[1]]
            y = batch[2]
            class_info = batch[3]
            
            # Forward pass
            yhat = self.siamese_model(x, training=True)
            
            # Compute per-sample loss
            per_sample_loss = self.loss_function(y, yhat)
            
            # Apply sample weights if enabled
            if self.weight_calculator.enabled:
                # Convert class_info tensor to list of strings
                class_info_list = [c.decode('utf-8') if isinstance(c, bytes) else c.numpy().decode('utf-8') 
                                   for c in class_info.numpy()]
                
                # Compute sample weights
                weights = self.weight_calculator.compute_sample_weights(
                    labels=y.numpy(),
                    class_info=class_info_list
                )
                
                # Apply weights to loss
                weights_tf = tf.constant(weights, dtype=tf.float32)
                weighted_loss = per_sample_loss * weights_tf
                loss = tf.reduce_mean(weighted_loss)
            else:
                # No weighting, just reduce mean
                loss = tf.reduce_mean(per_sample_loss)

        grad = tape.gradient(loss, self.siamese_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.siamese_model.trainable_variables))
        return loss

    def fit(self, bat_type: str, augmented_data: bool, data_source: str):
        self._start_parent_run(bat_type, augmented_data, data_source)
        try:
            self.train()
            self.test()
            self.save_model()
        finally:
            self._end_parent_run()

    def train(self):
        if self.siamese_model is None:
            raise ValueError(
                "No siamese model available for training. Initialize trainer with a siamese_model."
            )
        min_train_loss = 1
        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            print("\n Epoch {}/{}".format(epoch, self.num_epochs))
            progbar = tf.keras.utils.Progbar(len(self.train_batches))

            r = Recall()
            p = Precision()
            
            # Log weight statistics for the first batch if class balancing is enabled
            weight_stats_logged = False

            for idx, batch in enumerate(self.train_batches):
                # Log weight statistics for first batch of first epoch
                if idx == 0 and epoch == 1 and self.weight_calculator.enabled and not weight_stats_logged:
                    self._log_weight_statistics(batch)
                    weight_stats_logged = True
                    
                print(f"  Processing batch {idx + 1}/{len(self.train_batches)}")
                loss = self.train_step(batch)
                # Print loss after getting it from train_step
                print(f"    Loss: {float(loss.numpy()):.6f}")
                yhat = self.siamese_model.predict(x=batch[:2])
                
                # Use raw probabilities for metrics (TensorFlow metrics can handle probabilities)
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                progbar.update(idx + 1)
            train_loss = self._to_float(loss)
            train_recall = self._to_float(r.result())
            train_precision = self._to_float(p.result())

            self.save_model(version=epoch, save_format="tf")

            # periodic testing and checkpoints
            print(f"  Running test evaluation...")
            test_loss, test_recall, test_precision = self.test()
            print(f"  Test results - Loss: {test_loss:.6f}, Recall: {test_recall:.6f}, Precision: {test_precision:.6f}")

            self._update_metric_history(
                {"loss": train_loss, "recall": train_recall, "precision": train_precision},
                {"loss": test_loss, "recall": test_recall, "precision": test_precision},
            )
            self._log_epoch_metrics(
                epoch,
                {"loss": train_loss, "recall": train_recall, "precision": train_precision},
                {"loss": test_loss, "recall": test_recall, "precision": test_precision},
            )
            self._plot_and_log_artifacts(epoch)
            
            # Print current MLflow run info for monitoring
            if self.mlflow_enabled and self.parent_run:
                print(f"📊 MLflow Run ID: {self.parent_run.info.run_id}")
                print(f"📊 Experiment: {self.mlflow_experiment_name}")
                print(f"📊 View at: http://localhost:5000")

            if test_loss <= min_train_loss:
                min_train_loss = test_loss
                self.checkpoint.save(file_prefix=os.path.join(self.checkpoint_dir, "min"))

            if epoch % 10 == 0:
                self.checkpoint.save(file_prefix=os.path.join(self.checkpoint_dir, "periodic"))

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
            yhat = self.siamese_model.predict([test_input, test_val])
            
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
        
        print(f"    Test metrics - Batches: {num_batches}, Recall: {recall_val:.6f}, Precision: {precision_val:.6f}")
        return avg_loss_val, recall_val, precision_val

    @staticmethod
    def _to_float(x):
        try:
            return float(x.numpy())  # type: ignore[attr-defined]
        except Exception:
            return float(x)

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
