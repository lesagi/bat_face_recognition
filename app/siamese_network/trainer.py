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

from .network import SiameseNetwork, L1Dist
from .data_splitter import SiameseNetworkTrainingDataSplitter
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
        self.mlflow_tracking_uri = cfg.mlflow.tracking_uri
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
        self.loss_function = loss_function

        # Data
        print(f"🔧 Loading data from: {self.input_dir}")
        
        # Check GPU availability
        print(f"🔧 TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
        print(f"🔧 TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        
        data_splitter = SiameseNetworkTrainingDataSplitter(
            [self.input_dir], training_portion=0.7, mode="permutation"
        )
        train_data = data_splitter.train_data
        test_data = data_splitter.test_data
        if train_data is None or test_data is None:
            raise ValueError("Data splitter returned no train/test data")
        
        print(f"🔧 Creating data batches...")
        self.train_batches = train_data.batch(self.batch_size).prefetch(8)
        self.test_batches = test_data.batch(self.batch_size).prefetch(8)
        print(f"✅ Data loading completed")

        # TF checkpoint
        self.checkpoint = tf.train.Checkpoint(opt=self.optimizer, siamese_model=self.siamese_model)

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
        self.parent_run = mlflow.start_run(run_name="training")
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
                with Image.open(image_path) as img:
                    width, height = img.size
                    img_array = np.array(img)
                
                # Create artifact directory
                artifact_dir = f"sample_images/sample_{i+1}"
                
                # Log image dimensions as text
                dimensions_text = f"File: {os.path.basename(image_path)}\nDimensions: {width}x{height}\nChannels: {img_array.shape[2] if len(img_array.shape) == 3 else 1}"
                mlflow.log_text(dimensions_text, f"{artifact_dir}/info.txt")
                
                # Log the actual image
                mlflow.log_artifact(image_path, artifact_dir)
                
            except Exception as e:
                error_text = f"Error processing {os.path.basename(image_path)}: {str(e)}"
                mlflow.log_text(error_text, f"sample_images/sample_{i+1}/error.txt")

    @tf.function
    def train_step(self, batch):
        if self.siamese_model is None:
            raise ValueError(
                "No siamese model available for training. Initialize trainer with a siamese_model."
            )

        with tf.GradientTape() as tape:
            x = batch[:2]
            y = batch[2]
            yhat = self.siamese_model(x, training=True)
            loss = self.loss_function(y, yhat)

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

            for idx, batch in enumerate(self.train_batches):
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
        for batch_idx, (test_input, test_val, y_true) in enumerate(self.test_batches):
            yhat = self.siamese_model.predict([test_input, test_val])
            
            # Use raw probabilities for metrics (TensorFlow metrics can handle probabilities)
            r.update_state(y_true, yhat)
            p.update_state(y_true, yhat)
            batch_loss = self.loss_function(y_true, yhat)
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
