"""
Siamese network trainer module.
"""

import os
import csv
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from .network import SiameseNetwork, L1Dist
from .data_splitter import SiameseNetworkTrainingDataSplitter
from config.loader import load_config


class SiameseNetworkTrainer:
    def __init__(
        self,
        input_dir: str | None = None,
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss_function=tf.losses.BinaryCrossentropy(),
        log_file_name: str = "training_log.csv",
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

        # Logging
        self.log_file_path = os.path.join(self.model_output_dir, log_file_name)
        self._init_log_file()

        # Model, optimizer, loss
        self.siamese_model = SiameseNetwork(L1Dist()).model
        self.optimizer = optimizer
        self.loss_function = loss_function

        # Data
        data_splitter = SiameseNetworkTrainingDataSplitter(
            [self.input_dir], training_portion=0.7, mode="permutation"
        )
        self.train_batches = data_splitter.train_data.batch(self.batch_size).prefetch(8)
        self.test_batches = data_splitter.test_data.batch(self.batch_size).prefetch(8)

        # TF checkpoint
        self.checkpoint = tf.train.Checkpoint(opt=self.optimizer, siamese_model=self.siamese_model)

    def _init_log_file(self):
        if self.log_file_path:
            with open(self.log_file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Source", "Epoch", "Loss", "Recall", "Precision"])

    def _write_log(self, source, epoch, loss, recall, precision):
        if self.log_file_path:
            with open(self.log_file_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([source, epoch, loss, recall, precision])

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
        print(loss)

        grad = tape.gradient(loss, self.siamese_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.siamese_model.trainable_variables))
        return loss

    def fit(self):
        self.train()
        self.test()
        self.save_model()

    def train(self):
        if self.siamese_model is None:
            raise ValueError(
                "No siamese model available for training. Initialize trainer with a siamese_model."
            )

        min_train_loss = 1
        for epoch in range(1, self.num_epochs + 1):
            print("\n Epoch {}/{}".format(epoch, self.num_epochs))
            progbar = tf.keras.utils.Progbar(len(self.train_batches))

            r = Recall()
            p = Precision()

            for idx, batch in enumerate(self.train_batches):
                loss = self.train_step(batch)
                yhat = self.siamese_model.predict(x=batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                progbar.update(idx + 1)
            self._write_log("train", epoch, loss.numpy(), r.result().numpy(), p.result().numpy())
            self.save_model(version=epoch, save_format="tf")

            # periodic testing and checkpoints
            train_loss, train_recall, train_precision = self.test()
            self._write_log("test", epoch, train_loss, train_recall, train_precision)

            if train_loss <= min_train_loss:
                min_train_loss = train_loss
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
        total_loss = 0
        num_batches = 0

        for test_input, test_val, y_true in self.test_batches.as_numpy_iterator():
            yhat = self.siamese_model.predict([test_input, test_val])
            r.update_state(y_true, yhat)
            p.update_state(y_true, yhat)
            batch_loss = self.loss_function(y_true, yhat)
            total_loss += batch_loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss.numpy(), r.result().numpy(), p.result().numpy()

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
