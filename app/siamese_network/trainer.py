"""
Siamese network trainer module.
"""

import os
import csv
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from .network import SiameseNetwork, L1Dist
from .data_splitter import SiameseNetworkTrainingDataSplitter

# Training constants
SIAMESE_TRAIN_EPOCHS = 80

# Path constants - simplified for standalone use
TRAINING_BASE_PATH = os.path.join(
    "/home/sagilevi1/bat_face_rec_project/face_rec_rousettus_#1",
    "runs",
)
TRAINING_MODEL_PATH = os.path.join(TRAINING_BASE_PATH, "model")
TRAINING_CHECKPOINTS_PATH = os.path.join(TRAINING_BASE_PATH, "checkpoints")


class SiameseNetworkTrainer:
    def __init__(
        self,
        input_dir,
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss_function=tf.losses.BinaryCrossentropy(),
        log_file_path="training_log.csv",
    ):
        self.siamese_model = SiameseNetwork(L1Dist()).model
        self.optimizer = optimizer
        self.loss_function = loss_function

        data_splitter = SiameseNetworkTrainingDataSplitter(
            [input_dir], training_portion=0.7, mode="permutation"
        )
        self.train_batches = data_splitter.train_data.batch(16).prefetch(8)
        self.test_batches = data_splitter.test_data.batch(16).prefetch(8)

        self.checkpoint_dir = TRAINING_CHECKPOINTS_PATH
        self.checkpoint = tf.train.Checkpoint(
            opt=self.optimizer, siamese_model=self.siamese_model
        )
        self.log_file_path = os.path.join(TRAINING_MODEL_PATH, log_file_path)
        os.makedirs(TRAINING_MODEL_PATH, exist_ok=True)
        self.init_log_file()

    def init_log_file(self):
        if self.log_file_path:
            with open(self.log_file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Source", "Epoch", "Loss", "Recall", "Precision"])

    def write_log(self, source, epoch, loss, recall, precision):
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
        self.optimizer.apply_gradients(
            zip(grad, self.siamese_model.trainable_variables)
        )
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
        for epoch in range(1, SIAMESE_TRAIN_EPOCHS + 1):
            print("\n Epoch {}/{}".format(epoch, SIAMESE_TRAIN_EPOCHS))
            progbar = tf.keras.utils.Progbar(len(self.train_batches))

            r = Recall()
            p = Precision()

            for idx, batch in enumerate(self.train_batches):
                loss = self.train_step(batch)
                yhat = self.siamese_model.predict(x=batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                progbar.update(idx + 1)
            self.write_log(
                "train", epoch, loss.numpy(), r.result().numpy(), p.result().numpy()
            )
            self.save_model(version=epoch, save_format="tf")
            if self.checkpoint_dir:
                train_loss, train_recall, train_precision = self.test()
                self.write_log("test", epoch, train_loss, train_recall, train_precision)

                if train_loss <= min_train_loss:
                    min_train_loss = train_loss
                    self.checkpoint.save(
                        file_prefix=os.path.join(self.checkpoint_dir, "min")
                    )

                if epoch % 10 == 0:
                    self.checkpoint.save(
                        file_prefix=os.path.join(self.checkpoint_dir, "periodic")
                    )

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

        if version:
            name = f"{name}_v{version}"
        self.siamese_model.save(
            os.path.join(TRAINING_MODEL_PATH, name),
            save_format=save_format,
        )
