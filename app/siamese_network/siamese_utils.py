"""
Siamese network utilities module.
"""

import os
import tensorflow as tf
from .network import SiameseNetwork, L1Dist
from .trainer import SiameseNetworkTrainer
from .data_splitter import SiameseNetworkTrainingDataSplitter
from .paths import SIAMESE_PATHS


def build_siamese_network():
    return SiameseNetwork(L1Dist())


def get_trained_siamese_network(
    siamese_network_model,
    train_data_batches,
    test_data_batches,
    segmentation_model_path=None,
):
    network_trainer = SiameseNetworkTrainer(
        siamese_network_model, segmentation_model_path=segmentation_model_path
    )
    network_trainer.train(train_data_batches)
    network_trainer.test(test_data_batches)
    network_trainer.save_model()
    return network_trainer.siamese_model


def load_model(segmentation_model_path=None):
    if os.path.exists(SIAMESE_PATHS["TRAINING"]["MODEL"]):
        trained_model = tf.keras.models.load_model(
            SIAMESE_PATHS["TRAINING"]["MODEL"],
            custom_objects={
                "L1Dist": L1Dist,
                "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
            },
        )
        print("Loaded model from disk")
    else:
        print("Starting training")
        train_data_batches, test_data_batches = build_train_test_data_batches(
            segmentation_model_path=segmentation_model_path
        )
        siamese_model = build_siamese_network().model
        trained_model = get_trained_siamese_network(
            siamese_model,
            train_data_batches,
            test_data_batches,
            segmentation_model_path,
        )
        print("Trained a new model")

    return trained_model


def build_train_test_data_batches(segmentation_model_path=None):
    """Build training and test data batches with optional preprocessing pipeline."""

    # Create trainer instance to get preprocessing function if model path provided
    if segmentation_model_path:
        # Create a dummy model just to get the preprocessing function
        dummy_trainer = SiameseNetworkTrainer(
            siamese_model=None, segmentation_model_path=segmentation_model_path
        )
        preprocess_function = dummy_trainer.create_preprocessing_function()
        print("✅ Using advanced preprocessing pipeline with segmentation")
    else:
        preprocess_function = None
        print("⚠️  Using basic preprocessing (no segmentation)")

    # Split the data into training and validation sets
    data_splitter = SiameseNetworkTrainingDataSplitter(
        [SIAMESE_PATHS["TRAINING"]["DATA"]],
        preprocess_twin_input_function=preprocess_function,
        training_portion=0.7,
        mode="permutation",
    )

    train_data_batches = data_splitter.train_data.batch(16).prefetch(8)
    test_data_batches = data_splitter.test_data.batch(16).prefetch(8)
    return train_data_batches, test_data_batches
