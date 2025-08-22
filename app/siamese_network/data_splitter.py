"""
Data splitter for Siamese network training.
"""

import os
import random
import tensorflow as tf
from itertools import combinations, permutations

from config.loader import load_config


def get_files_from_dir(directory):
    """Get all files from a directory, excluding .DS_Store files."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and not f.startswith(".DS_Store")
    ]


def preprocess_siamese_input(file_path):
    """Preprocess a single image file path into a tensor."""
    try:
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image - try both PNG and JPEG
        try:
            img = tf.io.decode_png(byte_img)
        except:
            img = tf.io.decode_jpeg(byte_img)

        # Read input size from config locally
        cfg = load_config()
        input_edge = cfg.siamese_network.model.get("input_edge_length", 224)

        # Preprocessing steps - resizing the image
        img = tf.image.resize(img, (input_edge, input_edge))
        # Scale image to be between 0 and 1
        img = img / 255.0

        # Ensure the image has 3 channels
        img = tf.image.convert_image_dtype(img, tf.float32)
        if tf.shape(img)[2] == 1:  # Grayscale
            img = tf.repeat(img, 3, axis=2)
        elif tf.shape(img)[2] == 4:  # RGBA
            img = img[:, :, :3]

        return img
    except Exception as e:
        tf.print(f"Error processing {file_path}: {e}")
        # Return a black image as fallback (use local config)
        cfg = load_config()
        input_edge = cfg.siamese_network.model.get("input_edge_length", 224)
        return tf.zeros((input_edge, input_edge, 3), dtype=tf.float32)


def preprocess_twin_input_function(input_img_path, validation_img_path, label):
    """Preprocess twin input function for siamese pairs."""
    return (
        preprocess_siamese_input(input_img_path),
        preprocess_siamese_input(validation_img_path),
        label,
    )


class SiameseNetworkTrainingDataSplitter:
    """
    Split the data into training and validation sets
    Each images directory inside {images_directories_collection} will be considered as a class
    Siamese network requires pairs of images with a label
    So we will create pairs of images from the same class (directory) with a label of 1
    And pairs of images from different classes (different directories) with a label of 0
    :parameter classes_images_dir: str
    :parameter mode: combination | permutation
    :return: Dataset iterator
    """

    def __init__(
        self, images_dirs_paths_list, training_portion=0.7, mode="combination"
    ):
        self.images_directories_collection = images_dirs_paths_list
        self.training_portion = training_portion
        self.mode = mode
        self.labelled_data = None
        self.per_dir_pairs = {}
        self.class_dirs = []

        # "Class" in this context refer to a folder of a specific bat,
        # so we consider each bat as a class
        for class_dirs in images_dirs_paths_list:
            dirs = [
                os.path.join(class_dirs, class_dir_name)
                for class_dir_name in os.listdir(class_dirs)
                if os.path.isdir(os.path.join(class_dirs, class_dir_name))
            ]
            self.class_dirs.extend(dirs)

        for images_dir in self.class_dirs:
            anchors = self.__create_anchor_pairs(get_files_from_dir(images_dir))
            # Apply preprocessing to convert file paths to image tensors
            anchors = anchors.map(preprocess_twin_input_function, num_parallel_calls=tf.data.AUTOTUNE)
            self.__add_to_self_labelled_data(anchors)

        dir_pairs = combinations(self.class_dirs, 2)
        for dir_a, dir_b in dir_pairs:
            files_dir_a = get_files_from_dir(dir_a)
            files_dir_b = get_files_from_dir(dir_b)
            negatives = self.__create_negative_pairs(files_dir_a, files_dir_b, False)
            if self.mode == "permutation":
                negatives = negatives.concatenate(
                    self.__create_negative_pairs(files_dir_b, files_dir_a, False)
                )
            # Apply preprocessing to convert file paths to image tensors
            negatives = negatives.map(preprocess_twin_input_function, num_parallel_calls=tf.data.AUTOTUNE)
            self.__add_to_self_labelled_data(negatives)

        self.__build_train_test_data()

    def __create_anchor_pairs(self, anchor_images_list):
        pairs = (
            list(combinations(anchor_images_list, 2))
            if self.mode == "combination"
            else list(permutations(anchor_images_list, 2))
        )
        pairs_a = tf.data.Dataset.from_tensor_slices([a for a, b in pairs])
        pairs_b = tf.data.Dataset.from_tensor_slices([b for a, b in pairs])
        samples_count = len(pairs)
        dataset = tf.data.Dataset.zip((pairs_a, pairs_b, tf.data.Dataset.from_tensor_slices(tf.ones(samples_count))))

        # Local config for size limit
        cfg = load_config()
        max_limit = cfg.siamese_network.training.get("max_data_size_limit")
        if max_limit is not None:
            dataset = dataset.take(max_limit)
        print(f"anchors pairs count: {dataset.cardinality().numpy()}")
        return dataset

    def __create_negative_pairs(self, list_a, list_b, should_shuffle=True):
        list_a_size = len(list_a)
        data_set_a = tf.data.Dataset.from_tensor_slices(list_a)

        list_b_size = len(list_b)
        data_set_b = tf.data.Dataset.from_tensor_slices(list_b)

        if should_shuffle:
            data_set_a = data_set_a.shuffle(data_set_a.cardinality(), seed=random.randint(20, 80))
            data_set_b = data_set_b.shuffle(data_set_b.cardinality(), seed=random.randint(20, 80))

        # Local config for size limit
        cfg = load_config()
        max_limit = cfg.siamese_network.training.get("max_data_size_limit")

        min_size = min(list_a_size, list_b_size)
        if max_limit is not None:
            min_size = min(min_size, max_limit)

        data_set_a = data_set_a.take(min_size)
        data_set_b = data_set_b.take(min_size)

        dataset = tf.data.Dataset.zip((data_set_a, data_set_b, tf.data.Dataset.from_tensor_slices(tf.zeros(min_size))))
        print(f"negative pairs count: {dataset.cardinality().numpy()}, min size: {min_size}")
        return dataset

    def __build_train_test_data(self):
        if self.labelled_data is not None:
            labelled_data_size = self.labelled_data.cardinality().numpy()
            self.labelled_data = self.labelled_data.shuffle(
                buffer_size=labelled_data_size, seed=random.randint(20, 80)
            )
            # Build dataloader pipeline
            data_size = self.labelled_data.cardinality().numpy()
            train_size = round(data_size * self.training_portion)

            # Split the data into training and testing sets
            self.train_data = self.labelled_data.take(train_size)
            self.test_data = self.labelled_data.skip(train_size)
        else:
            self.train_data = None
            self.test_data = None

    def __add_to_self_labelled_data(self, data):
        if not self.labelled_data:
            self.labelled_data = data
        else:
            self.labelled_data = self.labelled_data.concatenate(data)
