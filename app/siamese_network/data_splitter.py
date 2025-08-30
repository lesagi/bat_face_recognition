"""
Data splitter for Siamese network training.
"""

import os
import random
import re
import tensorflow as tf
from itertools import combinations, permutations
from collections import defaultdict

from config.loader import load_config


def get_files_from_dir(directory):
    """Get all files from a directory, excluding .DS_Store files."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and not f.startswith(".DS_Store")
    ]


def parse_filename_class(filename):
    """
    Parse filename to extract class information.
    
    Expected pattern: (?<type>\w)--(?<class>\w+)--(?<id>\w+)(?<aug_suffix>--aug(?<aug_id>\d{3}))?
    
    Args:
        filename: The filename to parse
        
    Returns:
        tuple: (type, class_name, id, aug_suffix) or None if parsing fails
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # capturing groups: (?<type>\w)--(?<class>\w+)--(?<id>\w+)(?<aug_suffix>--aug(?<aug_id>\d{3}))?
    # Pattern: type--class--id[--aug###]
    pattern = r'^(\w+)--(\w+)--(\w+)(?:--aug(\d{3}))?$'
    match = re.match(pattern, name_without_ext)
    
    if match:
        file_type, class_name, file_id, aug_suffix = match.groups()
        return file_type, class_name, file_id, aug_suffix
    else:
        return None


def group_files_by_class(file_paths):
    """
    Group files by their class based on filename parsing.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        dict: Dictionary mapping class names to lists of file paths
    """
    class_files = defaultdict(list)
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        parsed = parse_filename_class(filename)
        
        if parsed:
            _, class_name, _, _ = parsed
            class_files[class_name].append(file_path)
        else:
            # Skip files that don't match the expected pattern
            print(f"Warning: Skipping file with unexpected naming pattern: {filename}")
    
    return dict(class_files)


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
    Each image filename follows the pattern: type--class--id[--aug###]
    Class information is extracted from the filename instead of directory structure
    Siamese network requires pairs of images with a label
    So we will create pairs of images from the same class with a label of 1
    And pairs of images from different classes with a label of 0
    :parameter images_dirs_paths_list: List of directories containing images
    :parameter training_portion: float, portion of data to use for training
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
        self.class_files = {}
        self.class_names = []

        # Collect all files from all directories
        all_files = []
        for directory in images_dirs_paths_list:
            if os.path.isdir(directory):
                files = get_files_from_dir(directory)
                all_files.extend(files)
            else:
                print(f"Warning: Directory does not exist: {directory}")

        # Group files by class based on filename parsing
        self.class_files = group_files_by_class(all_files)
        self.class_names = list(self.class_files.keys())
        
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        for class_name, files in self.class_files.items():
            print(f"Class '{class_name}': {len(files)} files")

        # Create positive pairs (same class)
        for class_name, files in self.class_files.items():
            if len(files) > 1:  # Need at least 2 files to create pairs
                anchors = self.__create_anchor_pairs(files, class_name)
                # Apply preprocessing to convert file paths to image tensors
                anchors = anchors.map(preprocess_twin_input_function, num_parallel_calls=tf.data.AUTOTUNE)
                self.__add_to_self_labelled_data(anchors)

        # Create negative pairs (different classes)
        class_names_list = list(self.class_files.keys())
        for i, class_a in enumerate(class_names_list):
            for class_b in class_names_list[i+1:]:
                files_a = self.class_files[class_a]
                files_b = self.class_files[class_b]
                negatives = self.__create_negative_pairs(files_a, files_b, class_a, class_b, False)
                if self.mode == "permutation":
                    negatives = negatives.concatenate(
                        self.__create_negative_pairs(files_b, files_a, class_b, class_a, False)
                    )
                # Apply preprocessing to convert file paths to image tensors
                negatives = negatives.map(preprocess_twin_input_function, num_parallel_calls=tf.data.AUTOTUNE)
                self.__add_to_self_labelled_data(negatives)

        self.__build_train_test_data()

    def __create_anchor_pairs(self, anchor_images_list, class_name):
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
        print(f"anchor pairs count: {dataset.cardinality().numpy()} for class '{class_name}'")
        return dataset

    def __create_negative_pairs(self, list_a, list_b, class_a, class_b, should_shuffle=True):
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
        print(f"negative pairs count: {dataset.cardinality().numpy()}, min size: {min_size} for classes '{class_a}' vs '{class_b}'")
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
