"""
Data splitter for Siamese network training.
"""

import os
import random
import re
import tensorflow as tf
from itertools import combinations, permutations, product
from collections import defaultdict

from app.config.loader import load_config


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
    
    Expected pattern: (?<type>\w)--(?<class>\w+)--(?<id>\w+(\.\d+)?)(?<aug_suffix>--aug(?<aug_id>\d{3}))?
    
    Args:
        filename: The filename to parse
        
    Returns:
        tuple: (type, class_name, id, aug_id) or None if parsing fails
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # New pattern with named groups and optional decimal in id
    # (?P<type>\w+)--(?P<class>\w+)--(?P<id>\w+(?:\.\d+)?)(?P<aug_suffix>--aug(?P<aug_id>\d{3}))?
    pattern = r'^(?P<type>\w+)--(?P<class>\w+)--(?P<id>\w+(?:\.\d+)?)(?P<aug_suffix>--aug(?P<aug_id>\d{3}))?$'
    match = re.match(pattern, name_without_ext)

    if match:
        groups = match.groupdict()
        # Maintain backward-compatible return order
        return groups.get('type'), groups.get('class'), groups.get('id'), groups.get('aug_id')
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
            _, class_name, id, _ = parsed
            if class_name not in class_files:
                class_files[class_name] = defaultdict(list)
            if id not in class_files[class_name]:
                class_files[class_name][id] = []
            class_files[class_name][id].append(file_path)
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


def preprocess_twin_input_function(input_img_path, validation_img_path, label, class_info):
    """Preprocess twin input function for siamese pairs."""
    return (
        preprocess_siamese_input(input_img_path),
        preprocess_siamese_input(validation_img_path),
        label,
        class_info,
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
        self, images_dirs_paths_list, training_portion=0.7, mode="combination", skip_preprocessing=False, preprocess_fn=preprocess_twin_input_function
    ):
        self.images_directories_collection = images_dirs_paths_list
        self.training_portion = training_portion
        self.mode = mode
        self.skip_preprocessing = skip_preprocessing
        self.preprocess_fn = preprocess_fn
        print(f"Mode: {mode}")
        self.train_data = None
        self.test_data = None
        self.class_files = {}
        self.class_names = []
        
        # Split individual images into train/test sets per class
        self.train_class_files = {}
        self.test_class_files = {}
        
        # Track class distribution for weighting
        self.train_class_distribution = {}
        self.test_class_distribution = {}

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

        # Split individual images into train/test sets per class
        self.__split_individual_images()
        
        # Create training pairs (only from training images)
        train_anchors, train_negatives = self.__create_training_pairs()

        # Create testing pairs (only from testing images)
        test_anchors, test_negatives = self.__create_testing_pairs()

        # Apply preprocessing at constructor if not skipped
        if not self.skip_preprocessing:
            if train_anchors is not None:
                train_anchors = train_anchors.map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
            if train_negatives is not None:
                train_negatives = train_negatives.map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
            if test_anchors is not None:
                test_anchors = test_anchors.map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
            if test_negatives is not None:
                test_negatives = test_negatives.map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Add to datasets
        if train_anchors is not None:
            self.__add_to_training_data(train_anchors)
        if train_negatives is not None:
            self.__add_to_training_data(train_negatives)
        if test_anchors is not None:
            self.__add_to_testing_data(test_anchors)
        if test_negatives is not None:
            self.__add_to_testing_data(test_negatives)
        
        # Shuffle the final datasets
        self.__shuffle_final_datasets()

    def __create_anchor_pairs(self, anchor_images_list, class_name):
        anchor_class_size = len(anchor_images_list)
        if anchor_class_size < 2:
            # Return empty dataset with correct structure (path, path, label, class_info)
            empty = tf.data.Dataset.from_tensor_slices(
                (
                    tf.constant([], dtype=tf.string),
                    tf.constant([], dtype=tf.string),
                    tf.constant([], dtype=tf.float32),
                    tf.constant([], dtype=tf.string),
                )
            )
            print(f"{class_name:<10}| {'Anchor size: ':<10}{anchor_class_size:<5}| {'Final count: ':<10}{0:<10}")
            return empty
        # Local config for size limit
        cfg = load_config()
        max_limit = cfg.siamese_network.training.get("max_samples_per_class")

        min_samples_count = min(anchor_class_size, max_limit) if max_limit is not None else anchor_class_size
        data = tf.data.Dataset.from_tensor_slices(anchor_images_list)
        # Use a concrete buffer size to avoid UNKNOWN_CARDINALITY issues
        buffer_size = max(1, anchor_class_size)
        data = data.shuffle(buffer_size, seed=random.randint(20, 80)).take(min_samples_count)
        
        pairs = (
            list(combinations(list(data), 2))
            if self.mode == "combination"
            else list(permutations(list(data), 2))
        )
    
        # combinations([1, 3, 4, 5], 2) -> [(1, 3), (1, 4), (1, 5), (3, 4), (3, 5), (4, 5)]
        pairs_lefties = tf.data.Dataset.from_tensor_slices([a for a, b in pairs]) # [1, 1, 1, 3, 3, 4]
        pairs_righties = tf.data.Dataset.from_tensor_slices([b for a, b in pairs]) # [3, 4, 5, 4, 5, 5]
        labels = tf.data.Dataset.from_tensor_slices(tf.ones(len(pairs))) # [1, 1, 1, 1, 1, 1]
        # Add class information for each pair
        class_info = tf.data.Dataset.from_tensor_slices([class_name] * len(pairs))
        dataset = tf.data.Dataset.zip((pairs_lefties, pairs_righties, labels, class_info)) # [(1, 3, 1, 'A'), ...]
        
        print(f"{class_name:<10}| {'Anchor size: ':<10}{anchor_class_size:<5}| {'Final count: ':<10}{dataset.cardinality().numpy():<10}")
        return dataset

    def __create_negative_pairs(self, list_a, list_b, class_a, class_b, should_shuffle=True):
        list_a_size = len(list_a)
        data_set_a = tf.data.Dataset.from_tensor_slices(list_a)

        list_b_size = len(list_b)
        data_set_b = tf.data.Dataset.from_tensor_slices(list_b)

        min_size_class = min(list_a_size, list_b_size)
        min_samples_per_class = min_size_class

        # Local config for size limit
        cfg = load_config()
        max_samples_per_class = cfg.siamese_network.training.get("max_samples_per_class")
        if max_samples_per_class is not None:
            min_samples_per_class = min(min_size_class, max_samples_per_class)

        shuffle_seed = random.randint(20, 80) if should_shuffle else None

        # If no samples are available, return an empty dataset with correct structure
        if min_samples_per_class == 0:
            empty = tf.data.Dataset.from_tensor_slices(
                (
                    tf.constant([], dtype=tf.string),
                    tf.constant([], dtype=tf.string),
                    tf.constant([], dtype=tf.float32),
                    tf.constant([], dtype=tf.string),
                )
            )
            print(f"('{class_a}', '{class_b}'): 0 negative pairs count. Samples per class: 0")
            return empty
      
        # Use concrete buffer sizes based on list lengths to avoid UNKNOWN_CARDINALITY
        buffer_a = max(1, list_a_size)
        buffer_b = max(1, list_b_size)
        data_set_a = data_set_a.shuffle(buffer_a, seed=shuffle_seed).take(min_samples_per_class)
        data_set_b = data_set_b.shuffle(buffer_b, seed=shuffle_seed).take(min_samples_per_class)
        dataset = self.__create_product_dataset(data_set_a, data_set_b, class_a)
        
        if self.mode == "permutation":
            dataset = self.__create_product_dataset(data_set_b, data_set_a, class_b)

        final_data_set_size = dataset.cardinality().numpy()
        print(f"('{class_a}', '{class_b}'): {final_data_set_size} negative pairs count. Samples per class: {min_samples_per_class}")
        return dataset

    def __create_product_dataset(self, tensor1, tensor2, class_name):
        list1 = list(tensor1)
        list2 = list(tensor2)
        lists_product = list(product(list1, list2))
        if len(lists_product) == 0:
            pairs_lefties = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.string))
            pairs_righties = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.string))
            labels = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.float32))
            class_info = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.string))
        else:
            pairs_lefties = tf.data.Dataset.from_tensor_slices([a for a, b in lists_product]) 
            pairs_righties = tf.data.Dataset.from_tensor_slices([b for a, b in lists_product])
            labels = tf.data.Dataset.from_tensor_slices(tf.zeros(len(list(lists_product)), dtype=tf.float32))
            # Add class information for negative pairs (use one of the classes)
            class_info = tf.data.Dataset.from_tensor_slices([class_name] * len(lists_product))
        return tf.data.Dataset.zip((pairs_lefties, pairs_righties, labels, class_info))


    def __split_individual_images(self):
        """Split individual images into train/test sets per class to prevent data leakage.
        We are considering all augemented images of specific image as a single image, not individual images."""
        print("\n🔍 Splitting individual images into train/test sets...")
        
        for class_name, ids in self.class_files.items():
            # Shuffle files for random split
            shuffled_ids = list(ids)
            random.shuffle(shuffled_ids)
            
            # Calculate split sizes
            total_samples = len(shuffled_ids)
            train_size = round(total_samples * self.training_portion)
            
            # Split files
            train_ids = shuffled_ids[:train_size]
            test_ids = shuffled_ids[train_size:]

            train_files = []
            test_files = []
            for id in train_ids:
                train_files.extend(ids[id])
            for id in test_ids:
                test_files.extend(ids[id])

            self.train_class_files[class_name] = train_files
            self.test_class_files[class_name] = test_files
            
            print(f"  Class '{class_name}': {len(train_files)} train, {len(test_files)} test")

    def __create_training_pairs(self):
        """Create training pairs only from training images. Returns (anchors_ds, negatives_ds) of file paths."""
        print("\n🔍 Creating training pairs...")
        anchors_ds = None
        negatives_ds = None

        # Create positive pairs (same class) from training images only
        for class_name, train_files in self.train_class_files.items():
            if len(train_files) > 1:  # Need at least 2 files to create pairs
                anchors = self.__create_anchor_pairs(train_files, class_name)
                anchors_ds = anchors if anchors_ds is None else anchors_ds.concatenate(anchors)

        # Create negative pairs (different classes) from training images only
        class_names_list = list(self.train_class_files.keys())
        for i, class_a in enumerate(class_names_list):
            for class_b in class_names_list[i+1:]:
                files_a = self.train_class_files[class_a]
                files_b = self.train_class_files[class_b]
                negatives = self.__create_negative_pairs(files_a, files_b, class_a, class_b, False)
                if self.mode == "permutation":
                    negatives = negatives.concatenate(
                        self.__create_negative_pairs(files_b, files_a, class_b, class_a, False)
                    )
                negatives_ds = negatives if negatives_ds is None else negatives_ds.concatenate(negatives)

        return anchors_ds, negatives_ds

    def __create_testing_pairs(self):
        """Create testing pairs only from testing images. Returns (anchors_ds, negatives_ds) of file paths."""
        print("\n🔍 Creating testing pairs...")
        anchors_ds = None
        negatives_ds = None

        # Create positive pairs (same class) from testing images only
        for class_name, test_files in self.test_class_files.items():
            if len(test_files) > 1:  # Need at least 2 files to create pairs
                anchors = self.__create_anchor_pairs(test_files, class_name)
                anchors_ds = anchors if anchors_ds is None else anchors_ds.concatenate(anchors)

        # Create negative pairs (different classes) from testing images only
        class_names_list = list(self.test_class_files.keys())
        for i, class_a in enumerate(class_names_list):
            for class_b in class_names_list[i+1:]:
                files_a = self.test_class_files[class_a]
                files_b = self.test_class_files[class_b]
                negatives = self.__create_negative_pairs(files_a, files_b, class_a, class_b, False)
                if self.mode == "permutation":
                    negatives = negatives.concatenate(
                        self.__create_negative_pairs(files_b, files_a, class_b, class_a, False)
                    )
                negatives_ds = negatives if negatives_ds is None else negatives_ds.concatenate(negatives)

        return anchors_ds, negatives_ds

    def __add_to_training_data(self, data):
        """Add data to training dataset."""
        if not self.train_data:
            self.train_data = data
        else:
            self.train_data = self.train_data.concatenate(data)

    def __add_to_testing_data(self, data):
        """Add data to testing dataset."""
        if not self.test_data:
            self.test_data = data
        else:
            self.test_data = self.test_data.concatenate(data)

    def __shuffle_final_datasets(self):
        """Shuffle the final training and testing datasets."""
        print("\n🔍 Shuffling final datasets...")
        
        if self.train_data is not None:
            train_size = self.train_data.cardinality().numpy()
            if train_size > 0:
                self.train_data = self.train_data.shuffle(
                    buffer_size=train_size, seed=random.randint(20, 80)
                )
            print(f"  Training dataset: {train_size} pairs")
        
        if self.test_data is not None:
            test_size = self.test_data.cardinality().numpy()
            if test_size > 0:
                self.test_data = self.test_data.shuffle(
                    buffer_size=test_size, seed=random.randint(20, 80)
                )
            print(f"  Testing dataset: {test_size} pairs")
    
    def get_class_distribution(self, dataset='train'):
        """
        Get class distribution statistics for the specified dataset.
        
        Args:
            dataset: 'train' or 'test'
            
        Returns:
            Dictionary mapping class names to number of pairs
        """
        if dataset == 'train':
            class_files = self.train_class_files
        elif dataset == 'test':
            class_files = self.test_class_files
        else:
            raise ValueError("dataset must be 'train' or 'test'")
        
        # Count number of files per class (these will be used to generate pairs)
        return {class_name: len(files) for class_name, files in class_files.items()}
