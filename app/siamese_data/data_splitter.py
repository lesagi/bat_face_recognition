"""
Data splitter for Siamese network training.
"""

import os
import random
import tensorflow as tf
from itertools import combinations, permutations, product

from config.loader import load_config
from utils.filename_parser import group_files_by_class
from siamese_data.pair_class_info import PairClassInfo
from siamese_core.network import SIAMESE_INPUT_EDGE_LENGTH

_preprocess_error_count = tf.Variable(0, dtype=tf.int32, trainable=False)


def get_files_from_dir(directory):
    """Get all files from a directory, excluding .DS_Store files."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and not f.startswith(".DS_Store")
    ]


def preprocess_siamese_input(file_path):
    """Preprocess a single image file path into a tensor."""
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_image(byte_img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    img = tf.image.resize(img, (SIAMESE_INPUT_EDGE_LENGTH, SIAMESE_INPUT_EDGE_LENGTH))
    img = img / 255.0
    img = tf.cast(img, tf.float32)

    return img


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
        self, images_dirs_paths_list, training_portion=0.7, mode="combination",
        skip_preprocessing=False, preprocess_fn=preprocess_twin_input_function,
        permute_labels=False, split_seed=None, split_mode="image_split",
    ):
        self.permute_labels = permute_labels
        self.split_seed = split_seed
        self.split_mode = split_mode
        self.images_directories_collection = images_dirs_paths_list
        self.training_portion = training_portion
        self.mode = mode
        self.skip_preprocessing = skip_preprocessing
        self.preprocess_fn = preprocess_fn
        print(f"Mode: {mode}")
        print(f"Split mode: {split_mode}")
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

        # Cache config once for pair generation
        cfg = load_config()
        self._max_samples_per_class = cfg.siamese_network.training.get("max_samples_per_class")

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

        # Split data into train/test sets
        if self.split_mode == "image_split":
            self.__split_individual_images()
        elif self.split_mode == "bat_split":
            self.__split_by_bat()
        else:
            raise ValueError(
                f"Unknown split_mode: '{self.split_mode}'. "
                "Must be 'image_split' or 'bat_split'."
            )
        
        # Create training pairs (only from training images)
        train_anchors, train_negatives = self.__create_training_pairs()

        # Create testing pairs (only from testing images)
        test_anchors, test_negatives = self.__create_testing_pairs()

        # Concatenate positive + negative pairs (still raw file-path tuples)
        if train_anchors is not None:
            self.__add_to_training_data(train_anchors)
        if train_negatives is not None:
            self.__add_to_training_data(train_negatives)
        if test_anchors is not None:
            self.__add_to_testing_data(test_anchors)
        if test_negatives is not None:
            self.__add_to_testing_data(test_negatives)

        # Permute labels across the combined positive+negative set.
        # Must happen BEFORE preprocessing so iteration is cheap (file-path strings).
        self.__permute_combined_labels()

        # Apply preprocessing
        if not self.skip_preprocessing:
            if self.train_data is not None:
                self.train_data = self.train_data.map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
            if self.test_data is not None:
                self.test_data = self.test_data.map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Shuffle the final datasets
        self.__shuffle_final_datasets()

    def __create_dataset_from_pairs(self, pairs_left, pairs_right, labels_list, class_info_strings):
        """
        Helper method to create TensorFlow dataset from pair components.
        
        Args:
            pairs_left: List of left images in pairs
            pairs_right: List of right images in pairs
            labels_list: List of labels (1.0 or 0.0)
            class_info_strings: List of serialized PairClassInfo strings
            
        Returns:
            TensorFlow Dataset with structure (img1, img2, label, class_info)
        """
        if len(pairs_left) == 0:
            # Return empty dataset with correct structure
            return tf.data.Dataset.from_tensor_slices((
                tf.constant([], dtype=tf.string),
                tf.constant([], dtype=tf.string),
                tf.constant([], dtype=tf.float32),
                tf.constant([], dtype=tf.string),
            ))
        
        pairs_lefties = tf.data.Dataset.from_tensor_slices(pairs_left)
        pairs_righties = tf.data.Dataset.from_tensor_slices(pairs_right)
        
        labels = tf.data.Dataset.from_tensor_slices(labels_list)
        class_info = tf.data.Dataset.from_tensor_slices(class_info_strings)
        
        return tf.data.Dataset.zip((pairs_lefties, pairs_righties, labels, class_info))

    def __create_anchor_pairs(self, anchor_images_list, class_name):
        """
        Create positive (anchor) pairs from same class.
        
        Args:
            anchor_images_list: List of image paths from same class
            class_name: Name of the class
            
        Returns:
            TensorFlow Dataset of positive pairs with single-class PairClassInfo
        """
        anchor_class_size = len(anchor_images_list)
        if anchor_class_size < 2:
            # Return empty dataset
            empty = self.__create_dataset_from_pairs([], [], [], [])
            print(f"{class_name:<10}| {'Anchor size: ':<10}{anchor_class_size:<5}| {'Final count: ':<10}{0:<10}")
            return empty
        
        max_limit = self._max_samples_per_class
        min_samples_count = min(anchor_class_size, max_limit) if max_limit is not None else anchor_class_size
        data = tf.data.Dataset.from_tensor_slices(anchor_images_list)
        buffer_size = max(1, anchor_class_size)
        data = data.shuffle(buffer_size, seed=random.randint(20, 80)).take(min_samples_count)
        
        # Create pairs using combinations or permutations
        pairs = (
            list(combinations(list(data), 2))
            if self.mode == "combination"
            else list(permutations(list(data), 2))
        )
        
        # Extract pair components
        pairs_left = [a for a, b in pairs]
        pairs_right = [b for a, b in pairs]
        labels_list = [1.0] * len(pairs)  # All positive pairs
        
        # Create single-class PairClassInfo for each pair
        # Note: We need one string per pair, even though they're all identical
        class_info_strings = [
            PairClassInfo.create_single_class(class_name).to_string() 
            for _ in range(len(pairs))
        ]
        
        dataset = self.__create_dataset_from_pairs(pairs_left, pairs_right, labels_list, class_info_strings)
        
        print(f"{class_name:<10}| {'Anchor size: ':<10}{anchor_class_size:<5}| {'Final count: ':<10}{len(pairs):<10}")
        return dataset

    def __create_negative_pair_dataset(self, list_a, list_b, class_a, class_b):
        """
        Create negative pair dataset from two different classes.
        
        Args:
            list_a: List of images from first class
            list_b: List of images from second class
            class_a: Name of first class
            class_b: Name of second class
            
        Returns:
            TensorFlow Dataset of negative pairs with dual-class PairClassInfo
        """
        lists_product = list(product(list_a, list_b))
        
        if len(lists_product) == 0:
            return self.__create_dataset_from_pairs([], [], [], [])
        
        # Extract pair components
        pairs_left = [a for a, b in lists_product]
        pairs_right = [b for a, b in lists_product]
        labels_list = [0.0] * len(lists_product)  # All negative pairs
        
        # Create dual-class PairClassInfo for each pair
        # Note: We need one string per pair with BOTH class names
        class_info_strings = [
            PairClassInfo.create_dual_class(class_a, class_b).to_string() 
            for _ in range(len(lists_product))
        ]
        
        return self.__create_dataset_from_pairs(pairs_left, pairs_right, labels_list, class_info_strings)

    def __create_negative_pairs(self, list_a, list_b, class_a, class_b, should_shuffle=True):
        """
        Create negative pairs from two different classes.
        
        Args:
            list_a: List of images from first class
            list_b: List of images from second class
            class_a: Name of first class
            class_b: Name of second class
            should_shuffle: Whether to shuffle the lists
            
        Returns:
            TensorFlow Dataset of negative pairs
        """
        list_a_size = len(list_a)
        data_set_a = tf.data.Dataset.from_tensor_slices(list_a)
        
        list_b_size = len(list_b)
        data_set_b = tf.data.Dataset.from_tensor_slices(list_b)
        
        min_size_class = min(list_a_size, list_b_size)
        min_samples_per_class = min_size_class
        
        max_samples_per_class = self._max_samples_per_class
        if max_samples_per_class is not None:
            min_samples_per_class = min(min_size_class, max_samples_per_class)
        
        # If no samples, return empty dataset
        if min_samples_per_class == 0:
            empty = self.__create_dataset_from_pairs([], [], [], [])
            print(f"('{class_a}', '{class_b}'): 0 negative pairs count. Samples per class: 0")
            return empty
        
        # Shuffle and take samples
        shuffle_seed = random.randint(20, 80) if should_shuffle else None
        buffer_a = max(1, list_a_size)
        buffer_b = max(1, list_b_size)
        data_set_a = data_set_a.shuffle(buffer_a, seed=shuffle_seed).take(min_samples_per_class)
        data_set_b = data_set_b.shuffle(buffer_b, seed=shuffle_seed).take(min_samples_per_class)
        
        # Create negative pair dataset with BOTH class names
        dataset = self.__create_negative_pair_dataset(
            list(data_set_a), 
            list(data_set_b), 
            class_a, 
            class_b
        )
        
        # For permutation mode, add reverse pairs
        if self.mode == "permutation":
            dataset = dataset.concatenate(
                self.__create_negative_pair_dataset(
                    list(data_set_b),
                    list(data_set_a),
                    class_b,
                    class_a
                )
            )
        
        final_data_set_size = dataset.cardinality().numpy()
        print(f"('{class_a}', '{class_b}'): {final_data_set_size} negative pairs count. Samples per class: {min_samples_per_class}")
        return dataset

    def __split_individual_images(self):
        """Split individual images into train/test sets per class to prevent data leakage.
        We are considering all augemented images of specific image as a single image, not individual images.
        Every class appears in both train and test."""
        print("\n🔍 Splitting individual images into train/test sets...")
        
        rng = random.Random(self.split_seed) if self.split_seed is not None else random
        
        for class_name, ids in self.class_files.items():
            # Shuffle files for random split
            shuffled_ids = list(ids)
            rng.shuffle(shuffled_ids)
            
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

    def __split_by_bat(self):
        """Split entire bat classes into train vs test (no class overlap).
        Train and test sets contain completely different bats, testing
        whether the learned similarity metric generalises to unseen individuals."""
        print("\n🔍 Splitting by bat class (disjoint classes)...")

        rng = random.Random(self.split_seed) if self.split_seed is not None else random
        class_names = list(self.class_files.keys())
        rng.shuffle(class_names)

        train_count = max(1, round(len(class_names) * self.training_portion))
        train_classes = class_names[:train_count]
        test_classes = class_names[train_count:]

        for class_name in train_classes:
            all_files = []
            for id_files in self.class_files[class_name].values():
                all_files.extend(id_files)
            self.train_class_files[class_name] = all_files

        for class_name in test_classes:
            all_files = []
            for id_files in self.class_files[class_name].values():
                all_files.extend(id_files)
            self.test_class_files[class_name] = all_files

        print(f"  Train classes ({len(train_classes)}): {train_classes}")
        for cn in train_classes:
            print(f"    '{cn}': {len(self.train_class_files[cn])} files")
        print(f"  Test classes ({len(test_classes)}): {test_classes}")
        for cn in test_classes:
            print(f"    '{cn}': {len(self.test_class_files[cn])} files")

    def __create_training_pairs(self):
        """Create training pairs only from training images. Returns (anchors_ds, negatives_ds) of file paths."""
        print("\n🔍 Creating training pairs...")

        # Initialize to None in case loops don't execute
        anchors = None
        negatives = None

        # Create positive pairs (same class) from training images only
        for class_name, train_files in self.train_class_files.items():
            if len(train_files) > 1:  # Need at least 2 files to create pairs
                class_anchors = self.__create_anchor_pairs(train_files, class_name)
                if anchors is None:
                    anchors = class_anchors
                else:
                    anchors = anchors.concatenate(class_anchors)

        # Create negative pairs (different classes) from training images only
        class_names_list = list(self.train_class_files.keys())
        for i, class_a in enumerate(class_names_list):
            for class_b in class_names_list[i+1:]:
                files_a = self.train_class_files[class_a]
                files_b = self.train_class_files[class_b]
                class_negatives = self.__create_negative_pairs(files_a, files_b, class_a, class_b, False)
                if negatives is None:
                    negatives = class_negatives
                else:
                    negatives = negatives.concatenate(class_negatives)

        return anchors, negatives

    def __create_testing_pairs(self):
        """Create testing pairs only from testing images. Returns (anchors_ds, negatives_ds) of file paths."""
        print("\n🔍 Creating testing pairs...")

        # Initialize to None in case loops don't execute
        anchors = None
        negatives = None

        # Create positive pairs (same class) from testing images only
        for class_name, test_files in self.test_class_files.items():
            if len(test_files) > 1:  # Need at least 2 files to create pairs
                class_anchors = self.__create_anchor_pairs(test_files, class_name)
                if anchors is None:
                    anchors = class_anchors
                else:
                    anchors = anchors.concatenate(class_anchors)

        # Create negative pairs (different classes) from testing images only
        class_names_list = list(self.test_class_files.keys())
        for i, class_a in enumerate(class_names_list):
            for class_b in class_names_list[i+1:]:
                files_a = self.test_class_files[class_a]
                files_b = self.test_class_files[class_b]
                class_negatives = self.__create_negative_pairs(files_a, files_b, class_a, class_b, False)
                if negatives is None:
                    negatives = class_negatives
                else:
                    negatives = negatives.concatenate(class_negatives)

        return anchors, negatives

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

    def __permute_combined_labels(self):
        if not self.permute_labels:
            return

        for attr in ('train_data', 'test_data'):
            dataset = getattr(self, attr)
            if dataset is None:
                continue

            size = dataset.cardinality().numpy()
            if size <= 0:
                continue

            labels = [float(item[2].numpy()) for item in dataset]
            pos_count = sum(1 for l in labels if l == 1.0)
            random.shuffle(labels)

            label_ds = tf.data.Dataset.from_tensor_slices(labels)
            other_ds = dataset.map(lambda a, b, _l, c: (a, b, c))
            new_ds = tf.data.Dataset.zip((other_ds, label_ds)).map(
                lambda data, new_label: (data[0], data[1], new_label, data[2])
            )
            setattr(self, attr, new_ds)

            print(f"  Permuted {attr} labels: {size} pairs ({pos_count} positive, {size - pos_count} negative)")

    def __shuffle_final_datasets(self):
        """Shuffle the final training and testing datasets."""
        print("\n🔍 Shuffling final datasets...")
        
        if self.train_data is not None:
            self.train_size = self.train_data.cardinality().numpy()
            if self.train_size > 0:
                self.train_data = self.train_data.shuffle(
                    buffer_size=self.train_size, seed=random.randint(20, 80)
                )
            print(f"  Training dataset: {self.train_size} pairs")
        else:
            self.train_size = 0
        
        if self.test_data is not None:
            self.test_size = self.test_data.cardinality().numpy()
            if self.test_size > 0:
                self.test_data = self.test_data.shuffle(
                    buffer_size=self.test_size, seed=random.randint(20, 80)
                )
            print(f"  Testing dataset: {self.test_size} pairs")
        else:
            self.test_size = 0
    
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
