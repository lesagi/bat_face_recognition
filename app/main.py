"""
Main application module.
"""

import random
import requests
from io import BytesIO
from PIL import Image
from itertools import combinations, permutations
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import albumentations as alb
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import csv

from ultralytics import YOLO

# Import configurations
from config.general import GPU_ENABLED
# Note: siamese_network.config and paths have been removed
# All configuration now managed through config.yml via config.loader.load_config()

# from background_replacement.paths import SEGMENTATION_PATHS  # TODO: Fix configuration system

from utils.main import *


# Configure GPU
def configure_gpu():
    """Configure GPU settings based on availability and configuration."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if GPU_ENABLED and gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("GPU is disabled or not available")


# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist."""
    for path_dict in [
        SIAMESE_PATHS,
        SEGMENTATION_PATHS,
        VIDEO_PATHS,
        AUGMENTATION_PATHS,
    ]:
        for path in path_dict.values():
            os.makedirs(path, exist_ok=True)


gpus = tf.config.experimental.list_physical_devices("GPU")
if GPU_ENABLED and gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def preprocess_siamese_input(file_path):
    try:
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image
        img = tf.io.decode_png(byte_img)

        # Preprocessing steps - resizing the image
        img = tf.image.resize(
            img, (SIAMESE_INPUT_EDGE_LENGTH, SIAMESE_INPUT_EDGE_LENGTH)
        )
        # Scale image to be between 0 and 1
        img = img / 255.0

        # Return image
        return img
    except Exception as e:
        print(e)
        print(file_path)


def preprocess_twin_input_function(input_img, validation_img, label):
    return (
        preprocess_siamese_input(input_img),
        preprocess_siamese_input(validation_img),
        label,
    )


def create_blur_image(height, width):
    # Create a random image (each pixel has random RGB values)
    random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Choose a random odd kernel size between 3 and 29 (Gaussian blur requires odd dimensions)
    kernel_size = random.choice([k for k in range(3, 30) if k % 2 == 1])

    # Apply Gaussian Blur with the random kernel size
    blurry_image = cv2.GaussianBlur(random_image, (kernel_size, kernel_size), 0)
    return blurry_image


def create_green_image(height, width):
    # Create a new image with the specified dimensions and 3 color channels
    green_image = np.zeros((height, width, 3), dtype=np.uint8)
    green_image[:] = (
        0,
        255,
        0,
    )  # In OpenCV, this represents green (BGR order: Blue=0, Green=255, Red=0)
    return green_image


def process_frame_of_masks_prediction(frame, mask):
    frame_height, frame_width = frame.shape[:2]
    background_image = create_blur_image(height=frame_height, width=frame_width)

    # Choose a random odd kernel size between 3 and 29 (Gaussian blur requires odd dimensions)
    kernel_size = random.choice([k for k in range(3, 30) if k % 2 == 1])

    # Apply Gaussian Blur with the random kernel size
    blurry_image = cv2.GaussianBlur(background_image, (kernel_size, kernel_size), 0)

    # Convert the mask to a numpy array
    mask = mask.cpu().numpy()
    mask = cv2.resize(mask, (frame_width, frame_height))

    # Convert mask to boolean array
    mask = mask.astype(bool)

    # Repeat the mask along the color dimension
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Use the mask to extract ROI from frame
    roi = frame * mask

    # Use the mask to replace the corresponding region in the green image with the ROI
    blurry_image[mask] = roi[mask]

    return blurry_image


def get_random_cropped_image(height, width):
    """
    Fetches a random image from Picsum Photos and crops it to the given height and width.

    Parameters:
        height (int): The desired height of the cropped image.
        width (int): The desired width of the cropped image.

    Returns:
        image_np (numpy.ndarray): The cropped image in BGR format (suitable for OpenCV).
    """
    # Ensure we fetch an image that's at least as large as the desired dimensions.
    # We take the max to get a square image that will definitely cover both dimensions.
    min_dim = max(height, width)

    # Fetch a random image from Picsum Photos with size min_dim x min_dim
    url = f"https://picsum.photos/{min_dim}/{min_dim}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch image from Picsum")

    # Open the image using PIL
    image = Image.open(BytesIO(response.content))

    # Calculate the coordinates to crop the image to the desired size (center crop)
    img_width, img_height = image.size
    left = (img_width - width) // 2
    top = (img_height - height) // 2
    right = left + width
    bottom = top + height

    cropped_image = image.crop((left, top, right, bottom))

    # Convert the cropped image to a NumPy array and convert from RGB to BGR (for OpenCV)
    image_np = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    return image_np


def replace_green_background(image):
    """
    Replaces the green background in an image with a random image from Picsum Photos.

    Parameters:
        image (numpy.ndarray): Image containing an object with a green background (BGR format).

    Returns:
        result (numpy.ndarray): The image with the green background replaced.
    """
    # Get dimensions of the input image
    height, width = image.shape[:2]

    # Fetch a random background image cropped to the same dimensions
    random_background = get_random_cropped_image(height, width)

    # Define lower and upper bounds for the green color (in BGR)
    # Here, we allow some tolerance around (0, 255, 0)
    lower_green = np.array([0, 230, 0], dtype=np.uint8)
    upper_green = np.array([30, 255, 30], dtype=np.uint8)

    # Create a mask for pixels within the green range
    green_mask = cv2.inRange(image, lower_green, upper_green)

    # Optionally, you can smooth the mask if needed:
    green_mask = cv2.medianBlur(green_mask, 5)

    # Invert mask to get the object area (non-green regions)
    object_mask = cv2.bitwise_not(green_mask)

    # Extract the object from the original image using the inverted mask
    object_foreground = cv2.bitwise_and(image, image, mask=object_mask)

    # Extract the background from the random image using the green mask
    background_region = cv2.bitwise_and(
        random_background, random_background, mask=green_mask
    )

    # Combine the object and the new background
    result = cv2.add(object_foreground, background_region)
    return result


# def process_frame_of_masks_prediction(frame, mask):
#     frame_height, frame_width = frame.shape[:2]
#
#     # Convert the mask to a numpy array and resize to match the frame dimensions
#     mask = mask.cpu().numpy()
#     mask = cv2.resize(mask, (frame_width, frame_height))
#
#     # Convert mask to boolean array (assuming nonzero values indicate the object)
#     mask = mask.astype(bool)
#
#     # --- New code: extract only the main object in the center ---
#     # Convert boolean mask to uint8 image (0 or 255) for contour detection
#     mask_uint8 = (mask.astype(np.uint8)) * 255
#
#     # Find external contours in the mask
#     contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Determine the frame's center
#     center_x, center_y = frame_width // 2, frame_height // 2
#
#     main_contour = None
#     best_distance = float('inf')
#
#     # Iterate over found contours to find the one closest to the center
#     for cnt in contours:
#         M = cv2.moments(cnt)
#         if M["m00"] == 0:
#             continue
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         # Calculate Euclidean distance from the frame center
#         distance = np.sqrt((cX - center_x) ** 2 + (cY - center_y) ** 2)
#         if distance < best_distance:
#             best_distance = distance
#             main_contour = cnt
#
#     # Create a new mask that contains only the main object
#     main_object_mask = np.zeros_like(mask_uint8)
#     if main_contour is not None:
#         cv2.drawContours(main_object_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
#
#     # Convert the main object mask back to a boolean array and repeat across channels
#     main_object_mask = main_object_mask.astype(bool)
#     main_object_mask = np.repeat(main_object_mask[:, :, np.newaxis], 3, axis=2)
#     # --- End of new code ---
#
#     # Use the refined mask to extract the region of interest (ROI) from the frame
#     roi = frame * main_object_mask
#
#     # Use the mask to replace the corresponding region in the blurry background with the ROI
#     background_image = get_random_cropped_image(frame_height, frame_width)
#     background_image[main_object_mask] = roi[main_object_mask]
#
#     return background_image


class VideoImagesExtractor:
    def __init__(self, video_path, output_base_dir):
        self.video_path = video_path
        self.output_base_dir = output_base_dir
        self.image_processor = BatFaceSegmentationBackgroundProcessor()

    def process_video(
        self,
        preprocess_frame=None,
        post_prediction_processor=None,
        output_sub_dir_name=None,
        threshold=0.5,
    ):
        video_name = os.path.basename(self.video_path)
        output_folder = os.path.join(
            self.output_base_dir, os.path.splitext(video_name)[0]
        )
        if output_sub_dir_name:
            output_folder = os.path.join(output_folder, output_sub_dir_name)

        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            serial_filename = f"{strip_filename_from_path(video_name)}.{frame_number}"

            frame = self.image_processor.process(
                frame,
                preprocess_frame,
                process_frame_of_masks_prediction,
                post_prediction_processor,
                threshold,
            )
            cv2.imwrite(os.path.join(output_folder, f"{serial_filename}.jpg"), frame)

        cap.release()
        # cv2.destroyAllWindows()


class BatFaceSegmentationBackgroundProcessor:
    def __init__(self):
        # self.model = YOLO(os.path.join(SEGMENTATION_PATHS["MODEL"], "best.pt"))  # TODO: Fix configuration system

    def process(
        self,
        image,
        preprocess_frame=None,
        prediction_processor=None,
        post_prediction_processor=None,
        threshold=0.5,
    ):
        if preprocess_frame:
            image = preprocess_frame(image)

        predictions = self.model.predict(image, conf=threshold)[0]
        if not predictions:
            return None

        box_prediction = predictions.boxes.data.tolist()[0]
        mask_prediction = predictions.masks.data[0]

        if prediction_processor:
            image = prediction_processor(image, mask_prediction)

        if post_prediction_processor:
            image = post_prediction_processor(box_prediction, image)

        return image


class ImagesAugmentor:
    def __init__(self, source_dir, augmentation_function, output_dir=None):
        """Initialize the image augmentor."""
        from data_augmentation.config import AUGMENTATION_PATHS

        self.source_dir = source_dir
        self.augmentation_function = augmentation_function
        self.output_dir = output_dir or AUGMENTATION_PATHS["OUTPUT"]
        os.makedirs(self.output_dir, exist_ok=True)

    def augment_images(self, output_sub_dir=None, samples_count=60):
        output_dir = self.output_dir
        if output_sub_dir:
            output_dir = os.path.join(self.output_dir, output_sub_dir)

        os.makedirs(output_dir, exist_ok=True)

        for image in os.listdir(self.source_dir):
            image_name = os.path.splitext(image)[0]
            img_path = os.path.join(self.source_dir, image)
            if is_DS_Store(image_name):
                continue

            for x in range(samples_count):
                aug_img = self.augmentation_function(img_path)
                aug_img_path = os.path.join(output_dir, f"{image_name}.{x}.png")
                cv2.imwrite(aug_img_path, aug_img)


def get_files_from_dir(dir):
    return [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and not is_DS_Store(f)
    ]


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
        self,
        images_dirs_paths_list,
        preprocess_twin_input_function=None,
        training_portion=0.7,
        mode="combination",
    ):
        self.images_directories_collection = images_dirs_paths_list
        self.preprocess_twin_input_function = preprocess_twin_input_function
        self.training_portion = training_portion
        self.labelled_data = None
        self.mode = mode
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
            if self.preprocess_twin_input_function is not None:
                anchors = anchors.map(self.preprocess_twin_input_function)
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
            if self.preprocess_twin_input_function is not None:
                negatives = negatives.map(self.preprocess_twin_input_function)
            self.__add_to_self_labelled_data(negatives)

        # self.add_pairs_to_per_dir_pairs(self.labelled_data)
        self.__build_train_test_data()

    def add_pairs_to_per_dir_pairs(self, pairs):
        for pair in pairs:
            left_in_pair_dir_name, left_in_pair_frame, _ = os.path.basename(
                pair[0].numpy().decode("utf-8")
            ).split(".")
            if left_in_pair_dir_name not in self.per_dir_pairs:
                self.per_dir_pairs[left_in_pair_dir_name] = {}
            left_in_pair_dict = self.per_dir_pairs[left_in_pair_dir_name]

            if left_in_pair_frame not in left_in_pair_dict:
                left_in_pair_dict[left_in_pair_frame] = {}
            left_in_pair_frame_dict = left_in_pair_dict[left_in_pair_frame]

            right_in_pair_dir_name = os.path.basename(
                pair[1].numpy().decode("utf-8")
            ).split(".")[0]
            if right_in_pair_dir_name not in left_in_pair_frame_dict:
                left_in_pair_frame_dict[right_in_pair_dir_name] = []
            left_in_pair_frame_dict[right_in_pair_dir_name].append(pair)

            # filter out entries with less than 5 samples of a pair

    def __create_anchor_pairs(self, anchor_images_list):
        pairs = (
            list(combinations(anchor_images_list, 2))
            if self.mode == "combination"
            else list(permutations(anchor_images_list, 2))
        )
        pairs_a = tf.data.Dataset.from_tensor_slices([a for a, b in pairs])
        pairs_b = tf.data.Dataset.from_tensor_slices([b for a, b in pairs])
        samples_count = len(pairs)
        dataset = tf.data.Dataset.zip(
            (
                pairs_a,
                pairs_b,
                tf.data.Dataset.from_tensor_slices(tf.ones(samples_count)),
            )
        )
        if TRAINING_DATA_MAX_SIZE_LIMIT is not None:
            dataset = dataset.take(TRAINING_DATA_MAX_SIZE_LIMIT)
        print(f"anchors pairs count: {dataset.cardinality().numpy()}")
        return dataset

    def __create_negative_pairs(self, list_a, list_b, should_shuffle=True):
        list_a_size = len(list_a)
        data_set_a = tf.data.Dataset.from_tensor_slices(list_a)

        list_b_size = len(list_b)
        data_set_b = tf.data.Dataset.from_tensor_slices(list_b)

        if should_shuffle:
            data_set_a = data_set_a.shuffle(
                data_set_a.cardinality(), seed=random.randint(20, 80)
            )
            data_set_b = data_set_b.shuffle(
                data_set_b.cardinality(), seed=random.randint(20, 80)
            )

        min_size = min(list_a_size, list_b_size)
        if TRAINING_DATA_MAX_SIZE_LIMIT is not None:
            min_size = min(min_size, TRAINING_DATA_MAX_SIZE_LIMIT)

        data_set_a = data_set_a.take(min_size)
        data_set_b = data_set_b.take(min_size)

        dataset = tf.data.Dataset.zip(
            (
                data_set_a,
                data_set_b,
                tf.data.Dataset.from_tensor_slices(tf.zeros(min_size)),
            )
        )
        print(
            f"negative pairs count: {dataset.cardinality().numpy()}, min size: {min_size}"
        )
        return dataset

    def __build_train_test_data(self):
        labelled_data_size = len(self.labelled_data)
        self.labelled_data = self.labelled_data.shuffle(
            buffer_size=labelled_data_size, seed=random.randint(20, 80)
        )
        # Build dataloader pipeline
        data_size = len(self.labelled_data)
        train_size = round(data_size * self.training_portion)

        # Split the data into training and testing sets
        self.train_data = self.labelled_data.take(train_size)
        self.test_data = self.labelled_data.skip(train_size)

    def __add_to_self_labelled_data(self, data):
        if not self.labelled_data:
            self.labelled_data = data
        else:
            self.labelled_data = self.labelled_data.concatenate(data)


class SiameseNetwork:
    def __init__(self, embedding_distance_calculator_layer):
        embedding_distance_calculator_layer._name = "distance"
        self.model = self.build_model(embedding_distance_calculator_layer)

    def build_model(self, embedding_distance_calculator_layer):
        embedding_model = self.__make_embedding()
        input_img = Input(
            name="input_img",
            shape=(SIAMESE_INPUT_EDGE_LENGTH, SIAMESE_INPUT_EDGE_LENGTH, 3),
        )
        validation_img = Input(
            name="validation_img",
            shape=(SIAMESE_INPUT_EDGE_LENGTH, SIAMESE_INPUT_EDGE_LENGTH, 3),
        )

        distances = embedding_distance_calculator_layer(
            embedding_model(input_img), embedding_model(validation_img)
        )

        classifier = Dense(1, activation="sigmoid")(distances)

        return Model(
            inputs=[input_img, validation_img],
            outputs=classifier,
            name="SiameseNetwork",
        )

    def __make_embedding(self):
        """
        Create the embedding network
        This network will take an image as input and output a vector of size 4096
        :return:
        """
        inp = Input(
            shape=(SIAMESE_INPUT_EDGE_LENGTH, SIAMESE_INPUT_EDGE_LENGTH, 3),
            name="input_image",
        )

        # First block
        c1 = Conv2D(64, (10, 10), activation="relu")(inp)
        m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

        # Second block
        c2 = Conv2D(128, (7, 7), activation="relu")(m1)
        m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

        # Third block
        c3 = Conv2D(128, (4, 4), activation="relu")(m2)
        m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)

        # Final embedding block
        c4 = Conv2D(256, (4, 4), activation="relu")(m3)
        f1 = Flatten()(c4)
        d1 = Dense(4096, activation="sigmoid")(f1)

        return Model(inputs=[inp], outputs=[d1], name="embedding")

    def compile(self, optimizer, loss, metrics):
        pass


class SiameseNetworkTrainer:
    def __init__(
        self,
        siamese_model,
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss_function=tf.losses.BinaryCrossentropy(),
        log_file_path="training_log.csv",
    ):
        self.siamese_model = siamese_model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.checkpoint_dir = SIAMESE_PATHS["TRAINING"]["CHECKPOINTS"]
        self.checkpoint = tf.train.Checkpoint(
            opt=self.optimizer, siamese_model=self.siamese_model
        )
        self.log_file_path = os.path.join(
            SIAMESE_PATHS["TRAINING"]["MODEL"], log_file_path
        )
        os.makedirs(SIAMESE_PATHS["TRAINING"]["MODEL"], exist_ok=True)
        self.init_log_file()  # Initialize the log file

    def init_log_file(self):
        with open(self.log_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Source", "Epoch", "Loss", "Recall", "Precision"]
            )  # Write the headers

    def write_log(self, source, epoch, loss, recall, precision):
        with open(self.log_file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([source, epoch, loss, recall, precision])  # Write the data

    @tf.function
    def train_step(self, batch):

        # Record all of our operations
        with tf.GradientTape() as tape:
            # Get anchor and positive/negative image
            x = batch[:2]
            # Get label
            y = batch[2]

            # Forward pass
            yhat = self.siamese_model(x, training=True)
            # Calculate loss
            loss = self.loss_function(y, yhat)
        print(loss)

        # Calculate gradients
        grad = tape.gradient(loss, self.siamese_model.trainable_variables)

        # Calculate updated weights and apply to siamese model
        self.optimizer.apply_gradients(
            zip(grad, self.siamese_model.trainable_variables)
        )

        # Return loss
        return loss

    def train(self, train_data_batches):
        # Loop through epochs
        min_train_loss = 1
        for epoch in range(1, SIAMESE_TRAIN_EPOCHS + 1):
            print("\n Epoch {}/{}".format(epoch, SIAMESE_TRAIN_EPOCHS))
            progbar = tf.keras.utils.Progbar(len(train_data_batches))

            # Creating a metric object
            r = Recall()
            p = Precision()

            # Loop through each batch
            for idx, batch in enumerate(train_data_batches):
                # Run train step here
                loss = self.train_step(batch)
                yhat = self.siamese_model.predict(x=batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                progbar.update(idx + 1)
            self.write_log(
                "train", epoch, loss.numpy(), r.result().numpy(), p.result().numpy()
            )  # Write to log file
            self.save_model(version=epoch, save_format="tf")
            if self.checkpoint_dir:
                train_loss, train_recall, train_precision = self.test(
                    train_data_batches
                )
                self.write_log(
                    "test", epoch, train_loss, train_recall, train_precision
                )  # Write to log file

                if train_loss <= min_train_loss:
                    min_train_loss = train_loss
                    self.checkpoint.save(
                        file_prefix=os.path.join(self.checkpoint_dir, "min")
                    )

                # Save checkpoints
                if epoch % 10 == 0:
                    self.checkpoint.save(
                        file_prefix=os.path.join(self.checkpoint_dir, "periodic")
                    )

    def test(self, test_data_batches):
        # Creating a metric object
        r = Recall()
        p = Precision()

        # Initialize a variable to store the total loss
        total_loss = 0
        num_batches = 0

        # Loop through each batch
        for test_input, test_val, y_true in test_data_batches.as_numpy_iterator():
            yhat = self.siamese_model.predict([test_input, test_val])
            r.update_state(y_true, yhat)
            p.update_state(y_true, yhat)

            # Calculate the loss for this batch
            batch_loss = self.loss_function(y_true, yhat)
            total_loss += batch_loss
            num_batches += 1

        # Calculate the average loss over all batches
        avg_loss = total_loss / num_batches
        return avg_loss.numpy(), r.result().numpy(), p.result().numpy()

    def save_model(self, name="siamesemodelv2", version=None, save_format="tf"):
        if version:
            name = f"{name}_v{version}"
        self.siamese_model.save(
            os.path.join(SIAMESE_PATHS["TRAINING"]["MODEL"], name),
            save_format=save_format,
        )


class L1Dist(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


class SiameseModelSaliencyMapCreator:
    def __init__(self, model, input_dir_path, output_dir_path=None, nesting=None):
        self.model = model
        dir_name = os.path.basename(input_dir_path)
        if not output_dir_path:
            output_dir_path = input_dir_path
        if nesting:
            output_dir_path = os.path.join(output_dir_path, nesting)
        self.output_dir_path = output_dir_path
        self.output_file_path = os.path.join(
            self.output_dir_path, f"{dir_name}_saliency_map.pdf"
        )
        self.input_dir_path = (
            input_dir_path if not nesting else os.path.join(input_dir_path, nesting)
        )
        self.images = [
            file for file in os.listdir(self.input_dir_path) if is_img_file(file)
        ]
        self.num_images = len(self.images)
        self.fig, self.axes = plt.subplots(
            self.num_images, 2, figsize=(8.27, self.num_images * 5)
        )  # A4 size
        plt.subplots_adjust(hspace=0.5)  # adjust space between rows

    def compute_saliency_map(self):
        # choose randomly 50 images to do the saliency maps on from self.images
        saliency_maps_images = random.sample(self.images, min(50, self.num_images))
        for idx, img_name in enumerate(saliency_maps_images):
            img_path = os.path.join(self.input_dir_path, img_name)
            ax1 = self.axes[idx, 0]
            ax2 = self.axes[idx, 1]

            img = preprocess_siamese_input(img_path)
            anchor = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)

            random_counterpart = create_random_image(img)
            counterpart = tf.convert_to_tensor(
                np.expand_dims(random_counterpart, axis=0), dtype=tf.float32
            )

            with tf.GradientTape() as tape:
                tape.watch(anchor)
                output = self.model([anchor, counterpart])
            gradients = tape.gradient(output, anchor)

            saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1).numpy()

            img = tf.image.resize(img, (350, 350))
            ax1.imshow(img)
            ax1.axis("off")
            ax1.set_title(img_name)

            saliency_map_image = tf.image.resize(
                tf.expand_dims(saliency_map[0], axis=-1), (350, 350)
            )
            ax2.imshow(saliency_map_image, cmap="hot")
            ax2.axis("off")

        os.makedirs(self.output_dir_path, exist_ok=True)
        plt.savefig(self.output_file_path, format="pdf")


class SegmentationModelTrainingHelper:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    @staticmethod
    def create_yolov8_labels(input_dir, output_dir):
        for img_file in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_file)
            # load the binary mask and get its contours
            mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            mask_height, mask_width = mask.shape
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # convert the contours to polygons
            polygons = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    polygon = []
                    for point in cnt:
                        x, y = point[0]
                        polygon.append(x / mask_width)
                        polygon.append(y / mask_height)
                    polygons.append(polygon)

            # print the polygons
            with open(
                "{}.txt".format(os.path.join(output_dir, img_file)[:-4]), "w"
            ) as f:
                for polygon in polygons:
                    for p_, p in enumerate(polygon):
                        if p_ == len(polygon) - 1:
                            f.write("{}\n".format(p))
                        elif p_ == 0:
                            f.write("0 {} ".format(p))
                        else:
                            f.write("{} ".format(p))

                f.close()

    @staticmethod
    def train_new_model():
        model = YOLO(
            "yolov8n-seg.pt"
        )  # load a pretrained model (recommended for training)

        model.train(
            data=FACE_SEGMENTATION_PATHS["CONFIG"], epochs=FACE_SEGMENTATION_EPOCHS
        )


def extract_images_from_video(post_process=crop_rectangle_from_cv2_frame):
    for video_file in os.listdir(VIDEOS_DIR)[:VIDEOS_PROCESSING_LIMIT]:
        video_image_extractor = VideoImagesExtractor(
            os.path.join(VIDEOS_DIR, video_file), PROCESSED_VIDEOS_BASE_DIR
        )
        video_image_extractor.process_video(
            post_prediction_processor=post_process,
            output_sub_dir_name=PROCESSED_VIDEOS_SUB_DIR,
            threshold=0.8,
        )


def augment_filtered_images_extracted_from_videos():
    def augment_image(img_path):
        augmentor = alb.Compose(
            [
                alb.HorizontalFlip(p=0.5),
                alb.RandomBrightnessContrast(p=0.5),
                alb.RandomGamma(p=0.5),
                alb.RGBShift(p=0.5),
                alb.VerticalFlip(p=0.5),
                alb.AdvancedBlur(),
            ]
        )

        img = cv2.imread(img_path)
        return augmentor(image=img)["image"]

    # Augment training images
    images_dirs = [
        dir_name
        for dir_name in os.listdir(AUGMENTATION_PATHS["INPUT"])
        if os.path.isdir(os.path.join(AUGMENTATION_PATHS["INPUT"], dir_name))
    ]
    for images_dir in images_dirs:
        image_augmentor = ImagesAugmentor(
            os.path.join(AUGMENTATION_PATHS["INPUT"], images_dir),
            augment_image,
            AUGMENTATION_PATHS["OUTPUT"],
        )
        image_augmentor.augment_images(output_sub_dir=images_dir, samples_count=60)


def build_train_test_data_batches():
    # Split the data into training and validation sets
    data_splitter = SiameseNetworkTrainingDataSplitter(
        [SIAMESE_PATHS["TRAINING"]["DATA"]],
        preprocess_twin_input_function,
        training_portion=0.7,
        mode="permutation",
    )
    train_data_batches = data_splitter.train_data.batch(16).prefetch(8)
    test_data_batches = data_splitter.test_data.batch(16).prefetch(8)
    return train_data_batches, test_data_batches


def build_siamese_network():
    return SiameseNetwork(L1Dist())


def get_trained_siamese_network(
    siamese_network_model, train_data_batches, test_data_batches
):
    network_trainer = SiameseNetworkTrainer(siamese_network_model)
    network_trainer.train(train_data_batches)
    network_trainer.test(test_data_batches)
    network_trainer.save_model()
    return network_trainer.siamese_model


def create_saliency_maps_on_data(
    siamese_network_model, input_dir, output_dir=None, nesting=None
):
    consolidated_input_dir = (
        input_dir if not nesting else os.path.join(input_dir, nesting)
    )
    images_files = [
        file
        for file in os.listdir(consolidated_input_dir)
        if file.endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    if len(images_files) > 0:
        print(f"processing {consolidated_input_dir}")
        saliency_map_creator = SiameseModelSaliencyMapCreator(
            siamese_network_model, input_dir, output_dir, nesting
        )
        saliency_map_creator.compute_saliency_map()

    for dir_name in os.listdir(consolidated_input_dir):
        if os.path.isdir(os.path.join(consolidated_input_dir, dir_name)):
            added_nesting = dir_name if not nesting else os.path.join(nesting, dir_name)
            create_saliency_maps_on_data(
                siamese_network_model, input_dir, output_dir, nesting=added_nesting
            )


def load_model():
    if TRAINED_SIAMESE_MODEL_PATH:
        trained_model = tf.keras.models.load_model(
            TRAINED_SIAMESE_MODEL_PATH,
            custom_objects={
                "L1Dist": L1Dist,
                "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
            },
        )
        print("Loaded model from disk")
    else:
        print("Starting training")
        train_data_batches, test_data_batches = build_train_test_data_batches()
        siamese_model = build_siamese_network().model
        trained_model = get_trained_siamese_network(
            siamese_model, train_data_batches, test_data_batches
        )
        print("Trained a new model")

    return trained_model


def collect_files(directory):
    image_files = set()
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            if is_img_file(file):
                image_files.add(file)
    return image_files


def check_model_on_seen_data(model):
    training_path = os.path.join(SIAMESE_PATHS["BASE"], "data", "training")
    unseen_path = os.path.join(SIAMESE_PATHS["BASE"], "data", "unseen")
    validation_path = os.path.join(SIAMESE_PATHS["BASE"], "data", "validation")

    seen_dirs_list = os.listdir(training_path)
    seen_files = collect_files(training_path)
    unseen_files = collect_files(unseen_path)
    data_splitter = SiameseNetworkTrainingDataSplitter(
        [training_path, unseen_path, validation_path],
        training_portion=1,
        mode="permutation",
    )

    print("labelled", data_splitter.labelled_data.cardinality().numpy())
    print("train", data_splitter.train_data.cardinality().numpy())
    print("test", data_splitter.test_data.cardinality().numpy())
    train_data_batches = data_splitter.train_data.batch(16).prefetch(8)

    # Creating a metric object
    r = Recall()
    p = Precision()

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the filename with the current timestamp
    filename = f"predictions_{timestamp}.csv"
    # Loop through each batch
    csv_path = os.path.join(SIAMESE_PATHS["BASE"], "data", filename)
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        # Write the headers
        writer.writerow(
            [
                "input1_name",
                "input1_frame",
                "input1_seen",
                "input2_name",
                "input2_frame",
                "input2_seen",
                "y_true",
                "y_hat",
                "success",
            ]
        )

        # Loop through each batch
        for (
            test_input1_batch,
            test_input2_batch,
            y_true,
        ) in train_data_batches.as_numpy_iterator():
            test_input1_processed = np.array(
                [preprocess_siamese_input(img_path) for img_path in test_input1_batch]
            )
            test_input2_processed = np.array(
                [preprocess_siamese_input(img_path) for img_path in test_input2_batch]
            )
            yhat = model.predict([test_input1_processed, test_input2_processed])
            r.update_state(y_true, yhat)
            p.update_state(y_true, yhat)

            # Write each prediction to the CSV file
            for i in range(len(yhat)):
                input1_name = os.path.basename(test_input1_batch[i]).decode("utf-8")
                name1, frame1, extension1 = input1_name.split(".")
                input2_name = os.path.basename(test_input2_batch[i]).decode("utf-8")
                name2, frame2, extension2 = input2_name.split(".")
                prediction = 1 if yhat[i][0] > 0.5 else 0
                writer.writerow(
                    [
                        name1,
                        frame1,
                        name1 in seen_dirs_list,
                        name2,
                        frame2,
                        name2 in seen_dirs_list,
                        y_true[i],
                        prediction,
                        1 - abs(y_true[i] - prediction),
                    ]
                )

    print(r.result().numpy(), p.result().numpy())


def process_training_images(process_func, base_dir):
    """Process training images using the specified function."""
    for dirpath, dirnames, filenames in os.walk(base_dir):
        bat_dir = dirpath.split(os.sep)[-1]
        parent_dir = os.path.dirname(base_dir)  # Go up one level from base_dir
        bat_output_path = os.path.join(parent_dir, "processed", bat_dir)

        for filename in filenames:
            if is_img_file(filename):
                image_file = os.path.join(dirpath, filename)
                print("Processing", image_file)
                process_func(image_file, bat_output_path)
                print(bat_output_path)


def replace_green_background_to_bat_face_image(image_path, output_path):
    # Load the image from a file path
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    image = replace_green_background(image)

    filename = strip_filename_from_path(image_path)
    # cv2.imshow(filename, image)
    # cv2.waitKey(0)
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(f"{os.path.join(output_path, filename)}.png", image)
    cv2.destroyAllWindows()


def add_background_to_bat_face_image(image_path, output_path):
    # Load the image from a file path
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    image_processor = BatFaceSegmentationBackgroundProcessor()
    image = image_processor.process(
        image,
        preprocess_frame=None,
        prediction_processor=replace_green_background,
        post_prediction_processor=crop_rectangle_from_cv2_frame,
        threshold=0.5,
    )

    if image is None:
        print("Failed to process image", image_path)
        return

    filename = strip_filename_from_path(image_path)
    # cv2.imshow(filename, image)
    # cv2.waitKey(0)
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(f"{os.path.join(output_path, filename)}.png", image)
    cv2.destroyAllWindows()


# extract_images_from_video()
# augment_filtered_images_extracted_from_videos()

# trained_model = load_model()
# check_model_on_seen_data(trained_model)
# create_saliency_maps_on_data(trained_model,
#                            os.path.join(SIAMESE_PATHS["BASE"], 'face_recognition-with_bg_31_03_25'),
#                            os.path.join(SIAMESE_PATHS["BASE"], 'saliency_maps'))


# permutations([1, 2, 3, 4], 2) # [(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)]
# combinations([1, 2, 3, 4], 2) # [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# def main():
#     """Main entry point for the application."""
#     try:
#         # Configure GPU
#         configure_gpu()

#         # Ensure required directories exist
#         ensure_directories()

#         # Process training images
#         process_training_images(replace_green_background_to_bat_face_image, SIAMESE_PATHS["TRAINING"]["DATA"])

#         # Extract images from videos
#         extract_images_from_video()

#         # Augment filtered images
#         augment_filtered_images_extracted_from_videos()

#         # Build and train siamese network
#         siamese_network = build_siamese_network()
#         train_data_batches, test_data_batches = build_train_test_data_batches()
#         trained_model = get_trained_siamese_network(siamese_network, train_data_batches, test_data_batches)

#         # Create saliency maps
#         create_saliency_maps_on_data(trained_model,
#                                    os.path.join(SIAMESE_PATHS["BASE"], 'face_recognition-with_bg_31_03_25'),
#                                    os.path.join(SIAMESE_PATHS["BASE"], 'saliency_maps'))

#         print("Application completed successfully")

#     except Exception as e:
#         print(f"Error in main application: {e}")
#         raise

# if __name__ == "__main__":
#     main()


# Background Generation Strategies
def create_solid_color_background(height, width, color=(0, 255, 0)):
    """Create a solid color background.

    Args:
        height: Image height
        width: Image width
        color: BGR color tuple (default: green)
    """
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:] = color
    return background


def create_random_noise_background(height, width):
    """Create a random noise background."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def create_blur_background_generator(height, width):
    """Create a blurred random background (existing function wrapped)."""
    return create_blur_image(height, width)


def create_picsum_background_generator(height, width):
    """Create a background from Picsum Photos (existing function wrapped)."""
    return get_random_cropped_image(height, width)


def create_gradient_background(
    height,
    width,
    start_color=(0, 0, 0),
    end_color=(255, 255, 255),
    direction="vertical",
):
    """Create a gradient background.

    Args:
        height: Image height
        width: Image width
        start_color: Starting BGR color
        end_color: Ending BGR color
        direction: 'vertical', 'horizontal', or 'diagonal'
    """
    background = np.zeros((height, width, 3), dtype=np.uint8)

    if direction == "vertical":
        for i in range(height):
            ratio = i / height
            color = [
                int(start_color[j] + (end_color[j] - start_color[j]) * ratio)
                for j in range(3)
            ]
            background[i, :] = color
    elif direction == "horizontal":
        for i in range(width):
            ratio = i / width
            color = [
                int(start_color[j] + (end_color[j] - start_color[j]) * ratio)
                for j in range(3)
            ]
            background[:, i] = color
    elif direction == "diagonal":
        for i in range(height):
            for j in range(width):
                ratio = (i + j) / (height + width)
                color = [
                    int(start_color[k] + (end_color[k] - start_color[k]) * ratio)
                    for k in range(3)
                ]
                background[i, j] = color

    return background


def apply_custom_background_to_bat_image(
    image_path, output_path, background_generator, suffix=""
):
    """
    Generalized function to apply any background to a bat image using segmentation.

    Args:
        image_path: Path to input image
        output_path: Directory to save output image
        background_generator: Function that takes (height, width) and returns background image
        suffix: Optional suffix for output filename
    """
    # Load the image from a file path
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    height, width = image.shape[:2]

    # Create background using the provided generator
    background = background_generator(height, width)

    # Use segmentation to get bat mask
    image_processor = BatFaceSegmentationBackgroundProcessor()
    predictions = image_processor.model.predict(image, conf=0.5)[0]

    if predictions and len(predictions.masks) > 0:
        # Get the mask for the bat
        mask = predictions.masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (width, height))
        mask = mask.astype(bool)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Extract bat from original image
        bat_roi = image * mask

        # Place bat on background
        result_image = background.copy()
        result_image[mask] = bat_roi[mask]
    else:
        print("No bat detected in image, using original image")
        result_image = image

    # Save result
    filename = strip_filename_from_path(image_path)
    os.makedirs(output_path, exist_ok=True)
    output_filename = f"{filename}{suffix}.png"
    cv2.imwrite(os.path.join(output_path, output_filename), result_image)
    print(f"Saved: {os.path.join(output_path, output_filename)}")


# Convenience Functions for Common Background Types
def add_green_background_to_bat_image(image_path, output_path):
    """Add green background to bat image."""
    background_gen = lambda h, w: create_solid_color_background(h, w, (0, 255, 0))
    apply_custom_background_to_bat_image(
        image_path, output_path, background_gen, "_green_bg"
    )


def add_white_background_to_bat_image(image_path, output_path):
    """Add white background to bat image."""
    background_gen = lambda h, w: create_solid_color_background(h, w, (255, 255, 255))
    apply_custom_background_to_bat_image(
        image_path, output_path, background_gen, "_white_bg"
    )


def add_black_background_to_bat_image(image_path, output_path):
    """Add black background to bat image."""
    background_gen = lambda h, w: create_solid_color_background(h, w, (0, 0, 0))
    apply_custom_background_to_bat_image(
        image_path, output_path, background_gen, "_black_bg"
    )


def add_random_noise_background_to_bat_image(image_path, output_path):
    """Add random noise background to bat image."""
    apply_custom_background_to_bat_image(
        image_path, output_path, create_random_noise_background, "_noise_bg"
    )


def add_blur_background_to_bat_image(image_path, output_path):
    """Add blurred random background to bat image."""
    apply_custom_background_to_bat_image(
        image_path, output_path, create_blur_background_generator, "_blur_bg"
    )


def add_picsum_background_to_bat_image(image_path, output_path):
    """Add random Picsum photo background to bat image."""
    apply_custom_background_to_bat_image(
        image_path, output_path, create_picsum_background_generator, "_picsum_bg"
    )


def add_gradient_background_to_bat_image(
    image_path,
    output_path,
    start_color=(0, 0, 0),
    end_color=(255, 255, 255),
    direction="vertical",
):
    """Add gradient background to bat image."""
    background_gen = lambda h, w: create_gradient_background(
        h, w, start_color, end_color, direction
    )
    suffix = f"_gradient_{direction}_bg"
    apply_custom_background_to_bat_image(
        image_path, output_path, background_gen, suffix
    )


# Batch Processing Function
def process_images_with_custom_background(input_dir, output_dir, background_function):
    """
    Process all images in a directory with a custom background function.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        background_function: Function that takes (image_path, output_path) and processes the image
    """
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    for dirpath, dirnames, filenames in os.walk(input_dir):
        # Create corresponding output directory structure
        relative_path = os.path.relpath(dirpath, input_dir)
        current_output_dir = (
            os.path.join(output_dir, relative_path)
            if relative_path != "."
            else output_dir
        )

        for filename in filenames:
            if is_img_file(filename):
                image_path = os.path.join(dirpath, filename)
                print(f"Processing: {image_path}")
                background_function(image_path, current_output_dir)


# Example usage of the generalized background system
# Choose your background strategy:

# Option 1: Green background
# process_images_with_custom_background(SEGMENTATION_PATHS["DATA"], os.path.join(os.path.dirname(SEGMENTATION_PATHS["DATA"]), "processed_green"), add_green_background_to_bat_image)

# Option 2: White background
# process_images_with_custom_background(SEGMENTATION_PATHS["DATA"], os.path.join(os.path.dirname(SEGMENTATION_PATHS["DATA"]), "processed_white"), add_white_background_to_bat_image)

# Option 3: Random Picsum photos
# process_images_with_custom_background(
#     SEGMENTATION_PATHS["DATA"],
#     os.path.join(os.path.dirname(SEGMENTATION_PATHS["DATA"]), "processed_picsum"),
#     add_picsum_background_to_bat_image,
# )  # TODO: Fix configuration system

# Option 4: Custom color background
# custom_blue_bg = lambda img_path, out_path: apply_custom_background_to_bat_image(img_path, out_path, lambda h, w: create_solid_color_background(h, w, (255, 0, 0)), "_blue_bg")
# process_images_with_custom_background(SEGMENTATION_PATHS["DATA"], os.path.join(os.path.dirname(SEGMENTATION_PATHS["DATA"]), "processed_blue"), custom_blue_bg)

# Option 5: Gradient background
# gradient_bg = lambda img_path, out_path: add_gradient_background_to_bat_image(img_path, out_path, (0, 0, 255), (255, 255, 0), 'diagonal')
# process_images_with_custom_background(SEGMENTATION_PATHS["DATA"], os.path.join(os.path.dirname(SEGMENTATION_PATHS["DATA"]), "processed_gradient"), gradient_bg)

# Current default: Use the original function for backward compatibility
# process_training_images(
#     process_func=add_background_to_bat_face_image, base_dir=SEGMENTATION_PATHS["DATA"]
# )
