import random
import os
import tensorflow as tf


def is_DS_Store(file):
    return file == '.DS_Store'


def count_files_in_directory(dir_path):
    return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])


def find_min_files_count_in_sub_dirs(dir_path):
    return min([count_files_in_directory(os.path.join(dir_path, dir_name)) for dir_name in os.listdir(dir_path)])


def crop_rectangle_from_cv2_frame(prediction, frame):
    min_x1, min_y1, max_x2, max_y2, score, class_id = prediction
    box_width = max_x2 - min_x1
    box_height = max_y2 - min_y1
    x_center = (min_x1 + max_x2) / 2
    y_center = (min_y1 + max_y2) / 2

    # crop rectangle
    # Define the rectangle region of interest (ROI)
    edge_length = min(box_width, box_height)
    x1_rect = int(x_center - edge_length / 2)
    x2_rect = int(x_center + edge_length / 2)
    y1_rect = int(y_center - edge_length / 2)
    y2_rect = int(y_center + edge_length / 2)
    return frame[y1_rect:y2_rect, x1_rect:x2_rect]


def build_train_test_data(data, training_portion=0.8):
    # Build dataloader pipeline
    data_size = len(data)
    train_size = round(data_size * training_portion)

    # Split the data into training and testing sets
    train_data = data.take(train_size)
    test_data = data.skip(train_size)

    return train_data, test_data



class LibraryConversionUtils:
    @staticmethod
    def yolo_to_albumentation(yolo_box):
        x_center, y_center, width, height = yolo_box
        print(x_center, y_center, width, height)
        # Convert from YOLO format to corner coordinates
        x_min = max(0, (x_center - width / 2))
        y_min = max(0, (y_center - height / 2))
        x_max = max(0, (x_center + width / 2))
        y_max = max(0, (y_center + height / 2))

        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def albumentation_to_yolo(albumentation_box, image_width, image_height):
        x_min, y_min, x_max, y_max = albumentation_box

        # Calculate the center, width, and height of the bounding box
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Normalize the bounding box by the image size
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        return [x_center, y_center, width, height]


def create_random_image(image):
    shape = tf.shape(image).numpy()
    random_image = tf.random.uniform(shape, minval=0, maxval=256, dtype=tf.int32)
    return random_image


def is_img_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))


def strip_filename_from_path(full_path):
    filename_with_extension = os.path.basename(full_path)
    filename, extension = os.path.splitext(filename_with_extension)
    return filename



