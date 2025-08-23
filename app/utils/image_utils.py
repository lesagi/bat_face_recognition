import os
import cv2

def is_DS_Store(file):
    return file == '.DS_Store'

def is_img_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def strip_filename_from_path(full_path):
    filename_with_extension = os.path.basename(full_path)
    filename, extension = os.path.splitext(filename_with_extension)
    return filename

def create_random_image(image):
    # Import TensorFlow only when this function is actually called
    import tensorflow as tf
    shape = tf.shape(image).numpy()
    random_image = tf.random.uniform(shape, minval=0, maxval=256, dtype=tf.int32)
    return random_image

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