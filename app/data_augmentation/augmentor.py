"""
Data augmentation module.
"""

import os
import cv2
import albumentations as alb
from utils.image_utils import is_DS_Store
from config.paths import FACE_RECOGNITION_PATHS

class ImagesAugmentor:
    def __init__(self, source_dir, augmentation_function, output_dir=None):
        self.source_dir = source_dir
        self.output_dir = output_dir if output_dir else source_dir
        self.augmentation_function = augmentation_function

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
                aug_img_path = os.path.join(output_dir, f'{image_name}.{x}.png')
                cv2.imwrite(aug_img_path, aug_img)

def augment_filtered_images():
    def augment_image(img_path):
        augmentor = alb.Compose([alb.HorizontalFlip(p=0.5),
                                 alb.RandomBrightnessContrast(p=0.5),
                                 alb.RandomGamma(p=0.5),
                                 alb.RGBShift(p=0.5),
                                 alb.VerticalFlip(p=0.5),
                                 alb.AdvancedBlur()])

        img = cv2.imread(img_path)
        return augmentor(image=img)['image']

    # Augment training images
    images_dirs = [dir_name for dir_name in os.listdir(FACE_RECOGNITION_PATHS["AUGMENTATION"]["INPUT"]) if
                   os.path.isdir(os.path.join(FACE_RECOGNITION_PATHS["AUGMENTATION"]["INPUT"], dir_name))]
    for images_dir in images_dirs:
        image_augmentor = ImagesAugmentor(os.path.join(FACE_RECOGNITION_PATHS["AUGMENTATION"]["INPUT"], images_dir), augment_image,
                                          FACE_RECOGNITION_PATHS["AUGMENTATION"]["OUTPUT"])
        image_augmentor.augment_images(output_sub_dir=images_dir, samples_count=60) 