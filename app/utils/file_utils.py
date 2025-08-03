import os
from utils.image_utils import is_DS_Store

def get_files_from_dir(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir) if
            os.path.isfile(os.path.join(dir, f)) and not is_DS_Store(f)]

def collect_files(directory):
    image_files = set()
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            if is_img_file(file):
                image_files.add(file)
    return image_files

def process_training_images(process_func, base_dir, output_dir):
    for dirpath, dirnames, filenames in os.walk(base_dir):
        bat_dir = dirpath.split(os.sep)[-1]
        bat_output_path = os.path.join(base_dir, output_dir, bat_dir)

        for filename in filenames:
            if is_img_file(filename):
                image_file = os.path.join(dirpath, filename)
                print("Processing", image_file)
                process_func(image_file, bat_output_path)
                print(bat_output_path) 