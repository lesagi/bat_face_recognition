import os
import shutil

from random import shuffle  # Corrected import

BASE_PATH = '../face_recognition-no_background_05_07_24'
RAW_DATA_PATH = os.path.join(BASE_PATH, 'raw_data')
base_dirs = [dir_name for dir_name in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, dir_name))]


def move_files_from_sub_dirs_to_parent(base_dirs, sub_dir_name):
    for base_dir in base_dirs:
        sub_dir_path = os.path.join(RAW_DATA_PATH, base_dir, sub_dir_name)
        sub_dir_files = os.listdir(sub_dir_path)
        for file in sub_dir_files:
            shutil.move(os.path.join(sub_dir_path, file),
                        os.path.join(RAW_DATA_PATH, base_dir, file))
        os.rmdir(sub_dir_path)


def divide_raw_data_folders_to_train_test_val(raw_data_dir, train_test_val_base_dir, training_portion=0.7):
    all_dirs = os.listdir(raw_data_dir)
    exclusion_size = round(len(all_dirs) * training_portion)
    training_path = os.path.join(train_test_val_base_dir, 'training')
    os.makedirs(training_path, exist_ok=True)
    validation_path = os.path.join(train_test_val_base_dir, 'validation')
    os.makedirs(validation_path, exist_ok=True)

    for_training_dirs = all_dirs[:exclusion_size]
    for bat_dir in for_training_dirs:
        if not os.path.isdir(os.path.join(raw_data_dir, bat_dir)):
            continue

        # copy all folder from raw_data to training
        current_photos_path = os.path.join(raw_data_dir, bat_dir)
        new_photos_path = os.path.join(training_path, bat_dir)
        shutil.move(current_photos_path, new_photos_path)

        # move some of the files from training to validation
        bat_dir_size = round(len(os.listdir(new_photos_path)) * 0.85)
        all_photos = os.listdir(new_photos_path)
        validation_photos = all_photos[bat_dir_size:]
        for photo in validation_photos:
            os.makedirs(os.path.join(validation_path, bat_dir), exist_ok=True)
            shutil.move(os.path.join(new_photos_path, photo), os.path.join(validation_path, bat_dir))

    excluded_dirs = all_dirs[exclusion_size:]
    unseen_path = os.path.join(train_test_val_base_dir, 'unseen')
    for bat_dir in excluded_dirs:
        if not os.path.isdir(os.path.join(raw_data_dir, bat_dir)):
            continue

        # copy all folder from raw_data to training
        current_photos_path = os.path.join(raw_data_dir, bat_dir)
        new_photos_path = os.path.join(unseen_path, bat_dir)
        shutil.move(current_photos_path, new_photos_path)

        # move some of the files from training to validation
        bat_dir_size = round(len(os.listdir(new_photos_path)) * 0.85)
        all_photos = os.listdir(new_photos_path)
        validation_photos = all_photos[bat_dir_size:]
        for photo in validation_photos:
            os.makedirs(os.path.join(validation_path, bat_dir), exist_ok=True)
            shutil.move(os.path.join(new_photos_path, photo), os.path.join(validation_path, bat_dir))


def reduce_folders_count(reduction_factor=0.5):
    # get random 0.15 of the training folders, and then delete the corresponding folders with the same name in the validation folder
    training_path = os.path.join(BASE_PATH, 'data', 'training')
    validation_path = os.path.join(BASE_PATH, 'data', 'validation')
    training_dirs = os.listdir(training_path)
    os.listdir(validation_path)
    shuffle(training_dirs)
    to_remove = training_dirs[:int(len(training_dirs) * reduction_factor)]
    for folder in to_remove:
        if not os.path.isdir(os.path.join(training_path, folder)):
            continue
        shutil.rmtree(os.path.join(training_path, folder))
        if os.path.exists(os.path.join(validation_path, folder)):
            shutil.rmtree(os.path.join(validation_path, folder))

    # now the same for the unseen folder
    unseen_path = os.path.join(BASE_PATH, 'data', 'unseen')
    unseen_dirs = os.listdir(unseen_path)
    shuffle(unseen_dirs)
    to_remove = unseen_dirs[:int(len(unseen_dirs) * reduction_factor)]
    for folder in to_remove:
        if not os.path.isdir(os.path.join(unseen_path, folder)):
            continue
        shutil.rmtree(os.path.join(unseen_path, folder))
        if os.path.exists(os.path.join(validation_path, folder)):
            shutil.rmtree(os.path.join(validation_path, folder))


def merge_files_from_validation_into_unseen_and_training():
    validation_path = os.path.join(BASE_PATH, 'data', 'validation')
    unseen_path = os.path.join(BASE_PATH, 'data', 'unseen')
    training_path = os.path.join(BASE_PATH, 'data', 'training')
    validation_dirs = os.listdir(validation_path)
    for folder in validation_dirs:
        if not os.path.isdir(os.path.join(validation_path, folder)):
            continue
        if os.path.exists(os.path.join(unseen_path, folder)):
            unseen_photos = os.listdir(os.path.join(validation_path, folder))
            for photo in unseen_photos:
                if not os.path.isdir(os.path.join(validation_path, folder, photo)):
                    continue
                shutil.move(os.path.join(validation_path, folder, photo), os.path.join(unseen_path, folder))
            shutil.rmtree(os.path.join(validation_path, folder))
        else:
            training_photos = os.listdir(os.path.join(validation_path, folder))
            for photo in training_photos:
                if not os.path.isdir(os.path.join(validation_path, folder, photo)):
                    continue
                shutil.move(os.path.join(validation_path, folder, photo), os.path.join(training_path, folder))
            shutil.rmtree(os.path.join(validation_path, folder))

def split_training_and_unseen_files_into_validation_folder(validation_portion=0.15):
    validation_path = os.path.join(BASE_PATH, 'data', 'validation')
    unseen_path = os.path.join(BASE_PATH, 'data', 'unseen')
    training_path = os.path.join(BASE_PATH, 'data', 'training')
    training_dirs = os.listdir(training_path)
    unseen_dirs = os.listdir(unseen_path)
    for folder in training_dirs:
        if not os.path.isdir(os.path.join(training_path, folder)):
            continue
        training_photos = os.listdir(os.path.join(training_path, folder))
        shuffle(training_photos)
        to_move = training_photos[:int(len(training_photos) * validation_portion)]
        for photo in to_move:
            os.makedirs(os.path.join(validation_path, folder), exist_ok=True)
            shutil.move(os.path.join(training_path, folder, photo), os.path.join(validation_path, folder))

    for folder in unseen_dirs:
        if not os.path.isdir(os.path.join(unseen_path, folder)):
            continue
        unseen_photos = os.listdir(os.path.join(unseen_path, folder))
        shuffle(unseen_photos)
        to_move = unseen_photos[:int(len(unseen_photos) * validation_portion)]
        for photo in to_move:
            os.makedirs(os.path.join(validation_path, folder), exist_ok=True)
            shutil.move(os.path.join(unseen_path, folder, photo), os.path.join(validation_path, folder))


# move_files_from_sub_dirs_to_parent(base_dirs, 'segmentation')
# divide_raw_data_folders_to_train_test_val(RAW_DATA_PATH, os.path.join(BASE_PATH, 'data'), 0.7)
# reduce_folders_count(0.7)
# merge_files_from_validation_into_unseen_and_training()
split_training_and_unseen_files_into_validation_folder(0.2)