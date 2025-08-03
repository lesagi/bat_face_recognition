
# Import standard dependencies
import cv2 as cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow import keras as tfks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

ANNOTATED_VIDEOS_SNAPSHOTS_DIR = os.path.abspath(os.path.join('..','videos','annotated'))
RAW_DATA = os.path.abspath(os.path.join('.'))
CHECKPOINTS_DIR = os.path.abspath(os.path.join('.','training_checkpoints'))
os.makedirs(CHECKPOINTS_DIR)

EPOCHS = 1
IMAGES_AMOUNT_LIMIT = 10
INPUT_EDGE_LENGTH = 105

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


def strip_filename_from_path(file_path):
    filename = os.path.split(file_path)[-1]
    return os.path.splitext(filename)[0]

def get_bat_snapshot_images_dir_path(bat_name):
    return os.path.join(ANNOTATED_VIDEOS_SNAPSHOTS_DIR, bat_name, 'images_rect_cropped')

def get_images_iterator_from_annotated_bat_snapshots_dir(bat_dir_name, limit = -1, format = 'png'):
    file_pattern = os.path.join(get_bat_snapshot_images_dir_path(bat_dir_name),'*.'+format)
    return tf.data.Dataset.list_files(file_pattern).take(limit)

def train_test_split(dataset, train_size=0.5, random_seed=42, shuffle=False):
    ds_size = int(dataset.cardinality())
    if shuffle:
        dataset.shuffle(buffer_size=ds_size, seed=random_seed)

    train_size = int(train_size * ds_size)
    val_size = ds_size - train_size

    return dataset.take(train_size), dataset.skip(train_size)


# find min files in bat snapshots
min_images_count_for_bat = min([int(get_images_iterator_from_annotated_bat_snapshots_dir(bat_snapshots_dir_name, -1).cardinality()) for bat_snapshots_dir_name in os.listdir(ANNOTATED_VIDEOS_SNAPSHOTS_DIR)])
min_images_count_for_bat = int(min_images_count_for_bat/2)*2
min_images_count_for_bat

def build_train_samples_from_anchor_and_counterpart(anchor, counterpart, is_positive):
    size = len(anchor)
    labels = tf.ones(size) if is_positive else tf.zeros(size)
    return tf.data.Dataset.zip((anchor, counterpart, tf.data.Dataset.from_tensor_slices(labels)))

# Collect labelled data
labelled_data = tf.data.Dataset.from_tensor_slices(
    (tf.constant([], dtype=tf.string),
    tf.constant([], dtype=tf.string),
    tf.constant([], dtype=tf.float32))
)


# iterating over the bats' snapshots and for each directory
# split the content of it to 2 and build pairs 
for anchor_bat_snapshots_dir_name in os.listdir(ANNOTATED_VIDEOS_SNAPSHOTS_DIR):
    images_sample_count = min(IMAGES_AMOUNT_LIMIT, min_images_count_for_bat)
    anchor = get_images_iterator_from_annotated_bat_snapshots_dir(anchor_bat_snapshots_dir_name, images_sample_count)
    anchor, pos = train_test_split(anchor, train_size=0.5, shuffle=True)
    labelled_data = labelled_data.concatenate(build_train_samples_from_anchor_and_counterpart(anchor, pos, True))
    
    for neg_bat_snapshots_dir_name in os.listdir(ANNOTATED_VIDEOS_SNAPSHOTS_DIR):
        if anchor_bat_snapshots_dir_name != neg_bat_snapshots_dir_name:
            neg = get_images_iterator_from_annotated_bat_snapshots_dir(neg_bat_snapshots_dir_name, len(anchor))
            labelled_data = labelled_data.concatenate(build_train_samples_from_anchor_and_counterpart(anchor, neg, False))

def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_png(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (INPUT_EDGE_LENGTH,INPUT_EDGE_LENGTH))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


def make_embedding(input_edge_length): 
    inp = Input(shape=(input_edge_length, input_edge_length, 3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


class L1Dist(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model(input_edge_length):
    embedding = make_embedding(INPUT_EDGE_LENGTH)
    input_img = Input(name='input_img', shape=(input_edge_length, input_edge_length, 3))
    validation_img = Input(name='validation_img', shape=(input_edge_length, input_edge_length, 3))

    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_img)[0], embedding(validation_img)[0])

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_img, validation_img], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model(INPUT_EDGE_LENGTH)

# Build dataloader pipeline
data = labelled_data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=len(labelled_data))

# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

checkpoint_prefix = os.path.join(CHECKPOINTS_DIR, 'ckpt')
os.makedirs(checkpoint_prefix)
checkpoint_best_prefix = os.path.join(CHECKPOINTS_DIR, 'ckpt_best')
os.makedirs(checkpoint_best_prefix)
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


@tf.function
def train_step(batch):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss


def train(data, epochs):
    # Loop through epochs
    min_loss = 1
    for epoch in range(1, epochs+1):
        print('\n Epoch {}/{}'.format(epoch, epochs))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
            
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        if loss.numpy() <= min_loss:
            checkpoint.save(file_prefix=os.path.join(checkpoint_best_prefix, 'ckpt_best'))
            
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


train(train_data, EPOCHS)
