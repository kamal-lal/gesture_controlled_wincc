"""
title   :   2_train_model.py
author  :   Kamal Lal (www.linkedin.com/in/kamal-lal-40671188/)
date    :   May 2020
version :   0.1
python  :   3.7

notes:
  - This module trains a neural network on 'hand gesture' images.
  - Assumes required set of images are created using '1_create_dataset.py' module.

usage:
  - Make sure the 'training_imgs' directory structure exists.
  - Run the module.

"""

# built-in modules import
import os
import cv2
import random

# external modules import
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout


IMG_SIZE = 50


def create_dataset():
    """
    Function traverses the 'training_imgs' folder and creates a labelled dataset.

    Args:
        (nil)

    Returns:
        list: This is a list of list, with inner-list structure [image, label].

    """

    labelled_dataset = []
    inner_folders = ['0', '1', '2', '3', '4', '5', 'none']

    for folder in inner_folders:
        path = os.path.join(os.getcwd(), 'training_imgs', folder)
        files_list = os.listdir(path)
        label = inner_folders.index(folder)
        for file in files_list:
            img_file = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img_file, (IMG_SIZE, IMG_SIZE))
            resized_img = cv2.threshold(resized_img, 30, 255, cv2.THRESH_BINARY)[1]
            labelled_dataset.append([resized_img, label])

    return labelled_dataset


# create dataset and shuffle it
dataset = create_dataset()
random.shuffle(dataset)

# separate images and labels from dataset
train_imgs = []
train_labels = []
for img, lbl in dataset:
    train_imgs.append(img)
    train_labels.append(lbl)

# pre-processing image data
train_imgs = np.array(train_imgs)
train_imgs = train_imgs.reshape((-1, IMG_SIZE, IMG_SIZE, 1))
train_imgs = train_imgs / 255

# pre-processing labels
train_labels = np.array(train_labels)
train_labels = tf.keras .utils.to_categorical(train_labels)

# create a Neural Netwok model
model = tf.keras.models.Sequential()

# add layers to the model
model.add(Conv2D(64, kernel_size=(5, 5), activation=relu, input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation=relu))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation=relu))
model.add(Dense(7, activation=softmax))

# compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# training the model
# NB: If running in IDLE, SublimeText etc., and random characters gets printed and display gets
#     noisy, then update the below with verbose=2.
model.fit(train_imgs, train_labels, batch_size=32, epochs=5, validation_split=0.1, verbose=1)

# saving the model
model.save('gest_recog_model.h5')
