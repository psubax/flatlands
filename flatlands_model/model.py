import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, \
    Activation, Dropout, MaxPooling2D, RandomRotation, ZeroPadding2D, Reshape, \
    RandomTranslation, RandomContrast, RandomBrightness, RandomFlip
from keras.models import Sequential
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2


data = np.load('flatland_train.npz')
X = data['X']
y = data['y']

y[y != 0] -= 2  # Correct labels so that triangle is mapped to class 1
X = X / 255.  # Scale down to range [0, 1]

# Construct and train your model (don't forget train/test split and other tricks)
# model = ...

# Save the model and upload it to git
# model.save('model.h5')


# Specify the amount of padding for each side (left, top, right, bottom)
# padding = (11, 11, 11, 11)
#
# X_padded = np.zeros([X.shape[0], 72, 72])
#
# for i in range(X.shape[0]):
#     padded_image = np.pad(X[i, :, :], ((padding[1], padding[3]), (padding[0], padding[2])), mode='constant')
#     X_padded[i, :, :] = padded_image
#
# X = X_padded
# X = np.expand_dims(X, (-1))

X = np.expand_dims(X, axis=-1)
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant'
)

train_generator = datagen.flow(
    train_data,
    train_labels,
    batch_size=128
)


model = Sequential()
# model.add(ZeroPadding2D(11, input_shape=[50, 50, 1]))
# model.add(RandomRotation(factor=np.pi/6, input_shape=[50, 50, 1]))
# model.add(RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)))#, fill_mode='constant', fill_value=0.0))
# model.add(RandomFlip(input_shape=[50, 50, 1]))
model.add(RandomContrast(factor=0.5, input_shape=[50, 50, 1]))
# model.add(ZeroPadding2D(padding=11, input_shape=[50, 50, 1]))
# model.add(RandomRotation(factor=np.pi / 2, fill_mode='constant', fill_value=0))
model.add(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.1))
model.add(Conv2D(16, kernel_size=(4, 4), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.1))
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(),
              metrics=[SparseCategoricalAccuracy()],
              )
model.summary()

# # Create an instance of ImageDataGenerator for data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=90,
#     shear_range=0.2,
#     zoom_range=0.2,
#     # horizontal_flip=True
# )
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Create a generator for reading images from the specified directory
# train_generator = datagen.flow(
#     X_train,
#     y_train,
#     batch_size=256
# )
#
# # Train your model using the generator
# model.fit(train_generator, epochs=50, batch_size=256)
# pred = model.predict(X_test).argmax(axis=1)
# print('Accuracy on test set - {0:.02%}'.format((pred == y_test).mean()))
loss = model.fit(train_generator,
                 epochs=1,
                 validation_data=(test_data, test_labels))

model.save('model.h5')


# image = X[0, :, :, 0]
#
# resized_image_gray = cv2.resize(image, (50, 50))
#
# # Display the original and resized images (optional)
# cv2.imshow('Original Grayscale Image', image)
# cv2.imshow('Resized Grayscale Image', resized_image_gray)