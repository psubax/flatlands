import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, \
    Activation, Dropout, MaxPooling2D, RandomRotation, ZeroPadding2D, Reshape, \
    RandomTranslation, RandomContrast, RandomBrightness, RandomFlip, Resizing
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

# Augmentation factor determines the size of augmented data
augmentation_factor = 2
# Determines the size of augmented image
new_dim = 50

# Generate zeros array to be used later for augmentation
X_generated = np.zeros([int(augmentation_factor * X.shape[0]), new_dim, new_dim])

# Copy the original image dataset
indices = [0, X.shape[0]]

for index in indices:
    X_generated[index:X.shape[0] + index, :, :] = X[:X.shape[0], :, :]
X_generated = np.expand_dims(X_generated, axis=-1)
y_generated = np.concatenate(augmentation_factor * [y])

# Defining image data generator
# Parameters selected to avoid possible loss of relevant pixels
# e.g. the angle is small so that the figure does not get
# rotated out of the picture
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='constant',
)

augmented_data = datagen.flow(
    X_generated,
    y_generated,
    batch_size=X_generated.shape[0]
)

X_augmented = augmented_data[0][0]
y_augmented = augmented_data[0][1]

# Defining the CNN
model = Sequential()
# Including random contrasting
model.add(RandomContrast(factor=0.5, input_shape=[X.shape[1], X.shape[2], 1]))
# Maintain dimensions with same padding
model.add(Conv2D(6, kernel_size=(3, 3), padding='same', activation='relu'))
# Dropout to prevent overfitting
model.add(Dropout(0.1))
model.add(Conv2D(10, kernel_size=(4, 4), padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(17, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(13, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(5, activation='softmax'))
# Learning rate set to optimal value obtained through testing
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=0.005),
              metrics=[SparseCategoricalAccuracy()],
              )
model.summary()

# Optimal no. of epochs selected
# Using the usual train-test split
loss = model.fit(X_augmented,
                 y_augmented,
                 epochs=45,
                 batch_size=128,
                 validation_split=0.2)

for metric in loss.history.keys():
    plt.plot(loss.history[metric], label=metric)

plt.title('Training Metrics Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.show()

model.save('try11.h5')
