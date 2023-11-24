import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, \
    Activation, Dropout, MaxPooling2D, RandomRotation, ZeroPadding2D
from keras.models import Sequential
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

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
padding = (11, 11, 11, 11)

X_padded = np.zeros([X.shape[0], 72, 72])

for i in range(X.shape[0]):
    padded_image = np.pad(X[i, :, :], ((padding[1], padding[3]), (padding[0], padding[2])), mode='constant')
    X_padded[i, :, :] = padded_image

X = X_padded
X = np.expand_dims(X, (-1))

model = Sequential()
# model.add(ZeroPadding2D(padding=11, input_shape=[50, 50, 1]))
# model.add(RandomRotation(factor=np.pi / 2, fill_mode='constant', fill_value=0))
model.add(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu', input_shape=[72, 72, 1]))
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
              metrics=[SparseCategoricalAccuracy()])
model.summary()

# Create an instance of ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=90,
    shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a generator for reading images from the specified directory
train_generator = datagen.flow(
    X_train,
    y_train,
    batch_size=256
)

# Train your model using the generator
model.fit(train_generator, epochs=50, batch_size=256)
pred = model.predict(X_test).argmax(axis=1)
print('Accuracy on test set - {0:.02%}'.format((pred == y_test).mean()))
# loss = model.fit(X,
#                  y,
#                  epochs=50,
#                  batch_size=128,
#                  validation_split=0.2)

model.save('model.h5')
