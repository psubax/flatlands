import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, \
    Activation, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

data = np.load('flatland_train.npz')
X = data['X']
y = data['y']

y[y != 0] -= 2  # Correct labels so that triangle is mapped to class 1
X = X / 255.  # Scale down to range [0, 1]

# Construct and train your model (don't forget train/test split and other tricks)
# model = ...

# Save the model and upload it to git
# model.save('model.h5')

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu', input_shape=[50, 50, 1]))
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

loss = model.fit(X,
                 y,
                 epochs=32,
                 batch_size=128,
                 validation_split=0.2)

model.save('model.h5')
