# Created by Chen Ming, 10/1/2021, chen_ming@brown.edu

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.keras import datasets, layers, models, regularizers
from keras.callbacks import Callback
from keras.models import Model

"""## Load training data and labels. They are generated in MATLAB and saved in .mat format.
allinone.mat contains the time-frequency representations of the echoes from 4 categories of targets - 1-glint, 2-glint, 3-glint, and 4 -glint.
labels.mat contains the labels of above data
"""

datadict = sio.loadmat('/content/drive/My Drive/cnn backup/allinone.mat') # <-- pls change the directory accordingly
labelsdict = sio.loadmat('/content/drive/My Drive/cnn backup/labels2.mat') # <-- pls change the directory accordingly
data = np.array(datadict['glint'])
data = np.expand_dims(data, axis=3) # input size - batch_size, height-freq, width-time, depth-leftear
print(data.shape)
label = np.array(labelsdict['labels'])

label_train, label_test, coch_train, coch_test = train_test_split(label, data, test_size=0.25, shuffle=True, random_state=30)
print(label_train.shape,  label_test.shape)
num_classes = 4
# Construct the CNN architecture
model = models.Sequential()

# conv - 1
model.add(layers.Conv2D(8, (5, 5), padding='same', input_shape=(161,1100,1))) #  CNN takes tensors of shape (image_height, image_width, color_channels)
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(2,2)))
# conv - 2
model.add(layers.Conv2D(16, (5, 5), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(2,2)))
# conv - 3
model.add(layers.Conv2D(32, (5, 5), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(2,2)))
# conv - 4
model.add(layers.Conv2D(64, (5, 5), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(2,2)))

model.add(layers.Flatten())
#model.add(layers.Dense(32, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(units=num_classes, 
                       use_bias=True, 
                       kernel_regularizer=regularizers.l2(0.2), 
                       activation='softmax'))

model.summary()

print(label_train[0:4])
plt.matshow(coch_train[0,:,:,0])
plt.matshow(coch_train[1,:,:,0])
plt.matshow(coch_train[2,:,:,0])
plt.matshow(coch_train[3,:,:,0])

"""# Some notes about choice of loss functions:
 using SparseCategoricalCrossentropy integer-tokens are converted to a one-hot-encoded label **starting at 0**. So it creates it, but it is not in your data. So having two classes you need to provide the labels as 0 and 1. And not -1 and 1. Therefore it is as you write, you can either:

Run it with one-hot encoding using Categorical Crossentropy or
Run it with integer labels 0 and 1 using Sparse Categorical Crossentropy,
"""

# Compile and Train
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', 
#               optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2), 
#               metrics=['accuracy'])

history = model.fit(coch_train, label_train, epochs=250, 
                    validation_data=(coch_test, label_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(coch_test, label_test, verbose=2)
