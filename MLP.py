import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPool2D

df = pd.read_csv('ad_dataset.csv')
df = df.drop('Unnamed: 0', axis=1)

X = df.drop('target', axis=1)
y = df['target']
print(X.shape)

import seaborn as sns
sns.countplot(y=y, data=df, palette="mako_r")
plt.title("Result before oversampling")
plt.ylabel('Target')
plt.xlabel('Total')
plt.show()

from imblearn.over_sampling import SMOTE

nm = SMOTE()
X, y = nm.fit_resample(X, y)

sns.countplot(y=y, data=df, palette="mako_r")
plt.title("Result after oversampling")
plt.ylabel('Loan Status')
plt.xlabel('Total')
plt.show()

train_data, test_data, train_labels, test_labels = train_test_split(X, y, 
                                                                    test_size = 0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, 
                                                                  test_size = 0.2, random_state=42)

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

n_cols = train_data.shape[1]


early_stopping = EarlyStopping(monitor='val_loss', patience=6)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

model = Sequential()
model.add(Dense(512, input_shape=(n_cols,), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

model.add(Dense(4, activation='softmax'))

METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'), 
          ]

CALLBACKS = [early_stopping,reduce_lr]
    
model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=METRICS)

model.summary()

EPOCHS = 100

history = model.fit(train_data, train_labels, 
                    validation_data=(val_data, val_labels), 
                    callbacks=CALLBACKS, epochs=EPOCHS, batch_size=32)

fig, ax = plt.subplots(1, 3, figsize = (30, 5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "auc", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
plt.show()

print(model.evaluate(test_data, test_labels))





