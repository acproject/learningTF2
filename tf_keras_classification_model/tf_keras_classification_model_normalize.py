import matplotlib as mpl
import matplotlib.pyplot as plt
# 如果需要在Jupyter的Notebook中显示matplotlib的图像需要使用下面的语句
# %matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

# 将训练级的数据拆分成验证级和训练级
# 将前5000张用作验证
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

# x = (x-u) / std

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28])) # output (28*28=784)
model.add(tf.keras.layers.Dense(300, activation="relu")) # output (784 * 300 + 300 = 235500)
model.add(tf.keras.layers.Dense(100,activation="relu")) # output ( 300 * 100 + 100 = 30100)
model.add(tf.keras.layers.Dense(10,activation="softmax")) # output (10 * 100 + 10 = 1010)
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model.layers
model.summary()

history = model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid))

model.evaluate(x_test, y_test)
model.evaluate(x_test_scaled, y_test)
