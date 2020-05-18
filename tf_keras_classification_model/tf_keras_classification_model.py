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

'''
测试版本信息与模块名字
'''
print(tf.__version__)
print(sys.version_info)
for moudule in mpl, np , pd , sklearn, tf, keras:
    print(moudule.__name__, moudule.__version__)
'''
导入测试数据集
'''
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

# 将训练级的数据拆分成验证级和训练级
# 将前5000张用作验证
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

def show_single_image(img_arr):
    plt.imshow(img_arr, cmap="binary")
    plt.show()

# show_single_image(x_train[0])

def show_images(n_rows, n_cols, x_data, y_data, class_names):
    assert  len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize= (n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index], cmap="binary",interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[y_data[index]])
    plt.show()
class_name = ['T-shirt', 'Thouser', 'Pullover', 'Dress', 'Cost', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# show_images(3, 5, x_train, y_train, class_name)

# 模型构建
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28]))
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model.layers
model.summary()

# train
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

# 对数据集进行归一化
plot_learning_curves(history)