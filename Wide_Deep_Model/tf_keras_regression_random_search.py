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
导入测试数据集,并拆分训练集、验证集和测试级
'''
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing.DESCR)
print(housing.target.shape)
print(housing.data.shape)

import pprint
pprint.pprint(housing.data[:5])
pprint.pprint(housing.target[:5])

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7, test_size=0.25)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

'''
归一化数据
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

'''
搭建模型
- 转换为sklearn的model
- 定义参数集合
- 搜索参数
'''
def build_model(hidden_layers = 1, layer_size = 30, learning_rate =3e-3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layer_size,
                                    activation='relu', input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(layer_size, activation= 'relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

sklearn_model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]
history = sklearn_model.fit(x_train_scaled, y_train,
                  epochs = 100,
                  validation_data = (x_valid_scaled, y_valid),
                  callbacks = callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

