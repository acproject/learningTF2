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
多输入
'''
input_wide = tf.keras.layers.Input(shape=[5])
input_deep = tf.keras.layers.Input(shape=[6])
hidden1 = tf.keras.layers.Dense(30, activation="relu")(input_deep)
hidden2 = tf.keras.layers.Dense(30, activation='relu')(hidden1)
concat = tf.keras.layers.concatenate([input_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
output2 = tf.keras.layers.Dense(1)(hidden2)
model = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output, output2])
model.build(input_shape=(None,8))

model.summary()

model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

'''
拟合
'''
x_train_scaled_wide = x_train_scaled[:, :5]
x_train_scaled_deep = x_train_scaled[:, 2:]
x_valid_scaled_wide = x_valid_scaled[:, :5]
x_valid_scaled_deep = x_valid_scaled[:, 2:]
x_test_scaled_wide = x_test_scaled[:, :5]
x_test_scaled_deep = x_test_scaled[:, 2:]
history = model.fit(
    [x_train_scaled_wide, x_train_scaled_deep], [y_train, y_train],
    validation_data=([x_valid_scaled_wide, x_valid_scaled_deep], [y_valid, y_valid]), epochs=100, callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

model.evaluate([x_test_scaled_wide, x_test_scaled_deep], [y_test, y_test])
