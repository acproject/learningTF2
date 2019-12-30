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
tf.keras.backend.set_floatx('float64')
'''
归一化数据
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)




# metric使用
metric = tf.keras.metrics.MeanSquaredError()
print(metric([5.], [2.]))
metric.reset_states()

epochs = 100
batch_size = 32
steps_per_epoch = len(x_train_scaled) // batch_size
optimizer = tf.keras.optimizers.SGD()

def random_batch(x, y, batch_size=32):
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]
'''
搭建模型
'''
'''
拟合
'''
# - batch 遍历训练集 metric
#   - 自动求导
# - epoch 结束
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(1),
])

for epoch in range(epochs):
    metric.reset_states()
    for step in range(steps_per_epoch):
        x_batch, y_batch = random_batch(x_train_scaled, y_train, batch_size)

        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_batch, y_pred))
            metric(y_batch, y_pred)
        grads = tape.gradient(loss, model.variables)
        grads_and_vars = zip(grads, model.variables)
        optimizer.apply_gradients(grads_and_vars)
        print("\r Epoch", epoch, " train mes:", metric.result().numpy(), end=" ")
    y_valid_pred = model(x_valid_scaled)
    valid_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_valid_pred, y_valid))
    print("\t", "valid mes: ", valid_loss.numpy())



# history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid), epochs=100, callbacks=callbacks)
#
# def plot_learning_curves(history):
#     pd.DataFrame(history.history).plot(figsize=(8,5))
#     plt.grid(True)
#     plt.gca().set_ylim(0, 1)
#     plt.show()
#
# plot_learning_curves(history)
#
# model.evaluate(x_test_scaled, y_test)
