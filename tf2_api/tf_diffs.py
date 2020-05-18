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
# import paddle.fluid
# paddle.fluid.install_check.run_check()

'''
测试版本信息与模块名字
'''
print(tf.__version__)
print(sys.version_info)
for moudule in mpl, np , pd , sklearn, tf, keras:
    print(moudule.__name__, moudule.__version__)
def fx(x):
    return 3.*x**2 + 2.*x -1


def approximae_derivative(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2. * eps)

print(approximae_derivative(fx, 1.))

def gh(x1, x2):
    return (x1 + 5) * (x2 ** 2)

def approximae_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximae_derivative(lambda x: g(x, x2), x1, eps)
    dg_x2 = approximae_derivative(lambda x: g(x1, x), x2 ,eps)
    return dg_x1, dg_x2
print(approximae_gradient(gh, 2.,  3.))

x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    z = gh(x1, x2)

# dz_x1 = tape.gradient(z, x1)
# print(dz_x1)
# dz_x2 = tape.gradient(z, x2)
# print(dz_x2)

dz = tape.gradient(z, [x1, x2])
print(dz)
# 因为使用了persistent=True，所以这里要手动增加del去释放资源
# 如果没有persistent=True，tf将为我们自动释放资源，所以一旦调用了tape就会自动释放
del tape

x = tf.Variable(5.0)
with tf.GradientTape() as tape2:
    z1 =3*x
    z2 =x**2
print(tape2.gradient([z1,z2], x))

with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = gh(x1, x2)
    inner_grads = inner_tape.gradient(z,[x1, x2])
outer_grads = [outer_tape.gradient(inner_grads, [x1, x2])
                    for inner_grad in inner_grads]
print(outer_grads[0][0].numpy())
print(outer_grads)
del inner_tape
del outer_tape

lr = 0.1
x = tf.Variable(0.0)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = fx(x)
    dz_x = tape.gradient(z, x)
    x.assign_sub(lr * dz_x)
print(x)
print(x.numpy())



lr = 0.1
x = tf.Variable(0.0)
optimzer = tf.keras.optimizers.SGD(lr=lr)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = fx(x)
    dz_x = tape.gradient(z, x)
    optimzer.apply_gradients([(dz_x, x)])
print(x)
print(x.numpy())

