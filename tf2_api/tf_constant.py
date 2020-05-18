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

t = tf.constant([[1, 2, 3], [4, 5, 6]])
# index
print(t)
print(t[:, 1:])
print(t[..., 1])
print(t[1, ...])
print(t[1, 1])

# op
print(t+10)
print(tf.square(t))
print(t @ tf.transpose(t))

# numpy conversion
print(t.numpy())
print(np.square(t))
np_t = np.array([[1,2,3],[4,5,6]])
print(tf.constant(np_t))

# scalars
t2 =tf.constant(2.5678)
print(t2.numpy())
print(t2.shape)
