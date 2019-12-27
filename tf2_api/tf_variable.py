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

# Variables
v = tf.Variable([[1,2],[3, 4], [5,6]])
print(v)
print(v.value())
print(v.numpy())

# ops
## assgin value
v.assign(2*v)
print(v.numpy())
v[0,1].assign(45)
print(v[0,1])
print(v.numpy())
v[1].assign([9,0])
print(v[1])
print(v.numpy())