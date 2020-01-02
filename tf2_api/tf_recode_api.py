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

# tfrecord 文件格式
# tf.train.Example
    # tf.train.Features - > {"key":tf.train.Feature}
     # -> tr.train.Feature -> tf.train.ByteList/FloatList/Int64List....


favorite_books = [name.encode('utf-8') for name in ["machine learning", "html5"]]
favorite_books_bytelist = tf.train.BytesList(value=favorite_books)
print(favorite_books_bytelist)