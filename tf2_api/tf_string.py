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
# tf.strings
ts = tf.constant("cafe发财")
print(ts)
print(tf.strings.length(ts))
print(tf.strings.length(ts, unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(ts, "UTF8"))

# string array
tss = tf.constant(["cafe","发财"])
print(tf.strings.length(tss))
print(tf.strings.length(tss, unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(tss, "UTF8"))

# ragged tensor
raggedTensor = tf.ragged.constant([[11, 11], [21, 22, 23], [33, 34, 45, 78]])
print(raggedTensor)
print(raggedTensor[1])
print(raggedTensor[1:2])

# ops on ragged tensor
r2 = tf.ragged.constant([[55,66], [77, 88], []])
print(tf.concat([raggedTensor, r2], axis=0))
print(tf.concat([raggedTensor, r2], axis=1))

# Ragged tensor to normal tensor
print(raggedTensor.to_tensor())
print(r2.to_tensor())

# sparse tensor
sp = tf.SparseTensor(indices=[[0,1],[1,0],[2,3]], values=[1, 2, 3], dense_shape=[3, 4])
# 注意：上面的indices需要事先排序，如果忘记排序可以调用下面的方法进行自动排序
# sp_order = tf.sparse.reorder(sp)
# print(sp_order)

print(sp)
print(tf.sparse.to_dense(sp))

# ops on sparse tensors
sp2 = sp * 2
print(sp2)

try:
    sp3 = sp + 1
except TypeError as ex:
    print(ex)

sp4 = tf.constant([[10, 32], [44, 7], [9, 5], [8, 5]])
print(tf.sparse.sparse_dense_matmul(sp, sp4))