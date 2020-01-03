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
features = tf.train.Features(
    feature = {
        'favorite_books': tf.train.Feature(bytes_list = favorite_books_bytelist)
    }
)
example = tf.train.Example(features=features)
print(example)

serialized_example = example.SerializeToString()
print(serialized_example)

output_dir = 'tfrecord_basic'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = 'test.tfrecords'
filename_fullpath= os.path.join(output_dir,filename)
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)


dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    print(serialized_example_tensor)

expected_features =  {
    'favorite_books': tf.io.VarLenFeature(dtype=tf.string)
}

dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))

filename_fullpath_zip= filename_fullpath + '.zip'
options = tf.io.TFRecordOptions(compression_type='GZIP')
# write zip file
with tf.io.TFRecordWriter(filename_fullpath_zip, options) as writer:
    for i in range(3):
        writer.write(serialized_example)


# read zip file
dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip], compression_type='GZIP')
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))