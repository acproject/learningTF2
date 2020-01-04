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
output_dir = 'generate_csv'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# read csv
# 1. get filename -> dataset
# 2. read file -> dataset -> datasets -> merge
# 3. pares csv
# filename_dataset = tf.data.Dataset.list_files(train_filenames)
# for filename in filename_dataset:
#     print(filename)
# n_readers = 5
# dataset = filename_dataset.interleave(
#     lambda filename: tf.data.TextLineDataset(filename).skip(1),
#     cycle_length = n_readers
# )
# for line in dataset.take(15):
#     print(line.numpy())

# tf.io.decode_csv(str, record_defaults)
sample_str = '1,2,3,4,5'
record_defaults = [tf.constant(0, dtype=tf.int32),
                   0,
                   np.nan,
                   "hello",
                   tf.constant([])
                   ]
parsed_fields = tf.io.decode_csv(sample_str, record_defaults)
print(parsed_fields)

def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y

source_dir = './generate_csv/'
def get_filename_by_perfix(source_dir, prefix_name):
    all_files = os.listdir(source_dir)
    results = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            results.append(os.path.join(source_dir, filename))
    return results
train_csv_filenames = get_filename_by_perfix(source_dir, "train")
valid_csv_filenames = get_filename_by_perfix(source_dir, "valid")
test_csv_filenames = get_filename_by_perfix(source_dir, "test")
def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32,n_parsed_thread=5, shuffle_buffer_size=1024):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parsed_thread)
    dataset = dataset.batch(batch_size)
    return dataset

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation='relu', input_shape=[8]),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(loss='mse', optimizer='sgd')
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
batch_size = 32
train_set = csv_reader_dataset(filenames=train_csv_filenames)
valid_set = csv_reader_dataset(filenames=valid_csv_filenames)
test_set = csv_reader_dataset(filenames=test_csv_filenames)
history = model.fit(train_set,
                    validation_data=valid_set,
                    steps_per_epoch=11160 // batch_size,
                    validation_steps=3870 // batch_size,
                    epochs=100,
                    callbacks=callbacks)

model.evaluate(test_set, steps=5160 // batch_size)


