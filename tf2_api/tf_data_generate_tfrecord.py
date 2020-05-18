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
for moudule in mpl, np, pd, sklearn, tf, keras:
    print(moudule.__name__, moudule.__version__)

source_dir = './generate_csv/'
print(os.listdir(source_dir))

def get_filename_by_perfix(source_dir, prefix_name):
    all_files = os.listdir(source_dir)
    results = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            results.append(os.path.join(source_dir, filename))
    return results

train_filenames = get_filename_by_perfix(source_dir, "train")
valid_filenames = get_filename_by_perfix(source_dir, "valid")
test_filenames = get_filename_by_perfix(source_dir, "test")
import pprint
pprint.pprint(train_filenames)
pprint.pprint(valid_filenames)
pprint.pprint(test_filenames)

import tf2_api.tf_data_read_csv as tf_rd_csv
batch_size = 32
train_set =tf_rd_csv.csv_reader_dataset(train_filenames, batch_size)
valid_set =tf_rd_csv.csv_reader_dataset(valid_filenames, batch_size)
test_set =tf_rd_csv.csv_reader_dataset(test_filenames, batch_size)

def serialize_example(x ,y):
    '''Converts x, y to tf.train.Example and serialize'''
    input_features = tf.train.FloatList(value=x)
    label = tf.train.FloatList(value=y)
    features = tf.train.Features(
        feature={
            "input_features": tf.train.Feature(float_list=input_features),
            "label": tf.train.Feature(float_list=label)
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()

def csv_dataset_to_tfrecords(base_filename, dataset, n_shards,
                             step_per_shard, compression_type=None):
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    all_filenames = []
    for shard_id in range(n_shards):
        filename_fullpath = '{}_{:05d}-of-{:05d}'.format(
            base_filename,shard_id, n_shards
        )
        with tf.io.TFRecordWriter(filename_fullpath, options) as writer:
            for x_batch, y_batch in dataset.take(step_per_shard):
                for x_example, y_example in zip(x_batch, y_batch):
                    writer.write(serialize_example(x_example, y_example))
        all_filenames.append(filename_fullpath)
    return all_filenames
# 生成20个文件
n_shards = 20
train_steps_per_shard = 11610 // batch_size // n_shards
valid_steps_per_shard = 3880 // batch_size // n_shards
test_steps_per_shard = 5170 // batch_size // n_shards

output_dir = 'generate_tfrecords'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_basename = os.path.join(output_dir, 'train')
valid_basename = os.path.join(output_dir, 'valid')
test_basename = os.path.join(output_dir, 'test')

train_tfrecord_filenames = csv_dataset_to_tfrecords(
    train_basename, train_set, n_shards, train_steps_per_shard, None
)

vaild_tfrecord_filenames = csv_dataset_to_tfrecords(
    valid_basename, valid_set, n_shards, valid_steps_per_shard, None
)

test_trrecord_filenames = csv_dataset_to_tfrecords(
    test_basename, valid_set, n_shards, test_steps_per_shard, None
)
