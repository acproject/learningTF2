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

expected_features = {
    "input_features": tf.io.FixedLenFeature([8], dtype=tf.float32),
    "label": tf.io.FixedLenFeature([1], dtype=tf.float32)
}

def parse(serialized_example):
    example = tf.io.parse_single_example(serialized_example, expected_features)
    return example["input_features"], example["label"]

def tfrecords_reader_dataset(filenames, n_readers=5,
                       batch_size=32,n_parsed_thread=5, shuffle_buffer_size=1024, compression_type=None):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type=compression_type),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse, num_parallel_calls=n_parsed_thread)
    dataset = dataset.batch(batch_size)
    return dataset
source_dir = './generate_tfrecords/'

def get_filename_by_perfix(source_dir, prefix_name):
    all_files = os.listdir(source_dir)
    results = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            results.append(os.path.join(source_dir, filename))
    return results

train_tfrecord_filenames = get_filename_by_perfix(source_dir, "train")
valid_tfrecord_filenames = get_filename_by_perfix(source_dir, "valid")
test_tfrecord_filenames = get_filename_by_perfix(source_dir, "test")
tfrecords_train = tfrecords_reader_dataset(train_tfrecord_filenames, batch_size=3)

for x_batch , y_batch in tfrecords_train.take(2):
    print(x_batch)
    print(y_batch)

batch_size = 32
tfrecords_train_set = tfrecords_reader_dataset(train_tfrecord_filenames, batch_size=batch_size)
tfrecords_valid_set = tfrecords_reader_dataset(valid_tfrecord_filenames, batch_size=batch_size)
tfrecords_test_set = tfrecords_reader_dataset(test_tfrecord_filenames, batch_size=batch_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation='relu', input_shape=[8]),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(loss='mse', optimizer='sgd')
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
batch_size = 32

history = model.fit(tfrecords_train_set,
                    validation_data=tfrecords_valid_set,
                    steps_per_epoch=11160 // batch_size,
                    validation_steps=3870 // batch_size,
                    epochs=100,
                    callbacks=callbacks)

model.evaluate(tfrecords_test_set, steps=5160 // batch_size)

