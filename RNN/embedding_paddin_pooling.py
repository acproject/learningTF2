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

imdb = tf.keras.datasets.imdb
vocab_size = 10000
index_from = 3
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size, index_from=index_from)
print(train_data[0], train_labels[0])
print(train_data.shape, train_labels.shape)
print(len(train_data[0]), len(train_data[1]))

print(test_data.shape, test_labels.shape)

word_index = imdb.get_word_index()
print(len(word_index))
print(word_index)

word_index = {k: (v + 3) for k, v in word_index.items()}

word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<END>'] = 3

reverse_word_index = dict([(value, key) for key, value in word_index.items()])


def decode_review(text_ids):
    return ' '.join([reverse_word_index.get(word_id, '<UNK>') for word_id in text_ids])


print(decode_review(train_data[0]))

max_length = 500

train_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index['<PAD>'],
    padding='post',  # pre | post
    maxlen=max_length
)

test_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index['<PAD>'],
    padding='post',  # pre | post
    maxlen=max_length
)

print(train_data[0])

embedding_dim = 16
batch_size = 128

model = tf.keras.models.Sequential([
    # 1. define matrix : [vocab_size , embedding_dim]
    # 2. [1,2,3...]. max_length * embedding_dim
    # 3 .batch_size * max_length * embedding_dim
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # <input> batch_size * max_lenght * embedding_dim ->  batch_size * embedding_dim
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=30, batch_size=batch_size,
                    validation_split=0.2)


def plot_learning_curves(histroy, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


plot_learning_curves(history, 'accuracy', 30, 0, 1)
plot_learning_curves(history, 'loss', 30, 0, 1)

model.evaluate(test_data, test_labels, batch_size=batch_size)
