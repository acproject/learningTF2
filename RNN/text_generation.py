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

# load data
input_filepath= './data.txt'
text = open(input_filepath, 'r').read()

print(len(text))
print(text[0:100])

# 1. generate vocab
# 2. build mapping char-> id
# 3. data-> id_data
# 4. 'test' - > 'est<eos>'

vocab = sorted(set(text))
print(len(vocab))
print(vocab)

char2idx = {char:idx for idx , char in enumerate(vocab)}
print(char2idx)

idx2char = np.array(vocab)
print(idx2char)

text_as_int = np.array([char2idx[c] for c in text])
print(text_as_int[0:10])
print(text[0:10])

def split_input_target(id_text):
    return id_text[0:-1], id_text[1:]

char_dateset = tf.data.Dataset.from_tensor_slices((text_as_int))
seq_length = 100
seq_dataset = char_dateset.batch(seq_length + 1, drop_remainder= True)
for ch_id in char_dateset.take(2):
    print(ch_id, idx2char[ch_id.numpy()])

for seq_id in seq_dataset.take(2):
    print(seq_id)
    print(repr(''.join(idx2char.numpy())))

seq_dataset = seq_dataset.map(split_input_target)
for item_input , item_out in seq_dataset.take(2):
    print(item_input.numpy())
    print(item_out.numpy())

batch_size = 64
buffer_size = 1024 * 100
seq_dataset = seq_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_mode(vovab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vovab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.SimpleRNN(units=rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vovab_size)
    ])
    return model;
model = build_mode(vovab_size=vocab_size, embedding_dim= embedding_dim, rnn_units=rnn_units,batch_size=batch_size)

model.summary()
for input_example_batch, target_example_batch in seq_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)

# random sampling.
# greedy, random.
sample_indices = tf.random.categorical(logits= example_batch_predictions[0], num_samples= 1)
sample_indices = tf.squeeze(sample_indices, axis=-1)
print(sample_indices)
print("Input: ", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Output: ", repr("".join(idx2char[target_example_batch[0]])))
print()
print("Predictions: ", repr("".join(idx2char[sample_indices])))
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)
example_loss = loss(target_example_batch, example_batch_predictions)
print(example_loss.shape)
print(example_loss.numpy().mean())

output_dir = "./text_generation_checkpoints"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

checkpoint_prefix = os.path.join(output_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)
epochs = 100
history = model.fit(seq_dataset, epochs = epochs, callbacks=[checkpoint_callback])

print(tf.train.latest_checkpoint(output_dir))

# laod mode
model2 = build_mode(vovab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=1)
model2.load_weights(tf.train.latest_checkpoint(output_dir))
model2.build(tf.TensorShape([1, None]))
# start char sequence A,
# A -> model -> b
# A.append(b) -> B
# B -> model -> c
# B.append(c) -> C
# C -> model -> ....
model2.summary()
def generate_text(model, start_str, num_generate = 1000):
    input_eval = [char2idx[ch] for ch in start_str]
    input_eval = tf.expand_dims[input_eval, 0]

    text_generated = []
    model.reset_states()

    for _ in range(num_generate):
        # 1. model inference -> predictions
        # 2. sample -> ch -> text_generated[]
        # 3. update input_eval

        # predictions : [batch_size, input_eval_len, vocab_size]
        predictions = model(input_eval)
        # predictions : [input_eval_len, vocab_size]
        predictions = tf.squeeze(predictions, 0)
        # prediction_id : [input_eval_len , 1]
        prediction_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        text_generated.append(idx2char[prediction_id])
        # s, x-> rnn -> s', y
        input_eval = tf.expand_dims([prediction_id], 0)
    return start_str + ''.join((text_generated))

new_text = generate_text(model2, "All: ")
print((new_text))

