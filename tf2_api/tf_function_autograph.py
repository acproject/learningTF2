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

def scaled_elu(z, scale=1.0, alpha=1.0):
    # z >=0 ? scale * z : scale * alpha * tf.nn.elu(z)
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))

print(scaled_elu(tf.constant(-3.0)))
print(scaled_elu(tf.constant([-3, 2.5])))

scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))

@tf.function
def converge_to_2(n_iter):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iter):
        total += increment
        increment /= 2.0

    return total

def display_tf_code(func):
    code = tf.autograph.to_code(func)
    from IPython.display import display, Markdown
    display(Markdown('```pyhton\n{}\n```'.format(code)))

display_tf_code(scaled_elu)

@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)

try:
    print(cube(tf.constant([1.0, 2., 3.])))
except ValueError as ex:
    print(ex)

print(cube(tf.constant([1, 2, 3])))


# tf.function py func -> tf_graph
# get_concrete_function -> add input signature - > saveModel
cube_function_int32 = cube.get_concrete_function(tf.TensorSpec([None], tf.int32))
print(cube_function_int32 is not cube)
print(cube_function_int32.graph.get_operations())

pow_op = cube_function_int32.graph.get_operations()[2]
# print(pow_op)
print(list(cube_function_int32.inputs))
print(list(cube_function_int32.outputs))
print(cube_function_int32.graph.get_operation_by_name("x"))
print(cube_function_int32.graph.get_tensor_by_name("x:0"))

# 打印整个图对象
print(cube_function_int32.graph.as_graph_def())
