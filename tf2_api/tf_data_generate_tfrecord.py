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


