# coding=utf-8
# Copyright 2023 .
# Author : Shuaiqi Ren.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as : python3
"""Shared utils for converting datasets"""

import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

from spn import spn_flags


FLAGS = flags.FLAGS

def generate_sharded_filenames(filename):
    name, num_shardes = filename.split('@')
    filenames = []
    for num in range(int(num_shardes)):
        filenames.append(name + f'@{num}.tfrecord')
    return filenames

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def read_input(filepath, dim_x, dim_y):
    input_sequence_xy = np.zeros((dim_y, dim_x), dtype=np.float32)
    with open(filepath, mode='r') as f:
        inputs = f.readlines()
        row = 0
        for input in inputs:
            input_sequence_xy[row] = input.split('\n')[0].split('\t')[0:dim_x]
            row += 1
        # input_sequence_xy = input_sequence_xy.astype(np.int64)
    return input_sequence_xy

def read_output(filepath, dim_x, dim_y=2):
    output_sequence_xy_true = np.zeros((dim_x, dim_y), dtype=np.float32)
    with open(filepath, 'r') as f:
        outputs = f.readlines()
        row = 0
        for output in outputs:
            if row == 0:
                row += 1
            else:
                output =output.split('\n')[0].split(' ')
                output1 = list(filter(str.strip, output))
                # print(output_sequence_xy_true[row - 1])
                # print(output1[3])
                # print(output1[4])
                output_sequence_xy_true[row - 1] = [output1[3], output1[4]]
                row += 1
        return output_sequence_xy_true.T


def main(argv):
    input = read_input('./spn/spn_dataset/Model_2_NxNy_log.txt', FLAGS.dim_x, FLAGS.dim_y)
    print(input)
    print(input.shape)
    print(input.dtype)

    output = read_output('./spn/spn_dataset/Model_2_1D_Nx_outputlog.txt', FLAGS.dim_x)
    print(np.array(output))
    print(output.shape)
    print(output.dtype)




if __name__ == '__main__':
  app.run(main)
 