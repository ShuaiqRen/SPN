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

# Lint as: python3
"""
    Apply the SPN to predicte the shape using the imput array
"""

from absl import app
from absl import flags

import numpy as np

import time

import tensorflow as tf

from spn import spn_flags
from spn import spn_plotting
from spn import spn_main

FLAGS = flags.FLAGS


def forward_load_model():


    spn_x, spn_y = spn_main.create_spn()

    # 加载spn_x
    if FLAGS.init_checkpoint_x:
        print('Initializing model from checkpoint {}.'.format(FLAGS.init_checkpoint_x))
        spn_x.update_checkpoint_dir(FLAGS.init_checkpoint_x)
        spn_x.restore(
            reset_optimizer=FLAGS.reset_optimizer,
            reset_global_step=FLAGS.reset_global_step)
        spn_x.update_checkpoint_dir(FLAGS.checkpointx)
    elif FLAGS.checkpoint_x:
        # print('Rsestoring model from checkpoint {}.'.format(FLAGS.checkpoint_x))
        spn_x.restore()
    else:
        print('We need the checkpoint to predicte the x dim information for the final shape')

    # 加载spn_y

    if FLAGS.init_checkpoint_y:
        # print('Initializing model from checkpoint {}.'.format(FLAGS.init_checkpoint_y))
        spn_y.update_checkpoint_dir(FLAGS.init_checkpoint_y)
        spn_y.restore(
            reset_optimizer=FLAGS.reset_optimizer,
            reset_global_step=FLAGS.reset_global_step)
        spn_y.update_checkpoint_dir(FLAGS.checkpointx)
    elif FLAGS.checkpoint_y:
        #print('Rsestoring model from checkpoint {}.'.format(FLAGS.checkpoint_y))
        spn_y.restore()
    else:
        print('We need the checkpoint to predicte the y dim inforamtion for the final shape')

    return spn_x, spn_y

def forward(spn_x, spn_y, input):

    # input = input.T
    input = tf.convert_to_tensor(input)
    input = tf.cast(input, tf.float32)

    # print(input)
    input = tf.expand_dims(input, 0)# 1*24*4
    input = tf.transpose(input, [0, 2, 1])
    input = (input - 0.5) * 2
    # print(input)

    output_x = spn_x(input).numpy()[0]# 24
    output_y = spn_y(input).numpy()[0]# 24

    output_xy = [output_x, output_y]
    # print(output_xy[0])
    # print(output_xy[1])

    return output_xy


def main(unused_argv):

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # #
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

    if not FLAGS.plot_dir:
        raise ValueError('apply_spn need plot directory')
    if not tf.io.gfile.exists(FLAGS.plot_dir):
        print('Making new plot directory', FLAGS.plot_dir)
        tf.io.gfile.makedirs(FLAGS.plot_dir)

#     input = np.array(
#         [
# [1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0],
# [0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0],
# [1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,0,1,0,1,0,1,1,1,1,0],
# [0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,0,1,1,1]
# ])
    input = np.array(
        [
[0,0,0,0,0,0,0,0,0],
[1,1,1,1,1,1,1,1,1],
[0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0]
        ]
    )

    time_start = time.time()

    spn_x, spn_y = forward_load_model()
    output = forward(spn_x, spn_y, input)

    # for i in range(900):
    #     output = forward(spn_x, spn_y, input)
    #     print(i)

    time_end = time.time()
    print('output :')
    print(output)
    print('time :')
    print(time_end - time_start)

    # a = np.array(output).T

    # with open("pre.txt", "w") as file:
    #     for j in range(36):
    #         file.write(str(a[j, 0]))
    #         file.write('\t')
    #         file.write(str(a[j, 1]))
    #         file.write('\t')
    #         file.write('\n')

    spn_plotting.plt_output_shape(output, FLAGS.plot_dir)

if __name__ == '__main__':
    app.run(main)
    
