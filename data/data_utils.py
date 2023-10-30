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
"""Data loading and evaluation utilities for our_dataset."""

from collections import defaultdict
import sys
import time

import numpy as np
import tensorflow as tf

from spn import spn_plotting
from spn import spn_utils

def parse_data(proto):
    """Parse a data proto with the GT.
    
    Args:
        proto: path to data proto file
        dim_x: the x dim information for sequence_xy
        dim_y: the y dim infoemation for sequence_xy

    Return:
        sequence_xy [1, dim_y, dim_x], sequence_xy_true[2, dim_x] 
    """

    # Pares context from protobuffer
    context_features = {
        'input_shape_0': tf.io.FixedLenFeature([], tf.int64),
        'input_shape_1': tf.io.FixedLenFeature([], tf.int64),
        'output_shape_0': tf.io.FixedLenFeature([], tf.int64),
        'output_shape_1': tf.io.FixedLenFeature([], tf.int64),
        'input_data': tf.io.FixedLenFeature([], tf.string),
        'output_data': tf.io.FixedLenFeature([], tf.string)
    }

    context_parsed, _ =  tf.io.parse_single_sequence_example(
        proto,
        context_features=context_features,
    )

    def deserialize(s, dtype):
        return tf.io.decode_raw(s, dtype)


    input_data = deserialize(context_parsed['input_data'], tf.float32)
    input_data = tf.reshape(input_data, [context_parsed['input_shape_0'], context_parsed['input_shape_1']])

    input_data = (input_data - 0.5) * 2

    output_data = deserialize(context_parsed['output_data'], tf.float32)
    output_data = tf.reshape(output_data, [context_parsed['output_shape_0'], context_parsed['output_shape_1']])

    output = [input_data, output_data] 

    # Only put the output in a list if there more than one items in there
    if len(output) == 1:
        output = output[0]
    
    return output


def evaluate(
    inference_fn_x,
    inference_fn_y,
    dataset,
    progress_bar=False,
    plot_dir='',
    num_plots=0,
    max_num_evals=10000,
    prefix='',
    ):
    """Evaluate an inference function for flow.
        
    Args:
        inference_fn: An inference function that produces a sequence_x and sequence_y from the sequence_xy_input
        dataset: A dataset produced by method above with for_eval_True
        progress_bar: boolean, flag to incicate whether te function should print a progress_bar during evaluation.
        plot_dir: string, optional path to a directory in which plots are seved(if num_plots > 0)
        num_plots: int, maximun number of qualitative results to plot for evaluations
        max_num_evals: int, maxmim number of evaluations.
        prefix: str, prefix to prepend to filenames for saved plots and for keys in
            results dictionary.

    Returns:
        A dictionary of floats that represent different evaluation metrics. The keys of dictionary are returned 
        by the method list_eval_keys (see below).
    """

    eval_start_in_s = time.time()

    it = tf.compat.v1.data.make_one_shot_iterator(dataset)
    x_point_errors = []
    y_point_errors = []
    x_y_points_mean_errors = []
    inference_times = []

    plot_count = 0
    eval_count = -1

    pre_values = []
    truth_values = []
    num = 0

    for test_batch in it:

        num = num + 1

        if eval_count >= max_num_evals:
            break
            
        eval_count += 1
        if eval_count >= max_num_evals:
            break

        if progress_bar:
            sys.stdout.write(':')
            sys.stdout.flush()

        (sequence_xy, sequence_xy_true) = test_batch


        start_time = time.time()

        # print(sequence_xy)

        sequence_xy = tf.transpose(sequence_xy, [1, 0])
        sequence_xy = tf.expand_dims(sequence_xy, 0)
        # print(sequence_xy)        

        f_x = inference_fn_x(sequence_xy)
        f_y = inference_fn_y(sequence_xy)

        
        pre_values.append(f_x.numpy().T)
        pre_values.append(f_y.numpy().T)
        truth_values.append(sequence_xy_true.numpy()[0].T)
        truth_values.append(sequence_xy_true.numpy()[1].T)

        f_x = tf.squeeze(f_x, axis=0)
        f_y = tf.squeeze(f_y, axis=0)
      

        end_time = time.time()

        inference_time = end_time - start_time
        inference_times.append(inference_time)

        # print(f_x)
        # print(sequence_xy_true)
        # print(sequence_xy_true[0])

        x_point_error = tf.reduce_mean(tf.math.square(f_x - sequence_xy_true[0]))
        y_point_error = tf.reduce_mean(tf.math.square(f_y - sequence_xy_true[1]))
        x_y_points_mean_error = tf.reduce_mean(x_point_error + y_point_error)

        x_point_errors.append(x_point_error)
        y_point_errors.append(y_point_error)
        x_y_points_mean_errors.append(x_y_points_mean_error)

        if plot_dir and plot_count < num_plots:
            plot_count += 1
            spn_plotting.complete_paper_plot(
                plot_dir,
                plot_count,
                sequence_xy.numpy(),
                sequence_xy_true.numpy(),
                f_x.numpy(),
                f_y.numpy()
            )
    
    # save the pre 
    random_choose = [617, 306, 474, 26, 181, 565, 530, 496, 103, 661, 424, 630]
    pre_values = np.array(pre_values)
    with open("pre_values.txt", "w") as file:
        for j in range(24):
            for i in random_choose:
                file.write(str(pre_values[i*2, j][0]))
                file.write('\t')
                file.write(str(pre_values[i*2 +1, j][0]))
                file.write('\t')
            file.write('\n')

    truth_values = np.array(truth_values)
    with open("truth_values.txt", "w") as file:
        for j in range(24):
            for i in random_choose:
                file.write(str(truth_values[i*2, j]))
                file.write('\t')
                file.write(str(truth_values[i*2 + 1, j]))
                file.write('\t')
            file.write('\n')

    if progress_bar:
        sys.stdout.write('\n')
        sys.stdout.flush()

    eval_stop_in_s = time.time()

    results = {
        'x_point_error': np.mean(np.array(x_point_errors)),
        'y_point_error': np.mean(np.array(y_point_errors)),
        'x_y_points_mean_error': np.mean(np.array(x_y_points_mean_errors)),
        'inf-time': np.mean(inference_times),
        'eval-time': eval_stop_in_s - eval_start_in_s
    }

    if prefix:
        return {prefix + '-' + k: v for k, v in results.items()}

    return results

def list_eval_keys(prefix=''):
    """List the keys of the dictionary returned by the evaluate function."""
    keys = [
        'x_point_error', 'y_point_error', 'x_y_points_mean_error', 
        'inf-time', 'eval-time'
    ]

    if prefix:
        return [prefix + '-' + k for k in keys]
    return keys
    


