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
"""SPN utils.

This library contains the various util functions used in SPN.
"""

import time

import tensorflow as tf

from spn import spn_plotting

def time_it(f_x, f_y, num_reps=1, execute_once_before=False):
    """Times a tensorflow function in eager mode.
    
    Args:
        f: function with no arguments that should be timed
        num_reps: int, number of repetitions for timing
        excute_once_before: oolean, whether to excute the function opce before timing in order
            to not count the tf.function compile time

    Returns:
        tuple of the average tiume in ms and the functions otput.
    """

    assert num_reps >= 1
    # Execute f once before timing it to allow tf.function to compile the graph.
    if execute_once_before:
        x = f_x()
        y = f_y()
    # Make sure that there is nothing still running on the GPU by waiting for the
    # completion of a bogus command.
    _ = tf.square(tf.random.uniform([1])).numpy()
    # Time f for a number of repetitions.
    start_in_s = time.time()
    for _ in range(num_reps):
        x = f_x()
        y = f_y()
        # Make sure that f has finished and was not just enqueued by using another
        # bogus command. This will overestimate the computing time of f by waiting
        # until the result has been copied to main memory. Calling reduce_sum
        # reduces that overestimation.
    if isinstance(x, tuple) or isinstance(x, list):
        _ = [tf.reduce_sum(input_tensor=xi).numpy() for xi in x]
    else:
        _ = tf.reduce_sum(input_tensor=x).numpy()
    end_in_s = time.time()
    # Compute the average time in ms.
    avg_time = (end_in_s - start_in_s) * 1000. / float(num_reps)
    
    return avg_time, x, y
