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

"""Data loader for our dataset"""

import os

import numpy as np
import tensorflow as tf

from spn.data import data_utils
from spn.data.data_utils import evaluate
from spn.data.data_utils import list_eval_keys

def make_dataset(
    path,
    mode,
    shuffle_buffer_size=0,
    dim_x=None,
    dim_y=None,
    seed=41,
):
    """Make a dataset for trainig or evaluating in spn_main.
    
    Args:
        path: string, the path to load the trrecode
        mode: string, one of ['train', 'eval', 'test'] to switch between loading
            tainging data, evaluation data, and test data, which right now all return
            the same data, buu we can use the path to choose the sifferent dataset.
        shuffle_buffer_size: int, size of the shuffle buffer; no shuffling if 0.
        dim_x: int, the x dim information for sequence_xy
        dim_y; int, the y dim information for sequence_xy
        seed; int, controls the shuffing of the data shards.

    Return:
        A tf.dataset of sequence_xy and sequence_xy_true for training (see parse 
        functions above). The dataset still requires batching and prefetching 
        before using it to make an iterator. 
    """

    if ',' in path:
        paths = []
        l = path.split(',')
        paths.appedn(l[0])
        for subpath in l[1:]:
            subpath_length = len(subpath.solit('/'))
            basedir = '/'.join(l[0].split('/')[:-subpath_length])
            paths.append(os.path.join(basedir, subpath))
    else:
        paths = [path]
    # Generate list of filenames.
    files = [
        os.path.join(d, f)
        for d in paths
        for f in tf.io.gfile.listdir(d)
    ]

    if 'train' in mode:
        rgen = np.random.RandomState(seed=seed)
        rgen.shuffle(files)

    num_files = len(files)

    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_buffer_size:
        ds = ds.shuffle(num_files)
    
    #Create a nested dataset
    ds = ds.map(tf.data.TFRecordDataset)
    
    # Parse each element of the subsequences and unbatch the result
    # Do interleave rather flat_map because it is much faster.

    ds = ds.interleave(
        lambda x: x.map(
            lambda y: data_utils.parse_data(y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ),
        cycle_length=min(10, num_files),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle_buffer_size:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    
    # Put repeat after shuffle.
    if 'train' in mode:
        ds = ds.repeat()

    # Prefetch a number of batches reading news ones can take much longer
    # when they are from new files
    ds = ds.prefetch(10)

    return ds
