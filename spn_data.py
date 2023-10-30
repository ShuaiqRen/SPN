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

"""Library for loading train and eval data.

    This libary contains two functions, make_train_iterator for generating a 
    traing data iterator from our dataset, and make_eval_function for creating
    an evaluation function that evaluates on our dataset.
"""

# pylint:disable=g-importing-member
from functools import partial

import tensorflow as tf

# from spn import spn_augmentation
from spn.data import our_dataset

from absl import flags
from absl import app

from spn import spn_flags
# pylint:disable=g-long-lambda

FLAGS = flags.FLAGS

def make_train_iterator(
    train_on,
    dim_x,
    dim_y,
    shuffle_buffer_size,
    batch_size,
    # apply_augmentation=True
    seed=41,
    mode='train',
):
    """Build training iterator for our dataset in train_on.
    
    Args:
        train_on: string of the path for our dataset, e.g. './data/SPN_datasets/'
        dim_X: int, the x dim information for the input sequence_xy
        dim_y: int, the y dim information for the input sequence_xy
        shufflw_buffer_size: int, size that will be used for the shuffle buffer
        bathc_size: int, batch size for thr iterator
        seed: A seed for a random number generator, controls shuffling of data
        mode: str, will be passed on to the data iterator class, Can be used to 
        specify different settings within the data iterator.

    Returen:
        A tf.data.Iterator that produces batches of input_data of shape [batch_size,
        1, dim_y, dim_x] and the output_true_data of shape [batch_size, 2, dim_x]
    """

    dataset = our_dataset.make_dataset(
        train_on,
        mode=mode,
        shuffle_buffer_size=shuffle_buffer_size,
        dim_x=dim_x,
        dim_y=dim_y,
        seed=seed,
    )

    # the data augmentation but now i don't now how to do this


    #return a function to apply ensure_shape on all the available data
    def _ensure_shapes():
        # shape of the data
        input_shape = (batch_size, dim_y, dim_x)
        output_shape = (batch_size, 2, dim_x)

        return lambda input, output: (tf.ensure_shape(input, input_shape), tf.ensure_shape(output, output_shape))

    # Perform data augmentation


    train_ds = dataset
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(1)
    train_ds = train_ds.map(_ensure_shapes())
    train_it = tf.compat.v1.data.make_one_shot_iterator(train_ds)

    return train_it

def make_eval_function(eval_on,  progress_bar, plot_dir, num_plots):
    """Bulid an evaluation function for spn.
    
    Args:
        eval_on: string, the path to load the test dataset
        progress_bar: boolean, flag to indicate whether the function should print a
            progress_bar during evaluaton.
        plot_dir: string, optional path to a directory in which plots are saved (if
            num_plots > 0).
        num_plots: int, maximum number of qualitative results to plot for the
            evaluation
    Returns :
        A pair consisting of an evaluation function and a list of strings that
            holds the keys of the evaluation result.
    """

    eval_functions_and_datastes = []
    eval_keys = []

    dataset = our_dataset.make_dataset(eval_on, mode='test')
    eval_fn = partial(
        our_dataset.evaluate,
        prefix='ours',
        max_num_evals=10000,
    )
    eval_keys += our_dataset.list_eval_keys(prefix='ours')

    dataset = dataset.prefetch(1)
    eval_functions_and_datastes.append((eval_fn, dataset))

    # Make an eval function that aggregates all evaluations.
    def eval_function(spn_x, spn_y):
        result = dict()
        for eval_fn, ds in eval_functions_and_datastes:
            results = eval_fn(spn_x, spn_y, ds, progress_bar, plot_dir, num_plots)
            for k, v in results.items():
                result[k] = v
        return result

    return eval_function, eval_keys





def main(argv):
    dataset = make_train_iterator(FLAGS.train_on, FLAGS.dim_x, FLAGS.dim_y, 0, FLAGS.batch_size)
    print('Let us test the dataset load')

    print(dataset)




if __name__ == '__main__':
  app.run(main)
 
