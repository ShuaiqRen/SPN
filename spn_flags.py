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
"""Flags used by spn training and evaluation."""

from absl import flags

FLAGS = flags.FLAGS

# General flags.
# the parameter for debugging
flags.DEFINE_bool(
    'no_tf_function', False, 'If true, run without'
    'tf functions, This incurs a performance hit, but can'
    'make debugging easier.'
)
# the parameter for datasets
flags.DEFINE_string('train_on', '',
                    'the path to the training dataset')
flags.DEFINE_string('eval_on', '',
                    'the path to the evaluating dataset')
flags.DEFINE_integer('dim_x', 24, 'the x dim for sequence_xy')
flags.DEFINE_integer('dim_y', 4, 'the y dim for sequence_xy')
# the parameter for visualization
flags.DEFINE_string('plot_dir', '',
                    'the path to the directory where plots are saved')
flags.DEFINE_string('plot_dir_loss', '',
                    'the path to the directory where plots are saved')
# the training ade testing information 
flags.DEFINE_string('summary_dir', '', '')
# the parameter for model saving and restoring
flags.DEFINE_string('checkpoint_x','',
                    'the path to the directory for saving and restoring checkpoints of predicting the x coordinate')
flags.DEFINE_string('checkpoint_x_best', '',
                    'the path to directory for saving and restoring the best checkpoints of predicting the x coordinate'
                    )
flags.DEFINE_string('checkpoint_y','',
                    'the path to the directory for saving and restoring checkpoints of predicting the y coordinate')
flags.DEFINE_string('checkpoint_y_best', '',
                    'the path to directory for saving and restoring the best checkpoints of predicting the y coordinate'
                    )
# the parameter for model initializing
flags.DEFINE_string('init_checkpoint_x', '',
                    'the path to the directory for initializing from the checkpoint of model_x')
flags.DEFINE_string('init_checkpoint_y', '',
                    'the path to the directory for initializing from the checkpoint of model_y')

# the parameter for debugging
flags.DEFINE_bool('plot_debug_info', False,
                  'Flag to indicate whether to plot debug info during training')
# the parameter for using tensorboard
flags.DEFINE_bool('use_tensorboard', False, 'Toggles logging to tensorboard')
flags.DEFINE_string('tensorboard_logdir', '',
                  'where to log tensorboard summaries')
# the parameter for reset optimizer parameter
flags.DEFINE_bool('reset_global_stpe', True,
                  'reset global step to 0 after loading from the init_checkpoint')
flags.DEFINE_bool('reset_optimizer', True,
                  'reset optimizer internals after loading init_checkpoint')
# the parameter for training
flags.DEFINE_bool('evaluate_during_train', True,
                  'whether or not to have the GPU train job perform evaluation between epochs')
flags.DEFINE_bool('from_scratch', True,
                  'train from scratch, Do not restore the last checkpoint')
flags.DEFINE_bool('no_checkpointing', False,
                  'Do not save the model checkpoint during training')
flags.DEFINE_integer('epoch_length', 1000,
                     'number of gradient steps per epoch')
flags.DEFINE_integer('num_train_steps', 10000000,
                     'number of gradient steps to train for')
flags.DEFINE_integer('shuffle_buffer_size', 1024, 
                     'Shufflw buffer size for training')
flags.DEFINE_integer('batch_size', 64,
                     'Batch size for training on GPU ')
flags.DEFINE_string('optimizer', 'adam', 'one of the "adam" and "sgd"')
flags.DEFINE_float('gpu_learning_rate', 5e-3,
                     'learning rate for training SPN on GPU')
flags.DEFINE_integer('lr_decay_after_num_steps', 0, '')
flags.DEFINE_integer('lr_decay_steps', 15000, '') 
flags.DEFINE_string('lr_decay_type', 'sqrt-2',
                    'One of ["none", "exponential", "linear", "gaussian", "sqrt-2"]')

flags.DEFINE_multi_string(
    'config_file', None,
    'Path to a Gin config file. Can be specified multiple times. '
    'Order matters, later config files override former ones.')

flags.DEFINE_multi_string(
    'gin_bindings', None,
    'Newline separated list of Gin parameter bindings. Can be specified '
    'multiple times. Overrides config from --config_file.')

flags.DEFINE_integer('n', 24, 'the expect num for design')
flags.DEFINE_integer('length', 72, 'the expect num for design')
flags.DEFINE_integer('direction', 1,
                     'the direction of the path used by convert the img to points')
flags.DEFINE_float('point_x', 10, 'the except coordinate x for the special point')
flags.DEFINE_float('point_y', 8.5, 'the expect coordinate y for the special point')
flags.DEFINE_float('point_occlusion_x', 10, 'the except coordinate x for the special point')
flags.DEFINE_float('point_occlusion_y', 8.5, 'the expect coordinate y for the special point')
flags.DEFINE_integer('d', 10, 'the redion for the occlusion point')