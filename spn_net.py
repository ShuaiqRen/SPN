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
"""SPN: shape prediction network.

This library provides a simple interface for training and inference.
"""

import math
import sys
import time

import gin
from absl import app

import tensorflow as tf
import collections
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

from spn import spn_flags

FLAGS = tf.compat.v1.app.flags.FLAGS

class SPN_model(Model):
    def __init__(self,
                 batch_size = FLAGS.batch_size,
                 optimizer = 'adam',
                 checkpoint_dir = '',
                 checkpoint_dir_best = '',
                 summary_dir = '',
                 learning_rate = 0.005,
                 ):
        super(SPN_model, self).__init__()
        self._batch_size = batch_size
        self._dtype_policy = tf.keras.mixed_precision.experimental.Policy('float32')
        self._leaky_relu_alpha = 0.1

        self._optimizer_type = optimizer
        self._learning_rate = learning_rate
        self._make_or_reset_optimizer()

        # set up checkpointing.
        self._make_or_reset_checkpoint()
        self.update_checkpoint_dir(checkpoint_dir)
        self.update_checkpoint_dir_best(checkpoint_dir_best)

        #Set up tensorboard log files.
        self.summary_dir = summary_dir
        if summary_dir:
            self.writer = tf.compat.v1.summary.create_file_writer(summary_dir)
            self.writer.set_as_default()

        self.lstm = LSTM(1, return_sequences=True)

        self.dense_1 = Dense(256)
        self.dense_2 = Dense(24)

        self.activity = LeakyReLU(alpha=self._leaky_relu_alpha, dtype=self._dtype_policy)

        self.batchnormalization = BatchNormalization()

        # self.convs_1 = []
        # self.convs_1.append(BatchNormalization())
        # self.convs_1.append(LSTM(1, return_sequences=True))
        # self.convs_1.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_1.append(Dense(256))
        # self.convs_1.append(BatchNormalization())
        # self.convs_1.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_1.append(Dense(1))


        # self.convs_2 = []
        # self.convs_2.append(BatchNormalization())
        # self.convs_2.append(Dense(1))
        # self.convs_2.append(BatchNormalization())
        # self.convs_2.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))

        # self.convs_3 = []
        # self.convs_3.append(LSTM(1, return_sequences=True))
        # # self.convs_3.append(LSTM(1))
        # self.convs_3.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_3.append(Dense(256))
        # self.convs_3.append(BatchNormalization())
        # self.convs_3.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_3.append(Dense(24))

        # self.convs_4 = []
        # self.convs_4.append(Conv2D(
        #     32, kernel_size=(3, 3), strides=1,
        #     padding='same', dtype=self._dtype_policy))
        # self.convs_4.append(BatchNormalization())
        # self.convs_4.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_4.append(Conv2D(
        #     64, kernel_size=(3, 3), strides=1,
        #     padding='same', dtype=self._dtype_policy))
        # self.convs_4.append(BatchNormalization())
        # self.convs_4.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_4.append(Conv2D(
        #     128, kernel_size=(3, 3), strides=1,
        #     padding='same', dtype=self._dtype_policy))
        # self.convs_4.append(BatchNormalization())
        # self.convs_4.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))        
        # self.convs_4.append(Conv2D(
        #     256, kernel_size=(3, 3), strides=1,
        #     padding='same', dtype=self._dtype_policy))
        # self.convs_4.append(BatchNormalization())
        # self.convs_4.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_4.append(Conv2D(
        #     512, kernel_size=(3, 3), strides=1,
        #     padding='same', dtype=self._dtype_policy))
        # self.convs_4.append(BatchNormalization())
        # self.convs_4.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_4.append(tf.keras.layers.Dropout(0.5))
        # self.convs_4.append(Conv2D(
        #     1, kernel_size=(3, 3), strides=1,
        #     padding='same', dtype=self._dtype_policy))
        # self.convs_4.append(BatchNormalization())
        # self.convs_4.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))

        self.convs_5 = []
        self.convs_5.append(tf.keras.layers.LSTM(128, return_sequences=True))
        self.convs_5.append(tf.keras.layers.Dropout(0.5))
        self.convs_5.append(tf.keras.layers.LSTM(32, return_sequences=True))
        self.convs_5.append(Dense(256))
        self.convs_5.append(Dense(1))

    def update_checkpoint_dir(self, checkpoint_dir):
        """Changes the checkpoint directory for saving and restoring."""
        self._manager = tf.train.CheckpointManager(
            self._checkpoint, directory=checkpoint_dir, max_to_keep=1)

    def update_checkpoint_dir_best(self, checkpoint_dir_best):
        """Changes the checkpoint directory for saving and restoring the best model."""
        self._manager_best = tf.train.CheckpointManager(
            self._checkpoint, directory=checkpoint_dir_best, max_to_keep=1)

    def restore(self, reset_optimizer=False, reset_global_step=False):
        """Restores a saved model from a checkpoint."""
        status = self._checkpoint.restore(self._manager.latest_checkpoint)
        try: 
            status.assert_existing_objects_matched()
        except AssertionError as e:
            print('Error while attempting to restore SPN models:', e)
        if reset_optimizer:
            self._make_or_reset_optimizer()
            self._make_or_reset_checkpoint()
        if reset_global_step:
            tf.compat.v1.train.get_or_create_global_step().assign(0)

    def save(self):
        """Saves a model checkpoint."""
        self._manager.save()

    def save_best(self):
        """Saves the best model checkpoint."""
        self._manager_best.save()

    def _make_or_reset_optimizer(self):
        if self._optimizer_type == 'adam':
            self._optimizer = tf.compat.v1.train.AdamOptimizer(
                self._learning_rate, name='Optimizer')
        elif self._optimizer_type == 'sgd':
            self._optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                self._learning_rate, name='Optimizer')
        else:
            raise ValueError('Optimizer "{}" not yet implemented.'.format(
                self._optimizer_type))

    # @property
    # def optimizer(self):
    #     return self._optimizer
    
    def _make_or_reset_checkpoint(self):
        self._checkpoint = tf.train.Checkpoint(
            optimizer=self._optimizer,
            model=self,
            optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    
        
    def call(self, sequence_xy):

        # model_1
        # for layer in self.convs_1:
        #     sequence_xy = layer(sequence_xy)

        # model_2
        # for layer in self.convs_2:
        #     sequence_xy = layer(sequence_xy)

        # sequence_xy = tf.squeeze(sequence_xy, axis=-1)

        # for layer in self.convs_3:
        #     sequence_xy = layer(sequence_xy)

        # model_1
        for layer in self.convs_5:
            sequence_xy = layer(sequence_xy)

        sequence_xy = tf.squeeze(sequence_xy, axis=-1)

        return sequence_xy


def compute_loss(model, sequence_xy, sequence_xy_true):
    # print(sequence_xy) # 64*4*24
    # print(sequence_xy_true) # 64*24
    sequence_xy_fake = model(sequence_xy)
    # print(sequence_xy_fake)
    # loss = tf.reduce_mean(tf.math.square(sequence_xy_fake - sequence_xy_true))/ 2.0
    loss = 0.
    
    for num in range(24):
        # loss += (sequence_xy_fake[:, num] - sequence_xy_true[:, num])**2 / (num+1) 
        # loss += (sequence_xy_fake[:, num] - sequence_xy_true[:, num])**2 * ((num + 1)/24)
        loss += (sequence_xy_fake[:, num] - sequence_xy_true[:, num])**2 / 2.

    loss = tf.reduce_mean(loss)

    return loss

@tf.function
def train_step_x(spn_x, sequence_xy, sequence_x_true):
    with tf.GradientTape() as tape:
        loss_x = compute_loss(spn_x, sequence_xy, sequence_x_true)

        variables_x = spn_x.trainable_variables

    grad_x = tape.gradient(loss_x, variables_x)

    spn_x._optimizer.apply_gradients(
        zip(grad_x, variables_x))

    return loss_x

@tf.function
def train_step_y(spn_y, sequence_xy, sequence_y_true):
    with tf.GradientTape() as tape:
        loss_y = compute_loss(spn_y, sequence_xy, sequence_y_true)

        variables_y = spn_y.trainable_variables

    grad_y = tape.gradient(loss_y, variables_y)

    spn_y._optimizer.apply_gradients(
        zip(grad_y, variables_y))

    return loss_y

@tf.function
def train_step(spn_x, spn_y, sequence_xy, sequence_x_true, sequence_y_true):
    loss_x = train_step_x(spn_x, sequence_xy, sequence_x_true)
    loss_y = train_step_y(spn_y, sequence_xy, sequence_y_true)
    
    return loss_x, loss_y


def main(argv):
    sequence_xy_1 = tf.fill([8, 24, 4], 1.)
    sequence_xy_2 = tf.fill([8, 1, 24, 4], 1.)

    sequence_x_true = tf.fill([8, 24], 1.)
    sequence_y_true = tf.fill([8, 24], 0.)

    spn_x = SPN_model()
    spn_y = SPN_model()

    for i in range(100):
        loss_x = train_step_x(spn_x, sequence_xy_1, sequence_x_true)
        loss_y = train_step_y(spn_y, sequence_xy_1, sequence_y_true)
        print(loss_x, loss_y)
    print(spn_x(sequence_xy_1))
    print(spn_y(sequence_xy_1))

    # sequence_x = spn_x(sequence_xy_1)
    # # sequence_x = tf.squeeze(sequence_x, dim=1)
    # sequence_y = spn_y(sequence_xy_1)
    # # sequence_y = tf.squeeze(sequence_y, dim=1)

    # print('the sequence_x is :', sequence_x)
    # print('the sequence_y is :', sequence_y)

    # spn_x.build(input_shape=(8, 24, 4))
    # spn_x.summary()




if __name__ == '__main__':
  app.run(main)
 



