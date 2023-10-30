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

"""Main script to train and evaluate SPN."""

# pylint:disable=g-importing-member
from functools import partial
from absl import app
from absl import flags

import tensorflow as tf
import sys
import gin
import time
import numpy as np

# from spn import spn_sugmentation
from spn import spn_data
from spn import spn_flags
from spn import spn_plotting
from spn import spn_net
from spn.spn_net import SPN_model

# import warnings
# ...
# warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码

FLAGS = flags.FLAGS

# create two models
def create_spn():
    # Define learning rate schedules [none, cosine, linear, expoential]
    def learning_rate_fn():
        step = tf.compat.v1.train.get_or_create_global_step()
        effective_step = tf.maximum(step - FLAGS.lr_decay_after_num_steps + 1, 0)
        lr_step_ratio = tf.cast(effective_step, 'float32') / float(
            FLAGS.lr_decay_steps)
        if FLAGS.lr_decay_type == 'none' or FLAGS.lr_decay_steps <= 0:
            return FLAGS.gpu_learning_rate
        elif FLAGS.lr_decay_type == 'cosine':
            x = np.pi * tf.minimum(lr_step_ratio, 1.0)
            return FLAGS.gpu_learning_rate * (tf.cos(x) + 1.0) / 2.0
        elif FLAGS.lr_decay_type == 'linear':
            return FLAGS.gpu_learning_rate * tf.maximum(1.0 - lr_step_ratio, 0.0)
        elif FLAGS.lr_decay_type == 'exponential':
            return FLAGS.gpu_learning_rate * 0.5**lr_step_ratio
        elif FLAGS.lr_decay_type == 'sqrt-2':
            return FLAGS.gpu_learning_rate * ((1/2)**0.5)**lr_step_ratio 
        else:
            raise ValueError('Unknown lr_decay_type', FLAGS.lr_decay_type)

    spn_x = SPN_model(
        checkpoint_dir = FLAGS.checkpoint_x,
        checkpoint_dir_best = FLAGS.checkpoint_x_best,
        optimizer = FLAGS.optimizer,
        learning_rate = learning_rate_fn,
    )

    spn_y = SPN_model(
        checkpoint_dir = FLAGS.checkpoint_y,
        checkpoint_dir_best = FLAGS.checkpoint_y_best,
        optimizer = FLAGS.optimizer,
        learning_rate = learning_rate_fn,
    )

    return spn_x, spn_y

def main(unused_argv):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    #
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])

    # 是否使用TF function运行代码，如果不使用，则会损失运行速度但是容易debug。
    if FLAGS.no_tf_function:
        tf.config.experimental_run_functions_eagerly(True)
        print("TFFUNCTION DISABLED")

    # 配置环境文件，基本没用，因为我是外部输入GPU使用情况的
    gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
    
    # 创建checkpoint的保存文件夹
    if FLAGS.checkpoint_x and not tf.io.gfile.exists(FLAGS.checkpoint_x):
        print("making new checkppint directory")
        tf.io.gfile.makedirs(FLAGS.checkpoint_x)
    if FLAGS.checkpoint_x_best and not tf.io.gfile.exists(FLAGS.checkpoint_x_best):
        print("making new checkppint directory")
        tf.io.gfile.makedirs(FLAGS.checkpoint_x_best)
    if FLAGS.checkpoint_y and not tf.io.gfile.exists(FLAGS.checkpoint_y):
        print("making new checkppint directory")
        tf.io.gfile.makedirs(FLAGS.checkpoint_y)
    if FLAGS.checkpoint_y_best and not tf.io.gfile.exists(FLAGS.checkpoint_y_best):
        print("making new checkppint directory")
        tf.io.gfile.makedirs(FLAGS.checkpoint_y_best)

    if FLAGS.plot_dir and not tf.io.gfile.exists(FLAGS.plot_dir):
        print('Making new plot directory', FLAGS.plot_dir)
        tf.io.gfile.makedirs(FLAGS.plot_dir)

    if FLAGS.plot_dir_loss and not tf.io.gfile.exists(FLAGS.plot_dir_loss):
        print('Making new plot directory', FLAGS.plot_dir_loss)
        tf.io.gfile.makedirs(FLAGS.plot_dir_loss)


    # 构建模型
    spn_x, spn_y = create_spn()

    # 加载模型或者重新开始运算。
    # 加载spn_x
    if not FLAGS.from_scratch:
    # first restore from init_checkpoint_dir, which is only restoreed from but
    # not saved to, and then restore from checkpoint_dir if there is already 
    # a model there (e.g. if the run was stopped and restored).
        if FLAGS.init_checkpoint_x:
            print('Initializing model from checkpoint {}.'.format(FLAGS.init_checkpoint_x))
            spn_x.update_checkpoint_dir(FLAGS.init_checkpoint_x)
            spn_x.restore(
                reset_optimizer=FLAGS.reset_optimizer,
                reset_global_step=FLAGS.reset_global_step)
            spn_x.update_checkpoint_dir(FLAGS.checkpoint_x)
        elif FLAGS.checkpoint_x:
            print('Rsestoring model from checkpoint {}.'.format(FLAGS.checkpoint_x))
            spn_x.restore()
    else:
        print('Starting from scratch')

    # 加载spn_y
    if not FLAGS.from_scratch:

        if FLAGS.init_checkpoint_y:
            print('Initializing model from checkpoint {}.'.format(FLAGS.init_checkpoint_y))
            spn_y.update_checkpoint_dir(FLAGS.init_checkpoint_y)
            spn_y.restore(
                reset_optimizer=FLAGS.reset_optimizer,
                reset_global_step=FLAGS.reset_global_step)
            spn_y.update_checkpoint_dir(FLAGS.checkpoint_y)
        elif FLAGS.checkpoint_y:
            print('Rsestoring model from checkpoint {}.'.format(FLAGS.checkpoint_y))
            spn_y.restore()
    else:
        print('Starting from scratch')

    # 评估的数据集和函数的构造
    print('Making eval datasets and eval funcions.')
    if FLAGS.eval_on:
        evaluate, _ =spn_data.make_eval_function(
            FLAGS.eval_on,
            progress_bar=True,
            plot_dir=FLAGS.plot_dir,
            num_plots=100)

    #  训练的数据集迭代器构建（已完成）
    if FLAGS.train_on:
        print('Making trainning iterator.')
        train_it = spn_data.make_train_iterator(
            FLAGS.train_on,
            FLAGS.dim_x,
            FLAGS.dim_y,
            FLAGS.shuffle_buffer_size,
            FLAGS.batch_size,   
        )

        print('Starting training loop')
        log = dict()
        epoch = 0

        loss_x_print=[]
        loss_y_print=[]
        loss_print=[]

        error_x = []
        error_y = []

        Error_x = 100.
        Error_y = 100.

        while True:

            current_step = tf.compat.v1.train.get_or_create_global_step().numpy() 

            num_steps = FLAGS.epoch_length
            log1 = dict()

            if callable(spn_x._learning_rate):
                lr = spn_x._learning_rate()
            else:
                lr = spn_x._learning_rate

            start_time_data = time.time()

            losses = {}
            loss_x_p = 0.
            loss_y_p = 0.
            loss_p = 0.


            #开始训练
            for num, squence in zip(range(num_steps), train_it):

                stop_time_data = time.time()
                sys.stdout.write(f'{num},')
                sys.stdout.flush()

                global_step = tf.compat.v1.train.get_or_create_global_step()
                global_step.assign(global_step + 1)

                sequence_xy, sequence_xy_true = squence

                # print(sequence_xy)
                sequence_xy = tf.transpose(sequence_xy, [0, 2, 1])
                # print('sequence_xy: ', sequence_xy)
                # print('sequence_xy_true[:,0] :', sequence_xy_true[:,0])
                # print('sequence_xy_true[:,0] :', sequence_xy_true[:,1])

                sequence_x_true = sequence_xy_true[:,0]
                sequence_y_true = sequence_xy_true[:,1]

                start_time_train_step = time.time()

                loss_x, loss_y = spn_net.train_step(spn_x, spn_y, sequence_xy, sequence_x_true, sequence_y_true)

                stop_time_train_data = time.time()

                losses['loss_x'] = loss_x
                losses['loss_y'] = loss_y
                losses['total-loss'] = (loss_x + loss_y) / 2.0

                log_updata = losses

                log_updata['data-time'] = (stop_time_data - start_time_data) * FLAGS.epoch_length
                log_updata['train-time'] = (stop_time_train_data - start_time_train_step) * FLAGS.epoch_length

                for key in log_updata:
                    if key in log1:
                        log1[key].append(log_updata[key])
                    else:
                        log1[key]= [log_updata[key]]

                start_time_data = time.time()

                # if num % 500 == 0:
                #     print('sequence_xy: ', sequence_xy)
                #     print('sequence_xy_true[:,0] :', sequence_xy_true[:,0])
                #     print('sequence_xy_true[:,1] :', sequence_xy_true[:,1])
                #     print('f_x: ', spn_x(sequence_xy))
                #     print('f_y: ', spn_y(sequence_xy))
            
            for key in log1:
                log1[key] = tf.reduce_mean(input_tensor=log1[key])

            sys.stdout.write('\n')
            sys.stdout.flush()

            for key in log1:
                if key in log:
                    log[key].append(log1[key])
                else:
                    log[key] = [log1[key]]

            # loss_x_print.append(log['loss_x'][-1])
            # loss_y_print.append(log['loss_y'][-1])
            # loss_print.append(log['total-loss'][-1])

            if FLAGS.checkpoint_x and not FLAGS.no_checkpointing:
                spn_x.save()
                
            if FLAGS.checkpoint_y and not FLAGS.no_checkpointing:
                spn_y.save()

            status = spn_plotting.print_log(log, epoch, lr)

            loss_x_print.append(float(status.split(':')[4].split(',')[0][1:]))
            loss_y_print.append(float(status.split(':')[5].split(',')[0][1:]))
            loss_print.append(float(status.split(':')[1].split(',')[0][1:]))

            with open("loss_x.txt", "w") as file:
                for loss_x in loss_x_print:
                    file.write(str(loss_x))
                    file.write('\n')

            with open("loss_y.txt", "w") as file:
                for loss_y in loss_y_print:
                    file.write(str(loss_y))
                    file.write('\n')

            with open("loss_total.txt", "w") as file:
                for loss in loss_print:
                    file.write(str(loss))
                    file.write('\n')

            # 绘制loss图像
            spn_plotting.plot_loss_line(FLAGS.plot_dir_loss, loss_x_print, loss_y_print, loss_print, epoch)

            if FLAGS.eval_on and FLAGS.evaluate_during_train and epoch % 5 == 0:
                # 开始评估
                eval_results = evaluate(spn_x, spn_y)
                status_error = spn_plotting.print_eval(eval_results)
                error_x.append(float(status_error.split(':')[3].split(',')[0][1:]))
                error_y.append(float(status_error.split(':')[5].split(',')[0][1:]))
                # spn_plotting.plot_error(FLAGS.plot_dir_loss, error_x, error_y, epoch)

                # 保存最好模型
                Error_x_now = float(status_error.split(':')[3].split(',')[0][1:])
                Error_y_now = float(status_error.split(':')[5].split(',')[0][1:])

                with open("Error_x.txt", "w") as file:
                    for loss_x in error_x:
                        file.write(str(loss_x))
                        file.write('\n')
        
                with open("Error_y.txt", "w") as file:
                    for loss_y in error_y:
                        file.write(str(loss_y))
                        file.write('\n')


                if Error_x_now < Error_x:
                    Error_x = Error_x_now
                    spn_x.save_best()

                if Error_y_now < Error_y:
                    Error_y = Error_y_now
                    spn_y.save_best()


            if current_step >= FLAGS.num_train_steps:
                break

            epoch += 1

    else:
        print('Specify flag train_on to enable training')
        print('Just doing evaluation now')
        eval_results = evaluate(spn_x, spn_y)
        if eval_results:
            spn_plotting.print_eval(eval_results)
        print('Evaluation complete')


if __name__ == '__main__':
    app.run(main)







