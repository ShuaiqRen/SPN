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

import numpy as np
import math
import time
import tensorflow as tf
import math

from absl import flags
from absl import app

from spn import apply_spn
from spn import spn_plotting
from spn import spn_flags
from spn import image_to_input_v2_for_8_point
from spn import image_to_input_v2_for_16_point
from spn import image_to_input_v2
from spn import image_to_input_v2_for_anylength

FLAGS = flags.FLAGS

flags.DEFINE_integer('dimx', 4, 'dim_x')
flags.DEFINE_integer('dimy', 8, 'dim_y')
flags.DEFINE_integer('a', 1, 'location parameter for crossover')
flags.DEFINE_integer('b', 1, 'location parameter for crossover')
flags.DEFINE_integer('p', 1, 'the strength of perturbation of power mutation')
flags.DEFINE_integer('running_num', 100, 'the num of running')
flags.DEFINE_integer('pop_size', 100 * 1, 'pop_size')
flags.DEFINE_integer('pe', 10 * 1, 'the percentage for the top 5%')
flags.DEFINE_integer('pc', 35 * 1, 'the percentage of crossover')
flags.DEFINE_integer('pm', 10 * 1, 'the percentage of mutation')
flags.DEFINE_integer('pn', 10 * 1, 'the percentage of new rand')

FLAGS.dimy = FLAGS.n

print(FLAGS.dimy)


#定义好想要的形状
# truth_value = image_to_input_v2_for_8_point.convert_image_to_array('./spn/12.jpg')
truth_value = image_to_input_v2_for_anylength.convert_image_to_array('./spn/12.jpg', FLAGS.n)

spn_plotting.plt_ev(FLAGS.plot_dir, truth_value)

#初始化第一代
def initpop(pop_size, dim_x, dim_y):
    pop = np.zeros([pop_size, dim_x, dim_y])
    for i in range(pop_size):
        pop[i] = np.random.randint(2, size=(dim_x, dim_y))

    return pop # 100*4*24


#生成后代 随机生成后代
def create_random_pop(dim_x, dim_y, pn):
    new_pop = np.zeros([pn, dim_x, dim_y])
    for i in range(pn):
        new_pop[i] = np.random.randint(2, size=(dim_x, dim_y))

    return new_pop


#利用spn进行输出形状的预测
def spn_pop(pop_size, pop, spn_x, spn_y):
    pop_value = [0 for _ in range(pop_size)]

    for i in range(pop_size):
        pop_i = pop[i, :, :]
        pop_value[i] = apply_spn.forward(spn_x, spn_y, pop_i)

    return pop_value # pop_size * 2 * 24


#与给定的形状进行对比，计算种群的函数值
def ev_pop(pop_size, pop_value, truth_value):
    k_x = 1.
    k_y = 1.
    loss = np.zeros([pop_size])

    for i in range(FLAGS.dimy):
        loss_x = (pop_value[:, 0, i] - truth_value[0, i])**2 
        loss_y = (pop_value[:, 1, i] - truth_value[1, i])**2 
        # 1
        # loss += loss_x + loss_y
        # 2
        # loss += (loss_x + loss_y)/(i+1)
        # 3
        # loss += (k_x *loss_x + k_y * loss_y) / (i/2 + 1)
        # 4
        loss += (k_x *loss_x + k_y * loss_y) / (abs(i - ((FLAGS.dimy - 1) / 2)) / 2 + 1)

    loss = (loss/FLAGS.dimy)**(0.5)

    return loss

#选择父代个体
def choose_elit(pop, loss, p):
    """
    pop: 上一代的所有种群
    loss: 上一代计算出的适应度
    p: 进行后续操作的比例
    """
    fit_index = np.argsort(loss)
    _, dim_x, dim_y = pop.shape
    new_pop = np.zeros((p, dim_x, dim_y))

    for i in range(p):
        new_pop[i, :, :] = pop[fit_index[i], :, :]

    return new_pop


def Limit_to_integer(num):
    if num < 0.5:
        return 0
    else:
        return 1


def Limit_to_integer_crossover(num):
    if num < 1.5:
        return 0
    else:
        return 1

def crossover(pop_1, pop_2):
    child_pop_1 = np.zeros((FLAGS.dimx, FLAGS.dimy))
    child_pop_2 = np.zeros((FLAGS.dimx, FLAGS.dimy))

    for i in range(FLAGS.dimx):
        for j in range(FLAGS.dimy):
            r1 = np.random.rand(1)
            r2 = np.random.rand(1)

            child_pop_1[i, j] = Limit_to_integer_crossover(pop_1[i, j] + pop_2[i, j] + r1)
            child_pop_2[i, j] = Limit_to_integer_crossover(pop_2[i, j] + pop_1[i, j] + r2)

    return child_pop_1, child_pop_2

# 杂交操作
def popcrossover(pop, pc, pop_value, truth_value):
    """
    pop: 进行杂交操作的父代们,经过比例选择的.
    loss: 每一个个体的损失
    pc: 杂交的个体数
    """
    _, dim_x, dim_y = pop.shape
    child_pop = np.zeros((pc*2, dim_x, dim_y))

    for i in range(pc):
        num_1, num_2 = np.random.randint(FLAGS.pe, size=2)
        child_pop[i], child_pop[i+1] = crossover(pop[num_1], pop[num_2])
        child_pop[i] = mutation(child_pop[i])
        child_pop[i+1] = mutation(child_pop[i+1])

    return child_pop

def mutation_compute(pop):
    s = np.random.rand(1)
    s1 = s**(FLAGS.p)
    
    r = np.random.rand(1)

    if r > 0.5:
        return Limit_to_integer(pop + s1 * (1 - pop))
    else: 
        return Limit_to_integer(pop - s1 * (pop - 0))

#单个变异(位置完全随机)
def mutation(pop):
    child_pop = np.zeros((FLAGS.dimx, FLAGS.dimy))
    child_pop = pop

    for i in range(FLAGS.dimx):
        # print(child_pop)
        i = np.random.randint(FLAGS.dimx, size=1)
        j = np.random.randint(FLAGS.dimy, size=1)
        # print(i, j)
        child_pop[i, j] = 1 - child_pop[i, j]
    
    return child_pop

#变异操作(变异位置完全随机)
def popmutation(pop, pm):
    """
    pop: 进行变异的群落,经过特定比例处理过的
    pm: 变异的个数
    """
    _, dim_x, dim_y = pop.shape
    child_pop = np.zeros((pm, dim_x, dim_y))# 10 * 4 *24
    for i in range(pm):
        #选择父代
        num_1 = np.random.randint(FLAGS.pe, size=1)
        # print(pop[num_1][0])
        # print(pop[num_1][0].shape)
        child_pop[i] = mutation(pop[num_1][0])

    # print(child_pop.shape)

    return child_pop

# 获得最佳个体
def best_fit(pop, loss):
    fit_index = np.argsort(loss)
    best_one = pop[fit_index[0]]
    best_loss = loss[fit_index[0]]

    return best_one, best_loss

def print_best_one(best_one):
    for i in range(FLAGS.dimx):
        if i != 3:
            str_best = '[' + ','.join(str(int(i)) for i in best_one[i]) + ']' + ','
            print(str_best)
        else:
            str_best = '[' + ','.join(str(int(i)) for i in best_one[i]) + ']'
            print(str_best)

def compute_length(value):
    length = 0.
    num = FLAGS.dimy - 1
    for i in range(num):
        length += ((value[0, i+1] - value[0, i])**2 + (value[1, i+1] - value[1, i])**2)**0.5

    return length

def compute_loss_xy(truth_value, output):
    loss_x = 0.
    loss_y = 0.
    for i in range(FLAGS.dimy):
        loss_x += (truth_value[0, i] - output[0, i])**2
        loss_y += (truth_value[1, i] - output[1, i])**2

    return loss_x, loss_y

 
def main(argv):

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # #
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3600)])

    print('truth_length :', compute_length(truth_value))

    if not FLAGS.plot_dir:
        raise ValueError('apply_spn need plot directory')
    if not tf.io.gfile.exists(FLAGS.plot_dir):
        print('Making new plot directory', FLAGS.plot_dir)
        tf.io.gfile.makedirs(FLAGS.plot_dir)

    # 生成spn_x和spn_y
    spn_x, spn_y = apply_spn.forward_load_model()

    #随机生成第一代数据
    pop = initpop(FLAGS.pop_size, FLAGS.dimx, FLAGS.dimy) # 100*4*24
    pop_value = np.array(spn_pop(FLAGS.pop_size, pop, spn_x, spn_y))
    loss = ev_pop(FLAGS.pop_size, pop_value, truth_value)

    for i in range(FLAGS.running_num):
        start_time = time.time()

        if FLAGS.pe != 0:
            pop_elit = choose_elit(pop, loss, FLAGS.pe)
            # print(pop_elit.shape)

        if FLAGS.pc != 0:
            pop_crossover = popcrossover(pop_elit, FLAGS.pc, pop_value, truth_value)
            # print(pop_crossover.shape)

        if FLAGS.pm != 0:
            # print(pop_elit.shape)
            # print(FLAGS.pm)
            # print('start')
            pop_mumation = popmutation(pop_elit, FLAGS.pm)
            # print(pop_mumation.shape)

        if FLAGS.pn != 0:
            pop_new_rand = create_random_pop(FLAGS.dimx, FLAGS.dimy, FLAGS.pn)
            # print(pop_new_rand.shape)

        if FLAGS.pe != 0 and FLAGS.pc != 0 and FLAGS.pm != 0 and FLAGS.pn != 0:
            pop = np.vstack((pop_elit, pop_crossover, pop_mumation, pop_new_rand))
        elif FLAGS.pe != 0 and FLAGS.pc != 0 and FLAGS.pm == 0 and FLAGS.pn != 0:
            pop = np.vstack((pop_elit, pop_crossover, pop_new_rand))
        elif FLAGS.pe != 0 and FLAGS.pc == 0 and FLAGS.pm != 0 and FLAGS.pn != 0:
            pop = np.vstack((pop_elit, pop_mumation, pop_new_rand))
        elif FLAGS.pe != 0 and FLAGS.pc != 0 and FLAGS.pm != 0 and FLAGS.pn == 0:
            pop = np.vstack((pop_elit, pop_crossover, pop_mumation))
        elif FLAGS.pe != 0 and FLAGS.pc != 0 and FLAGS.pm == 0 and FLAGS.pn == 0:
            pop = np.vstack((pop_elit, pop_crossover))
        elif FLAGS.pe != 0 and FLAGS.pc == 0 and FLAGS.pm != 0 and FLAGS.pn == 0:
            pop = np.vstack((pop_elit, pop_mumation))
        elif FLAGS.pe == 0 and FLAGS.pc != 0 and FLAGS.pm == 0 and FLAGS.pn == 0:
            pop = pop_crossover
        # print(pop.shape)
        # assert pop.shape != FLAGS.pop_size

        pop_value = np.array(spn_pop(FLAGS.pop_size, pop, spn_x, spn_y)) 
        loss = ev_pop(FLAGS.pop_size, pop_value, truth_value)

        best_one, best_loss = best_fit(pop, loss)
        end_time = time.time()
        print('------------------------------------')
        print('process :', i)
        print('best_one :')
        print_best_one(best_one)
        print('best_loss :', best_loss)
        print('process_time :', end_time - start_time)

    print('complete this process !!!')
    pop_best = best_one
    output = apply_spn.forward(spn_x, spn_y, pop_best)
    print('truth_length :', compute_length(truth_value))
    print('predict_length :', compute_length(np.array(output)))
    loss_x, loss_y = compute_loss_xy(truth_value, np.array(output))
    print('loss_x :', loss_x, 'loss_y :', loss_y)

    # print(truth_value)
    # print(output)

    # write the turth value and prediction value
    a = truth_value.T
    with open("./spn/truth_ev.txt", "w") as file:
        for j in range(FLAGS.dimy):
            file.write(str(a[j, 0]))
            file.write('\t')
            file.write(str(a[j, 1]))
            file.write('\t')
            file.write('\n')

    b = np.array(output).T
    with open("./spn/pre_ev.txt", "w") as file:
        for j in range(FLAGS.dimy):
            file.write(str(b[j, 0]))
            file.write('\t')
            file.write(str(b[j, 1]))
            file.write('\t')
            file.write('\n')

    spn_plotting.plt_ev_result(FLAGS.plot_dir, truth_value, output)
        

if __name__ == '__main__':
  app.run(main)