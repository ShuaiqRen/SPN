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

"""SPN plotting

This library orovides some plotting functionlity for plt
"""

import io
import os
import time 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import tensorflow as tf

def print_log(log, epoch, lr=0.005,  mean_over_num_steps=1):
    if epoch is None:
        status = ''
    else:
        status = '{} -- '.format(epoch)

    status += 'total-loss: {:.6f}'.format(
        np.mean(log['total-loss'][-mean_over_num_steps:]))
    
    status += ', learning-rate: {:.6f}'.format(lr)

    for key in sorted(log):
        if key not in ['total-loss']:
            loss_mean = np.mean(log[key][-mean_over_num_steps:])
            status += ', {}: {:.6f}'.format(key, loss_mean)
    print(status)

    return status

def print_eval(eval_dict):
    """Prints eval_dict to sonsole"""

    status = ''.join(
        ['{}: {:.6f},'.format(key, eval_dict[key]) for key in sorted(eval_dict)]
    )
    print(status)

    return status


def complete_paper_plot(
    plot_dir,
    index,
    sequence_xy,
    sequence_xy_true,
    f_x,
    f_y
):

    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = str(index) + '_' + name
        plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
        plt.clf()

    plt.figure()
    plt.clf()

    plt.axis('equal') 

    plt.plot(sequence_xy_true[0], sequence_xy_true[1], color='red', label='ground')
    plt.plot(f_x, f_y, color='blue', label='predict')
    save_fig('GT_PRE', plot_dir)

    plt.close()

def plt_output_shape(output, plot_dir):
    
    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
        plt.clf()

    plt.figure()
    plt.clf()


    plt.axis('equal') 
    plt.plot(output[0], output[1], color='blue', label='predict')
    save_fig('PRE', plot_dir)

    plt.close()

def plt_shape(plot_dir, input):
    
    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
        plt.clf()

    plt.figure()
    plt.clf()

    plt.axis('equal') 
    plt.plot(input[0], input[1], color='blue', label='predict')
    save_fig('input', plot_dir)

    plt.close()

def plt_line(plot_dir, input):
    
    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
        plt.clf()

    plt.figure()
    plt.clf()

    plt.plot(input)
    save_fig('line', plot_dir)

    plt.close()

def plt_line_withsegment(plot_dir, input):
    
    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
        plt.clf()

    x_values = np.arange(len(input))

    plt.figure()
    plt.clf()

    plt.scatter(x=x_values, y=input)
    plt.axhline(0.0420, color='green')
    plt.axhline(0.0326, color='green')
    plt.axhline(0.0200, color='green')
    plt.axhline(0.0040, color='green')
    plt.axhline(0.0005, color='green')
    plt.axhline(-0.0420, color='green')
    plt.axhline(-0.0326, color='green')
    plt.axhline(-0.0200, color='green')
    plt.axhline(-0.0040, color='green')
    plt.axhline(-0.0005, color='green')
    save_fig('line', plot_dir)

    plt.close()

def plt_ev_result(
    plot_dir,
    output_true,
    output_fake
):

    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()

    plt.figure()
    plt.clf()

    plt.axis('equal') 

    plt.plot(output_true[0], output_true[1], color='red', label='ground')
    plt.plot(output_fake[0], output_fake[1], color='blue', label='predict')
    save_fig('EV_GT_PRE', plot_dir)

    plt.close()

def plt_ev_result_with_point(
    plot_dir,
    tartget_point,
    output_fake
):

    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()

    plt.figure()
    plt.clf()

    plt.axis('equal')
    plt.plot(output_fake[0], output_fake[1], color='blue', label='predict')
    plt.scatter(tartget_point[0], tartget_point[1], color='red')
    save_fig('EV_Point_Pre', plot_dir)

    plt.close()


def plt_ev_result_with_point_occlusion(
    plot_dir,
    tartget_point,
    output_fake,
    occlusion_point,
    d
):

    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()

    r = d
    a, b = occlusion_point
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)

    plt.figure()
    plt.clf()

    plt.axis('equal')
    plt.plot(output_fake[0], output_fake[1], color='blue', label='predict')
    plt.scatter(tartget_point[0], tartget_point[1], color='red')
    plt.scatter(occlusion_point[0], occlusion_point[1], color='green')
    plt.plot(x,y, color='green', label='occlusion_region')
    save_fig('EV_Point_Pre', plot_dir)

    plt.close()

def plt_pre_result(
    plot_dir,
    output_true,
    output_pres,
    num
):

    def save_fig(name, num, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name + str(num)
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()

    plt.figure()
    plt.clf()

    plt.axis('equal') 

    plt.plot(output_true[0, :(num+1)], output_true[1, :(num+1)], color='red', label='ground')
    for i in range(16):
        plt.plot(output_pres[i, 0], output_pres[i, 1], color='blue', label='predict')
    save_fig('Pre', num, plot_dir)

    plt.close()

def plt_pre(
    plot_dir,
    output_pres
):

    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()

    plt.figure()
    plt.clf()

    plt.axis('equal') 

    for i in range(10):
        plt.plot(output_pres[i, 0], output_pres[i, 1], color='blue', label='predict')
    save_fig('Pre', plot_dir)

    plt.close()

def plt_ev(
    plot_dir,
    output_true
):

    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()

    plt.figure()
    plt.clf()

    plt.axis('equal') 

    plt.plot(output_true[0], output_true[1], color='red', label='ground')
    save_fig('GT', plot_dir)

    plt.close()

def plt_index_result(
    plot_dir,
    index,
    output
):

    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = str(index) + '_' + name
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()

    for i in range(16):
        plt.figure()
        plt.clf()

        plt.axis('equal') 

        plt.plot(output[i, 0, :], output[i, 1, :], color='blue')
        name = 'PRE_' + str(i) + '_'
        save_fig(name, plot_dir)

        plt.close()

def up_up(data):
    for i in range(np.array(data).shape[0]):
        if data[i] > 100. :
            data[i] = 100.

    return data

def up_up_error(data):
    for i in range(np.array(data).shape[0]):
        if data[i] > 10. :
            data[i] = 10.

    return data


def plot_loss_line(
    plot_dir,
    loss_x,
    loss_y,
    loss,
    epoch
):
    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()

    # print(loss_x)
    # print(np.array(loss_x))
    # print(np.array(loss_x).shape)
    # shape_x = np.array(loss_x).shape
    inter = [i+1 for i in range((epoch+1))]

    loss_x = up_up(loss_x)
    loss_y = up_up(loss_y)
    loss = up_up(loss)
    

    plt.figure()
    plt.clf()

    plt.plot(inter, loss_x, color='red')
    save_fig('loss_x', plot_dir)
    

    plt.plot(inter, loss_y, color='red')
    save_fig('loss_y', plot_dir)

    plt.plot(inter, loss, color='red')
    save_fig('loss', plot_dir)

    plt.close('all')


def plot_error(
    plot_dir,
    error_x,
    error_y,
    epoch
):
    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()

    inter = [i+1 for i in range(int(((epoch/5)+1)))]

    # error_x = up_up_error(error_x)
    # error_y = up_up_error(error_y)
    # print(error_x)
    # print(error_y)
    
    plt.figure()
    plt.clf()

    plt.plot(inter, error_x, color='red', label='error_x')
    save_fig('error_x', plot_dir)

    plt.plot(inter, error_y, color='blue', label='error_y')
    save_fig('error_y', plot_dir)

    plt.close('all')

def PJcurvature(x,y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    """
    t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
    t_b = LA.norm([x[2]-x[1],y[2]-y[1]])
    
    M = np.array([
        [1, -t_a, t_a**2],
        [1, 0,    0     ],
        [1,  t_b, t_b**2]
    ])

    a = np.matmul(LA.inv(M),x)
    b = np.matmul(LA.inv(M),y)

    kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    return kappa, [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)

def plot_curvature(plot_dir, pop_parent_crossover_value, truth_value):
        
    def save_fig(name, plot_dir):
        plt.xticks()
        plt.yticks()
        filepath = name
        plt.savefig(os.path.join(plot_dir, filepath))
        plt.clf()
    
    pop_size, pop_x, pop_y = pop_parent_crossover_value.shape

    for i in range(pop_size):
        ka = []
        no = []
        po = []
        for j in range(pop_y - 2):
            x = [pop_parent_crossover_value[i, 0, j], pop_parent_crossover_value[i, 0, j+1], pop_parent_crossover_value[i, 0, j+2]]
            y = [pop_parent_crossover_value[i, 1, j], pop_parent_crossover_value[i, 1, j+1], pop_parent_crossover_value[i, 1, j+2]]
            kappa,norm = PJcurvature(x,y)
            ka.append(kappa)
            no.append(norm)
            po.append([x[1],y[1]])

        po = np.array(po)
        no = np.array(no)
        ka = np.array(ka)

        plt.figure()
        plt.clf()
        plt.plot(po[:,0],po[:,1])
        plt.quiver(po[:,0],po[:,1],ka*no[:,0],ka*no[:,1])
        plt.axis('equal') 
        save_fig('pop_%d'%(i), plot_dir)
        plt.close()

    ka = []
    no = []
    po = []

    for i in range(pop_y - 2):

        x = [truth_value[0, i], truth_value[0, i+1], truth_value[0, i+2]]
        y = [truth_value[1, i], truth_value[1, i+1], truth_value[1, i+2]]
        kappa,norm = PJcurvature(x,y)
        ka.append(kappa)
        no.append(norm)
        po.append([x[1],y[1]])

    po = np.array(po)
    no = np.array(no)
    ka = np.array(ka)

    plt.figure()
    plt.clf()
    plt.plot(po[:,0],po[:,1])
    plt.quiver(po[:,0],po[:,1],ka*no[:,0],ka*no[:,1])
    plt.axis('equal') 
    save_fig('truth', plot_dir)
    plt.close('all')
