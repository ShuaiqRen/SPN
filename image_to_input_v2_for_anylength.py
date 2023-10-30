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

import numpy as np
import pdb
import cv2
import time

from absl import app
from absl import flags

import matplotlib.pyplot as plt

# from spn import spn_flags
from spn import spn_flags

FLAGS = flags.FLAGS

# flags.DEFINE_integer('length', 72, 'the expect num for design')

# n = 16 # n是需要提取的点数，也即是长度

# 黑色像素是0，白色像素是255
# 读取图片和裁剪图片大小
def read_with_resize(img_path):
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (800, 600))

    return img_resize


# canny边缘检测
def canny_edge_detection(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(img, 50, 150)

    return canny


# 选择性的剔除points中的值，每一条线上如果两点距离小于3就剔除，只保留最上方的坐标点
def choose_points(points):
    candid_points = []
    points_num, _ = points.shape
    per_point = points[0]
    candid_points.append(per_point)

    for i in range(points_num - 1):
        if per_point[0] == points[i + 1][0]:
            if points[i + 1][1] - per_point[1] > 10:
                candid_points.append(points[i + 1])
        else:
            candid_points.append(points[i + 1])
        per_point = points[i + 1]

    return np.array(candid_points)


# 随机选择一个点作为初始点，并随机初始化一个向量T1作为初始方向
def init_point_vector(img):
    img_x, img_y = img.shape
    points = []
    # print(img_x, img_y)
    # points = np.array(np.where(img == 255)).T
    for i in range(img_y):
        for j in range(img_x):
            if img[j, i] == 255:
                points.append([i, j])
    points = np.array(points)
    points = choose_points(points)
    points_x, points_y = points.shape # points_x * 2 一共有points_x个白色像素点，每一个第一个是y坐标，第二个是x坐标
    index = np.random.randint(5, points_x-5, size=1)
    point = points[index, :][0] # 随机选取的坐标点
    vector = np.random.randint(-5, 5, size=2)

    return point, vector, points


# 正向选取坐标点，并保存坐标点
def forward_choose_points(points, point, vector):

    tmp = 0
    candid_point = []
    candid_vector = []
    for p in points:
        if ((p[0] - point[0])**2 + (p[1] - point[1])**2)**0.5 < 30:
            vector1 = [p[0] - point[0], p[1] - point[1]]
            mask = (vector1[0] * vector[0]) + (vector1[1] * vector[1])# 就算向量点乘结果
            if mask > tmp:
                tmp = mask
                candid_point = p
                candid_vector = [p[0] - point[0], p[1] - point[1]]

    return candid_point, candid_vector, tmp


# 循环进行前向选择点
def forward_choose(points, point, vector):

    candid_point = []
    candid_vector = []
    while True:
        point, vector, tmp = forward_choose_points(points, point, vector)
        # 判断是否需要停止：
        if tmp == 0:
            break

        candid_point.append(point)
        candid_vector.append(vector)

    return candid_point, candid_vector


def choose_points_in_order(points, point, vector):

    # 正向
    candid_points, candid_vector = forward_choose(points, point, vector)
    candid_points = np.array(candid_points)
    candid_vector = np.array(candid_vector)
    # print(candid_points)
    # 反向
    vector1 = -vector
    # print(point, vector1)
    candid_points1, candid_vector1 = forward_choose(points, point, vector1)
    candid_points1 = np.array(candid_points1)
    candid_vector1 = np.array(candid_vector1)
    # print(candid_points1)

    # 拼接
    points_been_chosed = []
    if vector[0] < 0:# vector反向，即是前半部分
        num_before, _ = candid_points.shape
        for i in range(num_before):#倒序进入
            points_been_chosed.append(candid_points[num_before-i-1, :])

        points_been_chosed.append(point)

        num_after, _ = candid_points1.shape

        for i in range(num_after):#正序进入
            points_been_chosed.append(candid_points1[i])
    else:
        num_before, _ = candid_points1.shape
        for i in range(num_before):  # 倒序进入
            points_been_chosed.append(candid_points1[num_before - i - 1, :])

        points_been_chosed.append(point)

        num_after, _ = candid_points.shape

        for i in range(num_after):  # 正序进入
            points_been_chosed.append(candid_points[i])

    return np.array(points_been_chosed)


# 找到第i个点应该在那一段，即是哪两个坐标点之间
def find_between(length, index, total_length, n):
    length_new = (total_length / (2 * n)) * index
    # print('length.shape : ', length.shape)
    num = length.shape[0]
    num_new = 0
    for i in range(num):
        if (length[i] <= length_new) and (length[i+1] >= length_new):
            num_new = i
            break

    return num_new, num_new+1


# 给定两个点，和相应的距离，找到直线上最符合长度的坐标点
def find_between_points(num_1, num_2, points, length, index, n):
    length_new = (length[-1] / (2 * n)) * index
    # 相差的距离
    length_sub = length_new - length[num_1]
    length_dif = length[num_2] - length[num_1]
    # 计算点的坐标
    scale = length_sub / length_dif
    point_x = points[num_1][0] + scale * (points[num_2][0] - points[num_1][0])
    point_y = points[num_1][1] + scale * (points[num_2][1] - points[num_1][1])
    # print('point :', points[num_1], 'point_2 :', points[num_2], 'scale :', scale, 'length_new :', length_new, 'length[num_1] : ', length[num_1], 'length[num_2] : ', length[num_2])

    return [point_x, point_y]


# 数据坐标转化
def covert_points_to_coordinate(points, n):
    # 计算总长
    total_length = 0.0
    length = []
    length.append(0.0)
    num, _ = points.shape # 32 * 2
    for i in range(num-1):
        total_length += ((points[i, 0] - points[i+1, 0])**2 + (points[i, 1] - points[i+1, 1])**2)**0.5
        length.append(total_length)
    # 计算比例尺度
    # scale = total_length / (3.17 * n)
    scale = total_length / FLAGS.length
    print('scale: ', scale)
    print('length_scale: ', FLAGS.length)
    # 长度逐渐增加
    length = np.array(length)
    # print(length.shape)
    # 找到第i个点应该在那一段，即是哪两个坐标点之间
    num_point = []
    for i in range((2 * n) - 1):  # 分成了48段
        num_i1, num_i2 = find_between(length, i+1, total_length, n)
        # print(num_i1, num_i2, length[num_i1], (total_length / 48) * (i+1), length[num_i2])
        num_point.append([num_i1, num_i2])
    num_point = np.array(num_point)
    # print(num_point.shape)  # 47*2
    # 找到在直线上最符合距离的点
    point_mid = []
    for i in range((2 * n) - 1):
        point = find_between_points(num_point[i, 0], num_point[i, 1], points, length, i+1, n)
        # print(point)
        point_mid.append(point)
    point_mid.insert(0, points[0])
    point_mid.append(points[-1])
    point_mid = np.array(point_mid)
    # print(point_mid.shape)  # 47*2

    return point_mid, scale


def convert_49_to_24(points_mid):
    point = []

    for i in range(24):
        point.append(points_mid[2 * i+1])

    return np.array(point)


def convert_to_real(points, scale, point_one):
    num, _ = points.shape
    points[:, 1] = 600 - points[:, 1]
    point_one[1] = 600 - point_one[1]

    for i in range(num):
        points[i, 0] = (points[i, 0] - point_one[0]) / scale
        points[i, 1] = (points[i, 1] - point_one[1]) / scale
    # print(points)

    return points.T


def choose_the_direction(points_final_real, n):
    print('start convert the directiion')
    direction = points_final_real[0, -1] - points_final_real[0, 0]
    if FLAGS.direction < 0 and direction < 0:
        return points_final_real
    elif FLAGS.direction < 0 and direction > 0:
        num_x = points_final_real[-1, 0]
        num_y = points_final_real[-1, 1]
        points_final_real_new = points_final_real
        points_final_real_new[0, :] = points_final_real[0, :] - num_x
        points_final_real_new[1, :] = points_final_real[1, :] - num_y
        for i in range(n):
            points_final_real[0, i] = points_final_real_new[0, n - 1 - i]
            points_final_real[1, i] = points_final_real_new[1, n - 1 - i]
        return points_final_real
    elif FLAGS.direction > 0 and direction > 0:
        return points_final_real
    else:
        num_x = points_final_real[-1, 0]
        num_y = points_final_real[-1, 1]
        points_final_real_new = points_final_real
        points_final_real_new[0, :] = points_final_real[0, :] - num_x
        points_final_real_new[1, :] = points_final_real[1, :] - num_y
        for i in range(n):
            points_final_real[0, i] = points_final_real_new[0, n - 1 - i]
            points_final_real[1, i] = points_final_real_new[1, n - 1 - i]
        return points_final_real


def convert_image_to_array(img_path, n):
    img = read_with_resize(img_path)
    canny = canny_edge_detection(img)
    point, vector, points = init_point_vector(canny)
    points_been_chosen = choose_points_in_order(points, point, vector)
    points_mid, scale = covert_points_to_coordinate(points_been_chosen, n)
    # points_final = convert_49_to_24(points_mid)
    points_temp = []
    for i in range(n):
        points_temp.append(points_mid[2 * i + 1])
    points_final = np.array(points_temp)
    points_final_real = convert_to_real(points_final, scale, points_mid[0])
    points_final_real = choose_the_direction(points_final_real, n)

    return points_final_real

def convert_image_to_double_array(img_path, n):
    img = read_with_resize(img_path)
    canny = canny_edge_detection(img)
    point, vector, points = init_point_vector(canny)
    points_been_chosen = choose_points_in_order(points, point, vector)
    points_mid, scale = covert_points_to_coordinate(points_been_chosen, n)
    points_final = np.array(points_mid)
    points_final_real = convert_to_real(points_final, scale, points_mid[0])
    points_final_real = choose_the_direction(points_final_real, n)

    return points_final_real


def main(argv):
    start_time = time.time()
    n = 24
    img = read_with_resize('./6.jpg')

    # cv2.imshow('img',img)
    canny = canny_edge_detection(img)
    # cv2.imshow('canny', canny)
    # cv2.waitKey(0)
    point, vector, points = init_point_vector(canny)
    # print(point, vector)

    points_been_chosen = choose_points_in_order(points, point, vector)
    # print('points_been_chosen.shape: ', points_been_chosen.shape)
    # print(points_been_chosen)
    points_mid, scale = covert_points_to_coordinate(points_been_chosen, n)
    # print(points_mid.shape)
    points_temp = []
    for i in range(n):
        points_temp.append(points_mid[2 * i + 1])
    points_final = np.array(points_temp)
    # print(points_final.shape)
    points_final_real = convert_to_real(points_final, scale, points_mid[0])
    print(points_final_real)

    # plt.axis([0, 800, 0, 600])
    # #
    # # plt.axis('equal')
    # plt.plot(points_been_chosen[:, 0], 600 - points_been_chosen[:, 1])
    # plt.scatter(points_been_chosen[:, 0], 600 - points_been_chosen[:, 1])
    # # plt.scatter(points_mid[:, 0], 600 - points_mid[:, 1], color='#8B0000')
    # # plt.scatter(points_final[:, 0], points_final[:, 1], color='#8B0000')
    # # plt.plot(points_final[:, 0], points_final[:, 1])
    # # plt.quiver(point[0], 600 - point[1], vector[0], -vector[1])
    # # plt.quiver(point[0], 600 - point[1], vector1[0], -vector1[1])
    # # plt.quiver(candid_points[:, 0], 600 - candid_points[:, 1], candid_vector[:, 0], -candid_vector[:, 1])
    # # plt.quiver(candid_points1[:, 0], 600 - candid_points1[:, 1], candid_vector1[:, 0], -candid_vector1[:, 1])
    #
    # # plt.plot(points_final[:, 0], 600 - points_final[:, 1], 'r')
    # plt.show()
    # plt.close()
    #
    print(points_final_real.shape)
    plt.axis('equal')
    plt.plot(points_final[:, 0], points_final[:, 1], linewidth=5.0)
    plt.scatter(points_final[:, 0], points_final[:, 1], color='#8B0000', s=300)
    plt.axis('off')
    plt.show()
    end_time = time.time()

    print('time : ', -start_time + end_time)





if __name__ == '__main__':
  app.run(main)