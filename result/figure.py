# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-08 19:33:16
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-08 19:58:07
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as mtick
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

# x = np.linspace(0, 2*math.pi, 10)
def draw_same_min():
    x = np.array([i + 1 for i in range(10)])
    print(x)
    # print(x)
    # y1, y2 = np.sin(x), np.cos(x)
    # y_float = [0.88, 0.86, 0.86, 0.85, 0.84, 0.83, 0.82, 0.82, 0.82, 0.82]
    #持续训练和更新
    y_float = [0.88, 0.86, 0.86, 0.85, 0.85, 0.84, 0.85, 0.85, 0.85, 0.85]
    y_gcn = np.array([i * 100 for i in y_float])

    y_nn_float = [0.81, 0.77, 0.81, 0.80, 0.80, 0.78, 0.75, 0.79, 0.78, 0.79]
    y_nn = np.array([i * 100 for i in y_nn_float])

    y_dt_float = [0.82, 0.81, 0.82, 0.80, 0.78, 0.80, 0.81, 0.81, 0.80, 0.80]
    y_dt = np.array([i * 100 for i in y_dt_float])
    plt.plot(x, y_gcn, label='GNN', c='black', linestyle='-', linewidth=2, 
            marker='^', markeredgecolor='black', markerfacecolor='white', markersize=10)
    plt.plot(x, y_nn, label='NN', c='blue', linestyle='-', linewidth=2, 
            marker='o', markerfacecolor='white', markersize=10)
    plt.plot(x, y_dt, label='DTree', c='green', linestyle='-', linewidth=2, 
            marker='s', markerfacecolor='white', markersize=10)
    # plt.title('Dir A', fontsize=24)
    plt.xlabel('circle', fontsize=14)
    plt.ylabel('accuracy(%)', fontsize=14)

    x_major_locator=MultipleLocator(1)
    #把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator=MultipleLocator(10)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    plt.xlim(1,10)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    # plt.ylim(-5,100)
    plt.ylim(45, 100)
    #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.grid(axis='y', linestyle='-.')

    plt.legend(loc=3) 
    plt.show()

def AE():
    # AE
    mice_500 = [0.90 for i in range(8)]
    ele_500 = [0.93, 0.94, 0.93, 0.93, 0.92, 0.93, 0.93, 0.93]
    mice_2000 = [0.90 for i in range(5)]
    ele_2000 = [0.97, 0.96, 0.96, 0.97, 0.97]
    mice_20000 = [0.90 for i in range(4)]
    ele_20000 = [0.99, 0.97, 0.98, 0.98]

    # VAE
    VAE_result = {
        "500":{
            "mice":[0.90 for i in range(10)],
            "ele":[0.9 + i/100.0 for i in [3, 4, 3, 3, 2, 3, 3, 3, 3, 2]]
        },
        "2000":{
            "mice":[0.90 for i in range(10)],
            "ele":[0.9 + i/100.0 for i in [7,6,6,7,7,7,7,7,7,7]]
        },
        "20000":{
            "mice":[0.90 for i in range(10)],
            "ele":[0.9 + i/100.0 for i in [9,7,8,8,7,2,2,3,4,4]]
        }
    }

    x = np.array([i + 1 for i in range(10)])
    print(x)
    # print(x)
    # y1, y2 = np.sin(x), np.cos(x)
    # y_float = [0.88, 0.86, 0.86, 0.85, 0.84, 0.83, 0.82, 0.82, 0.82, 0.82]
    #持续训练和更新
    y_float = [0.88, 0.86, 0.86, 0.85, 0.85, 0.84, 0.85, 0.85, 0.85, 0.85]
    # y_gcn = np.array([i * 100 for i in y_float])
    y_gcn = [i * 100 for i in VAE_result["500"]["mice"]]
    print(y_gcn)

    y_nn_float = [0.81, 0.77, 0.81, 0.80, 0.80, 0.78, 0.75, 0.79, 0.78, 0.79]
    # y_nn = np.array([i * 100 for i in y_nn_float])
    y_nn = [i * 100 for i in VAE_result["2000"]["mice"]]

    y_dt_float = [0.82, 0.81, 0.82, 0.80, 0.78, 0.80, 0.81, 0.81, 0.80, 0.80]
    # y_dt = np.array([i * 100 for i in y_dt_float])
    y_dt = [i * 100 for i in VAE_result["20000"]["mice"]]
    plt.plot(x, y_gcn, label='500', c='black', linestyle='-', linewidth=2, 
            marker='^', markeredgecolor='black', markerfacecolor='white', markersize=10)
    plt.plot(x, y_nn, label='2000', c='blue', linestyle='-', linewidth=2, 
            marker='o', markerfacecolor='white', markersize=10)
    plt.plot(x, y_dt, label='20000', c='green', linestyle='-', linewidth=2, 
            marker='s', markerfacecolor='white', markersize=10)
    # plt.title('Dir A', fontsize=24)
    plt.xlabel('circle', fontsize=14)
    plt.ylabel('accuracy(%)', fontsize=14)

    x_major_locator=MultipleLocator(1)
    #把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator=MultipleLocator(10)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    plt.xlim(1,10)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    # plt.ylim(-5,100)
    plt.ylim(45, 100)
    #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.grid(axis='y', linestyle='-.')

    plt.legend(loc=3) 
    plt.show()


if __name__ == '__main__':
    # draw_same_min()
    # draw_differ_min()
    # switch_dir()
    # choose_p()
    # choose_k()
    # choose_w()
    # choose_sketch()
    # choose_aggregation()
    AE()