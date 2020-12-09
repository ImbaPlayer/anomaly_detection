# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-12-07 14:49:52
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-12-07 19:41:58

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as mtick
total_data = {}
total_data["train_5"] =  {'NN':[[0, 0, 0.028, 0.071], [1, 1, 0.997, 0.995]], 'DT': [[0.47079482791610117, 0.5841348478183052, 0.6588313623269415, 0.6799263414323965], [0.9677135906240943, 0.9489136243020474, 0.9326343444857725, 0.9109550348092438]], 'GPR': [[0.41729969834794034, 0.5405826961993507, 0.6407252346452645, 0.6862835472530862], [0.9784319429013458, 0.9617263838350449, 0.9448750764536039, 0.9255832131796425]], 'NB': [[0.997293132155562, 0.9766475864761727, 0.9487865232566609, 0.8849995118354562], [0.7296348660298086, 0.7618485405931456, 0.7120790449644304, 0.639576600900493]], 'SVM': [[0.0, 0.0008005865374124185, 0.37803201275228854, 0.5268372135946069], [1.0, 1.0, 0.9554100623595768, 0.9233218198246821]]}
total_data["train_size"] =  {'NN':[[0, 0, 0.465, 0.703], [1, 1, 0.995, 0.991]], 'GPR': [[0, 0.33146341756620185, 0.6543263946831027, 0.7613991028543261], [1, 0.9873811586170582, 0.9684659758605414, 0.9706898107563053]], 'NB': [[0.7975682267253866, 0.8302072803633559, 0.8483202940060629, 0.8464303010903121], [0.7456020240198705, 0.8212475731814545, 0.8452965827658341, 0.8830686436953981]], 'SVM': [[0.0, 0.6060768201344773, 0.6597118524193252, 0.722611422714007], [1.0, 0.9655723542771351, 0.9804937080323111, 0.9810240840378684]]}
total_data["train_stat"] =  {'NN':[[0, 0, 0.300, 0.366], [1, 1, 0.998, 0.995]], 'DT': [[0.9971301659294326, 0.9754246398303108, 0.9521375765863356, 0.9486948835275514], [0.9998566624559694, 0.9978558323463432, 0.9927405240714986, 0.9877104308336149]], 'GPR': [[0.9178023480386381, 0.8862275668396355, 0.951480979265988, 0.9399497840306542], [0.9998600968902147, 0.9995655540932752, 0.9912002937011297, 0.9872473352762894]], 'NB': [[0.995480677558881, 0.9949793917515463, 0.9707557955131376, 0.9019662702447105], [0.9736125140851639, 0.8046995588131316, 0.852469950744342, 0.9011130986193466]], 'SVM': [[0.9982129904607498, 0.9735074121337886, 0.9579358040873304, 0.949786844901055], [0.9995291864928632, 0.9967187467048605, 0.9897471520873354, 0.9810904498321816]]}
# total_data["train_time"] =  {'NN':[[0, 0, 0, 0.083], [1, 1, 1, 0.997]], 'NB': [[0.9843394166140257, 0.9822411703654564, 0.9821218037625853, 0.98253900394699], [0.11524806488558798, 0.062412431581458906, 0.050018336144793694, 0.0358086582057558]]}
train_title = "train_time"
def get_percent(x_list):
    return np.array([i * 100 for i in x_list])

def draw_other(train_data):
    x = np.array([0.05, 0.1, 0.15, 0.2])
    for key in ['GPR', 'NB', 'DT', 'SVM', 'NN']:
        if key in train_data:
            plt.plot(x, get_percent(train_data[key][0]), label=key, linestyle='-', linewidth=2, marker='^', markerfacecolor='white', markersize=10)
    plt.xlabel('ele percent', fontsize=14)
    plt.ylabel('accuracy(%)', fontsize=14)
    y_major_locator=MultipleLocator(10)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-5,105)
    # plt.ylim(0, 100)
    #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.grid(axis='y', linestyle='-.')
    plt.title(train_title)
    plt.legend(loc=3) 
    plt.show()

def draw_other_mice(train_data):
    x = np.array([0.05, 0.1, 0.15, 0.2])
    for key in ['GPR', 'NB', 'DT', 'SVM', 'NN']:
        if key in train_data:
            plt.plot(x, get_percent(train_data[key][1]), label=key, linestyle='-', linewidth=2, marker='^',markerfacecolor='white', markersize=10)
    plt.xlabel('mice percent', fontsize=14)
    plt.ylabel('accuracy(%)', fontsize=14)

    y_major_locator=MultipleLocator(10)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-5,105)
    # plt.ylim(0, 100)
    #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.grid(axis='y', linestyle='-.')
    plt.title(train_title)
    plt.legend(loc=3) 
    plt.show()

def draw_same_cmp(train_type, ele_mice):
    x = np.array([0.05, 0.1, 0.15, 0.2])
    for key in total_data:
        print(total_data[key][train_type][0])
        plt.plot(x, get_percent(total_data[key][train_type][ele_mice]), label=key, linestyle='-', linewidth=2, marker='^',markerfacecolor='white', markersize=10)
    if ele_mice == 0:
        plt.xlabel('ele percent', fontsize=14)
    else:
        plt.xlabel('mice percent', fontsize=14)
    plt.ylabel('accuracy(%)', fontsize=14)

    y_major_locator=MultipleLocator(10)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-5,105)
    # plt.ylim(0, 100)
    #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.grid(axis='y', linestyle='-.')
    plt.title(train_type)
    plt.legend(loc=3) 
    plt.show()


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

def main():
    # train_data = total_data[train_title]
    # draw_other(train_data)
    # draw_other_mice(train_data)

    draw_same_cmp("SVM", 0)
    draw_same_cmp("SVM", 1)


if __name__ == "__main__":
    main()
