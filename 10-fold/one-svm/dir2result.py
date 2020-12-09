# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-12-06 13:49:13
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-12-07 18:47:19
import numpy as np
import ast
import sys
import os

def test():
    # test_list = []
    # print("len", len(test_list))
    # print("mean", np.mean(test_list))
    file_path = "H:\\科研\\code\\anomaly_detection\\10-fold\\one-svm"
    file_list = os.listdir(file_path)
    print(file_list)
    print(type(file_list))
    print(sorted(file_list))

def root_other(file_path, process_func):
    # -1: mice 1: ele
    file_list = os.listdir(file_path)
    file_list = sorted(file_list)

    for dir_file in file_list:
        param_list = dir_file.replace(".log", "").split("-")
        print(param_list)
        dir_file_path = os.path.join(file_path, dir_file)
        # print(dir_file_path)
        process_func(dir_file_path)

def process_other(filename):
    print("file", filename)
    nn_mice = []
    nn_ele = []
    nn_acc = []
    mice_count = []
    ele_count = []
    with open(filename) as file_object:
        for line in file_object:
            if "final" in line:
                line = line.replace('{', '')
                line = line.replace('}', '')
                line = line.replace(':', '').replace('\'', '').replace(',', '')
                word_list = line.split()
                # print(word_list)
                for i in range(len(word_list)):
                    if word_list[i] == "-1":
                        nn_mice.append(float(word_list[i+4]))
                    if word_list[i] == "1":
                        nn_ele.append(float(word_list[i+4]))
                    if word_list[i] == "accuracy":
                        nn_acc.append(float(word_list[i+1]))
            elif "original mice count" in line:
                word_list = line.split()
                mice_count.append(float(word_list[3]))
            elif "original elephant count" in line:
                word_list = line.split()
                ele_count.append(float(word_list[3]))
    if len(nn_ele) > 0:
        print("ele recall", np.mean(nn_ele))
        print("mice recall", np.mean(nn_mice))
        print("acc", np.mean(nn_acc))
        print("mean flow count ", np.mean(mice_count) + np.mean(ele_count))
    else:
        print("empty error")
    print()

def process_torch(filename):
    print("file", filename)
    nn_mice = []
    nn_ele = []
    nn_acc = []
    mice_count = []
    ele_count = []
    with open(filename) as file_object:
        for line in file_object:
            if "final" in line:
                line = line.replace('{', '')
                line = line.replace('}', '')
                line = line.replace(':', '').replace('\'', '').replace(',', '')
                word_list = line.split()
                # print(word_list)
                if word_list[2] == "nn":
                    for i in range(len(word_list)):
                        if word_list[i] == "-1":
                            nn_mice.append(float(word_list[i+4]))
                        if word_list[i] == "1":
                            nn_ele.append(float(word_list[i+4]))
                        if word_list[i] == "accuracy":
                            nn_acc.append(float(word_list[i+1]))
            elif "original mice count" in line:
                word_list = line.split()
                mice_count.append(float(word_list[3]))
            elif "original elephant count" in line:
                word_list = line.split()
                ele_count.append(float(word_list[3]))

    print("nn ele recall", np.mean(nn_ele))
    print("nn mice recall", np.mean(nn_mice))
    print("nn acc", np.mean(nn_acc))
    print("mean flow count ", np.mean(mice_count) + np.mean(ele_count))
    print()

def main():
    file_path = "/data/dgl/anomaly_detection/10-fold/compare/GPR/log/unibs"
    # file_path = "/data/dgl/anomaly_detection/10-fold/compare/NN/torch_log/unibs/weight"
    # process_func = process_torch
    process_func = process_other
    root_other(file_path, process_func)


if __name__ == "__main__":
    # test()
    main()
    