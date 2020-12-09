# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-30 17:22:08
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-30 17:22:08
import numpy as np
import ast
import sys

def process_svm():
    filename = sys.argv[1]
    print("file", filename)
    svm_mice = []
    svm_ele = []
    svm_acc = []
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
                        svm_ele.append(float(word_list[i+4]))
                    if word_list[i] == "1":
                        svm_mice.append(float(word_list[i+4]))
                    if word_list[i] == "accuracy":
                        svm_acc.append(float(word_list[i+1]))
            elif "original mice count" in line:
                word_list = line.split()
                mice_count.append(float(word_list[3]))
            elif "original elephant count" in line:
                word_list = line.split()
                ele_count.append(float(word_list[3]))
    print(svm_ele)
    print(svm_mice)
    print(svm_acc)
    print("ele recall", np.mean(svm_ele))
    print("mice recall", np.mean(svm_mice))
    print("acc", np.mean(svm_acc))
    print("mean flow count ", np.mean(mice_count) + np.mean(ele_count))

def process_nn():
    filename = sys.argv[1]
    print("file", filename)
    nn_mice = []
    nn_ele = []
    nn_acc = []
    rf_mice = []
    rf_ele = []
    rf_acc = []
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
                            nn_ele.append(float(word_list[i+4]))
                        if word_list[i] == "1":
                            nn_mice.append(float(word_list[i+4]))
                        if word_list[i] == "accuracy":
                            nn_acc.append(float(word_list[i+1]))
                else:
                    for i in range(len(word_list)):
                        if word_list[i] == "-1":
                            rf_ele.append(float(word_list[i+4]))
                        if word_list[i] == "1":
                            rf_mice.append(float(word_list[i+4]))
                        if word_list[i] == "accuracy":
                            rf_acc.append(float(word_list[i+1]))
            elif "original mice count" in line:
                word_list = line.split()
                mice_count.append(float(word_list[3]))
            elif "original elephant count" in line:
                word_list = line.split()
                ele_count.append(float(word_list[3]))

    print("nn ele recall", np.mean(nn_ele))
    print("nn mice recall", np.mean(nn_mice))
    print("nn acc", np.mean(nn_acc))
    print()
    print("rf ele recall", np.mean(rf_ele))
    print("rf mice recall", np.mean(rf_mice))
    print("rf acc", np.mean(rf_acc))
    print("mean flow count ", np.mean(mice_count) + np.mean(ele_count))

def process_torch():
    filename = sys.argv[1]
    print("file", filename)
    nn_mice = []
    nn_ele = []
    nn_acc = []
    rf_mice = []
    rf_ele = []
    rf_acc = []
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

def test():
    test_str = "final report {'-1': {'precision': 0.425200450959228, 'recall': 0.6722753914782851, 'f1-score': 0.5206985430599612, 'support': 631.7}, '1': {'precision': 0.9610839876329204, 'recall': 0.8990071709755683, 'f1-score': 0.9289987544752181, 'support': 5685.5}, 'accuracy': 0.8763376615345727}"
    test_str = test_str.replace('{', '')
    test_str = test_str.replace('}', '')
    test_str = test_str.replace(':', '').replace('\'', '').replace(',', '')
    word_list = test_str.split()
    print(word_list)
    svm_mice = []
    svm_ele = []
    svm_acc = []
    for data in word_list:
        print(data)
    for i in range(len(word_list)):
        if word_list[i] == "-1":
            svm_ele.append(float(word_list[i+4]))
        if word_list[i] == "1":
            svm_mice.append(float(word_list[i+4]))
        if word_list[i] == "accuracy":
            svm_acc.append(float(word_list[i+1]))
    print(svm_mice)
    print(svm_ele)
    print(svm_acc)
    print("ele recall", np.mean(svm_ele))
    print("mice recall", np.mean(svm_mice))
    print("acc", np.mean(svm_acc))
if __name__ == "__main__":
    # process_svm()
    # process_nn()
    process_torch()