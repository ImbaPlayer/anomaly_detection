import numpy as np
import ast
import sys
import os

all_train_type = ["5-tuple", "time", "size", "stat"]
ele_type = [0.05, 0.1, 0.15, 0.2]

def root_other(file_path, process_func):
    # -1: mice 1: ele
    file_list = os.listdir(file_path)
    file_list = sorted(file_list)

    result = {}

    for dir_file in file_list:
        # sys params: ele_percent train_type clf_type
        param_list = dir_file.replace(".log", "").split("-")
        ele_per = param_list[0]
        train_type = all_train_type[int(param_list[1])]
        clf_type = param_list[2]
        print(param_list)
        dir_file_path = os.path.join(file_path, dir_file)
        # print(dir_file_path)
        temp_ele, temp_mice, temp_acc = process_func(dir_file_path)
        if not clf_type in result:
            result[clf_type] = {}
        if not train_type in result[clf_type]:
            result[clf_type][train_type] = {"ele_recall":[-1, -1, -1, -1], "mice_recall":[-1, -1, -1, -1], "acc":[-1, -1, -1, -1]}
            #[-1, -1, -1, -1]
        result[clf_type][train_type]["ele_recall"][ele_type.index(float(ele_per))] = temp_ele
        result[clf_type][train_type]["mice_recall"][ele_type.index(float(ele_per))] = temp_mice
        result[clf_type][train_type]["acc"][ele_type.index(float(ele_per))] = temp_acc
    print(result)
    print()
    for clf_type in result:
        print("clf_type", clf_type)
        for train_type in result[clf_type]:
            print("train_type", train_type)
            print(result[clf_type][train_type])
        


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
        return nn_ele[0], nn_mice[0], nn_acc[0]
    else:
        return "error", "error", "error"

if __name__ == "__main__":
    file_path = "/data/sym/anomaly_detection/10-fold/valid-caida/log/10W"
    process_func = process_other
    root_other(file_path, process_other)