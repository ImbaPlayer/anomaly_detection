# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-12-07 18:21:47
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-12-07 18:54:21
import numpy as np
import ast
import sys
import os

def get_list():
    filename = "./compare_result.log"
    result = {}
    with open(filename) as file_object:
        for line in file_object:
            if "[" in line:
                param_list = line.replace("[", "").replace("]", "").replace(",","").replace("\'", "").split()
                train_type = param_list[0]
                clf_type = param_list[1]
            if "ele" in line:
                if not train_type in result:
                    result[train_type] = {}
                if clf_type in result[train_type]:
                    result[train_type][clf_type][0].append(float(line.split()[-1]))
                else:
                    result[train_type][clf_type] = [[],[]]
                    result[train_type][clf_type][0].append(float(line.split()[-1]))
            elif "mice" in line:
                if not train_type in result:
                    result[train_type] = {}
                if clf_type in result[train_type]:
                    result[train_type][clf_type][1].append(float(line.split()[-1]))
                else:
                    result[train_type][clf_type] = [[],[]]
                    result[train_type][clf_type][1].append(float(line.split()[-1]))
    # print(result)
    for key in result:
        print("train_{} = ".format(key),result[key]) 
                

def main():
    get_list()


if __name__ == "__main__":
    main()
