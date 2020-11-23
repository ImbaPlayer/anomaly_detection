# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-15 18:10:09
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-15 19:18:33

import numpy as np

def process_svm():
    filename = "temp.log"
    nn_mice = []
    nn_ele = []
    with open(filename) as file_object:
        for line in file_object:
            word_list = line.split()
            if len(word_list) < 1:
                continue
            if word_list[0] == '-1':
                nn_ele.append(float(word_list[2]))
            if word_list[0] == '1':
                nn_mice.append(float(word_list[2]))
    print(nn_mice)
    print(nn_ele)

    print(np.mean(nn_mice))
    print(np.mean(nn_ele))
def process_auto():
    filename = "temp.log"
    nn_mice = []
    nn_ele = []
    with open(filename) as file_object:
        for line in file_object:
            word_list = line.split()
            if len(word_list) < 1:
                continue
            if word_list[0] == '0':
                nn_mice.append(float(word_list[2]))
            if word_list[0] == '1':
                nn_ele.append(float(word_list[2]))
    print(nn_mice)
    print(nn_ele)

    print(np.mean(nn_mice))
    print(np.mean(nn_ele))

def main():
    filename = "temp.log"
    nn_mice = []
    nn_ele = []
    random_mice = []
    random_ele = []
    index_1 = 1
    index_2 = 1
    with open(filename) as file_object:
        for line in file_object:
            word_list = line.split()
            if len(word_list) < 1:
                continue
            if word_list[0] == '-1':
                if index_1 % 2 == 1:
                    nn_mice.append(float(word_list[2]))
                else:
                    random_mice.append(float(word_list[2]))
                index_1 += 1
            if word_list[0] == '1':
                if index_2 % 2 == 1:
                    nn_ele.append(float(word_list[2]))
                else:
                    random_ele.append(float(word_list[2]))
                index_2 += 1
    print(nn_mice)
    print(nn_ele)
    print(random_mice)
    print(random_ele)

    print(np.mean(nn_mice))
    print(np.mean(nn_ele))
    print(np.mean(random_mice))
    print(np.mean(random_ele))


if __name__ == "__main__":
    main()
    # process_auto()
    # process_svm()
