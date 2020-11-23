# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-14 16:49:34
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-16 16:56:48

import numpy as np
import pandas as pd
from collections import Counter

def get_avg_report(report_list):
    # df = pd.DataFrame(report_list)
    report_array = np.array(report_list)
    np.save('a.npy', report_array)
    
    report_list_0 = []
    report_list_1 = []
    acc_list = []
    result = {}
    for data in report_list:
        report_list_0.append(data['0'])
        report_list_1.append(data['1'])
        acc_list.append(data['accuracy'])
    df_0 = pd.DataFrame(report_list_0)
    df_1 = pd.DataFrame(report_list_1)
    result['0'] = dict(df_0.mean())
    result['1'] = dict(df_1.mean())
    result["accuracy"] = np.mean(acc_list)

def main():
    a = np.load('a.npy',allow_pickle=True)
    a = a.tolist()
    report_list_0 = []
    report_list_1 = []
    acc_list = []
    result = {}
    for data in a:
        report_list_0.append(data['0'])
        report_list_1.append(data['1'])
        acc_list.append(data['accuracy'])
    df_0 = pd.DataFrame(report_list_0)
    df_1 = pd.DataFrame(report_list_1)
    result['0'] = dict(df_0.mean())
    result['1'] = dict(df_1.mean())
    result["accuracy"] = np.mean(acc_list)

    print(result)

def new_test(type):
    if type == 0:
        a = 1
        b = 2
    elif type == 1:
        a = 3
        b = 4
    print("a", a)
    print("b", b)

if __name__ == "__main__":
    # main()
    new_test(0)
