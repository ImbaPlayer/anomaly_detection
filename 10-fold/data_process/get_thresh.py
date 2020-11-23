# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-16 17:17:55
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-16 18:52:49

import pandas as pd
import numpy as np
import sys

def main():
    fileName = "../autoencoder/data/dec-test.csv"
    df = pd.read_csv(fileName)
    flowSize = df["flowSize"]
    print(type(flowSize))
    np_flowSize = np.array(flowSize)
    print(type(flowSize))
    # np_flowSize = flowSize.values()
    quantile = 0.9
    thres = np.quantile(np_flowSize, quantile)
    print("thresh", thres)
    yc = flowSize.copy(deep=True)
    yc[flowSize <= thres] = 0
    yc[flowSize > thres ] = 1
    print("original mice count: ", sum(yc==0))
    print("original elephant count: ", sum(yc==1))
    print("total count: ", yc.shape[0])
    print("ele percent",  sum(yc==1) / yc.shape[0])
    print(np.sort(np_flowSize))
    print(float("0.01"))
def get_thres(flowSize, elePercent):
    # param flowSize is DataFrame
    np_flowSize = np.array(flowSize)
    quantile = 1 - elePercent
    thres = np.quantile(np_flowSize, quantile)
    return thres

if __name__ == "__main__":
    main()
