import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

fileName1 = "data/dec-feature/caida-A-50W-1.csv"
# fileName1 = "dec-feature-univ1.csv"
thres = 200
def get_threshold(yr):
    print("mean", yr.mean())
    print("max", yr.max())
    print("min", yr.min())
    # plot histogram
    # plt.subplot(221)
    # plt.hist(yr.values,cumulative=True)
    # plt.show()
def show_count(yr):
    yc = yr.copy(deep=True)
    yc[yr > thres] = 1
    yc[yr <= thres ] = -1
    print("original mice count: ", sum(yc==-1), sum(yc==-1)/ yc.shape[0])
    print("original elephant count: ", sum(yc==1), sum(yc==1) / yc.shape[0])
if __name__ == "__main__":
    df = pd.read_csv(fileName1)
    yr = df['flowSize']
    get_threshold(yr)
    show_count(yr)