# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-16 19:09:10
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-25 18:45:25
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
from datetime import datetime


PACKET_NUMBER = 10
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1", "univ2"]
def main(num):
    data_type = ALL_DATA_TYPE[3]
    fileName = "/data/sym/anomaly_detection/data/10-fold/{}/dec-size/{}-{}.csv".format(data_type, data_type, num)
    saveName = "/data/sym/anomaly_detection/data/10-fold/{}/bin-size/{}-{}.csv".format(data_type, data_type, num)
    # fileName = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/univ1-50W-{0}-{1}.csv".format(5, num)
    # saveName = "/data/sym/one-class-svm/data/mean_of_five/bin-feature/univ1-50W-{0}-{1}.csv".format(5, num)
    # fileName = "size-1.csv"
    # saveName = "size-2.csv"
    df = pd.read_csv(fileName)

    #bit: number of bits, n: number to transfer
    getBits = lambda bits: lambda n: pd.Series(list(('{0:0%db}'%bits).format(int(n))))

    #create new dataframe to record features in binary
    dfb = pd.DataFrame()

    statistic_names = ["pkt-size-{}".format(i) for i in range(PACKET_NUMBER)]
    for col_name in statistic_names:
        temp_cols = [col_name + '-{}'.format(i) for i in range(32)]
        dfb[temp_cols] = df[col_name].apply(getBits(32))
    print(dfb.shape)
    dfb.to_csv(saveName, index=False)
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)
    for i in range(9):
        main(i)
        print("finish", i)
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)

    