# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-16 19:09:10
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-25 18:48:11
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
    fileName = "/data/sym/anomaly_detection/data/10-fold/{}/dec-stat/{}-{}.csv".format(data_type, data_type, num)
    saveName_5 = "/data/sym/anomaly_detection/data/10-fold/{}/bin-5/{}-{}.csv".format(data_type, data_type, num)
    saveName_stat = "/data/sym/anomaly_detection/data/10-fold/{}/bin-stat/{}-{}.csv".format(data_type, data_type, num)
    # fileName = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/univ1-50W-{0}-{1}.csv".format(5, num)
    # saveName = "/data/sym/one-class-svm/data/mean_of_five/bin-feature/univ1-50W-{0}-{1}.csv".format(5, num)
    # fileName = "size-1.csv"
    # saveName = "size-2.csv"
    df = pd.read_csv(fileName)
    # print(df)
    df["srcAddr1"], df["srcAddr2"], df["srcAddr3"], df["srcAddr4"] = df["srcIP"].str.split(".", 3).str
    df["dstAddr1"], df["dstAddr2"], df["dstAddr3"], df["dstAddr4"] = df["dstIP"].str.split(".", 3).str
    df = df.drop(["srcIP", "dstIP"], axis=1)
    # print(df)
    df = df.reset_index()
    df = df.drop("index", axis=1)
    # print(df)

    #bit: number of bits, n: number to transfer
    getBits = lambda bits: lambda n: pd.Series(list(('{0:0%db}'%bits).format(int(n))))
    protocolMap = {p:i for i,p in enumerate(df["protocol"].unique())}
    print(protocolMap)
    getProtoBits = lambda p: pd.Series(list(('{0:0%db}'%3).format(protocolMap[p])))

    #create new dataframe to record features in binary
    dfb = pd.DataFrame()

    #srcPort cols
    SP_cols = ['SP%d' % i for i in range(16)]
    dfb[SP_cols] = df['srcPort'].apply(getBits(16))

    #dstPort cols
    DP_cols = ['DP%d' % i for i in range(16)]
    dfb[DP_cols] = df['dstPort'].apply(getBits(16))

    for i in range(4):
        #source address cols
        SA_cols = ['SA%d' % (i*8 + j) for j in range(8)]
        dfb[SA_cols] = df['srcAddr%d' % (i + 1)].apply(getBits(8))
    for i in range(4):
        #source address cols
        DA_cols = ['DA%d' % (i*8 + j) for j in range(8)]
        dfb[DA_cols] = df['dstAddr%d' % (i + 1)].apply(getBits(8))

    
    #portocol cols
    Proto_cols = ['Proto-%d'%i for i in range(3)]
    dfb[Proto_cols] = df['protocol'].apply(getProtoBits)

    # save 5-tuple to file
    dfb.to_csv(saveName_5, index=False)

    dfb_stat = pd.DataFrame()
    statistic_names = ["mean", "var", "min", "max"]
    for col_name in statistic_names:
        temp_cols = [col_name + '-{}'.format(i) for i in range(32)]
        dfb_stat[temp_cols] = df[col_name].apply(getBits(32))
    print(dfb_stat.shape)
    # save stat to file
    dfb_stat.to_csv(saveName_stat, index=False)
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

    