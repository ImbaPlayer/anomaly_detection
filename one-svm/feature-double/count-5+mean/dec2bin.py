from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
from datetime import datetime
import re

def split_str(text, length):
    text_arr = re.findall(r'.{%d}' % int(length), text)
    result = [int(data, 2) for data in text_arr]
    return result

def main(num):
    fileName = "/data/sym/one-class-svm/data/count/dec-feature/caida-A-50W-{}.csv".format(0)
    saveName = "/data/sym/one-class-svm/data/count/bin-feature/caida-A-50W-{}.csv".format(10)
    df = pd.read_csv(fileName)
    # print(df)
    df["srcAddr1"], df["srcAddr2"], df["srcAddr3"], df["srcAddr4"] = df["srcIP"].str.split(".", 3).str
    df["dstAddr1"], df["dstAddr2"], df["dstAddr3"], df["dstAddr4"] = df["dstIP"].str.split(".", 3).str
    df = df.drop(["srcIP", "dstIP"], axis=1)
    # print(df)
    df = df.reset_index()
    df = df.drop("index", axis=1)
    # print(df)
    protocolMap = {p:i for i,p in enumerate(df["protocol"].unique())}
    print(protocolMap)
    df["protocol"] = df["protocol"].apply(lambda p: protocolMap[p])

    #bit: number of bits, n: number to transfer
    protocolMap = {p:i for i,p in enumerate(df["protocol"].unique())}
    print(protocolMap)
    getProtoBits = lambda p: pd.Series(list(('{0:0%db}'%3).format(protocolMap[p])))

    getBytes = lambda bits: lambda n: pd.Series(split_str(('{0:0%db}'%bits).format(int(n)), 8))

    #srcPort cols
    SP_cols = ['SP%d' % i for i in range(2)]
    df[SP_cols] = df['srcPort'].apply(getBytes(16))

    #dstPort cols
    DP_cols = ['DP%d' % i for i in range(2)]
    df[DP_cols] = df['dstPort'].apply(getBytes(16))

    # statistical data cols
    statistic_names = ["first", "second", "third", "fourth", "fifth",
                        "mean", "var", "min", "max"]
    for col_name in statistic_names:
        temp_cols = [col_name + '-{}'.format(i) for i in range(4)]
        df[temp_cols] = df[col_name].apply(getBytes(32))
    df = df.drop(["srcPort", "dstPort", "first", "second", "third", "fourth", "fifth",
                        "mean", "var", "min", "max", "flowSize"], axis=1)

    print(df.shape)
    df.to_csv(saveName, index=False)
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)
    for i in range(1):
        main(i)
        print("finish", i)
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)

    