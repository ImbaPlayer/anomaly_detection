from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
from datetime import datetime



def main(num):
    fileName = "/data/sym/one-class-svm/data/5-tuple/dec-feature/caida-A-50W-{}.csv".format(1)
    saveName = "/data/sym/one-class-svm/data/5-tuple/bin-feature/caida-A-50W-{}.csv".format(1)
    # fileName = "1.csv"
    # saveName = "2.csv"
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

    # first packet length
    # First_cols = ['First-%d'%i for i in range(16)]
    # dfb[First_cols] = df['length'].apply(getBits(16))

    print(dfb.shape)
    dfb.to_csv(saveName, index=False)
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

    