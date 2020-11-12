from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
from datetime import datetime



def main(num):
    fileName = "/data/sym/one-class-svm/data/online/dec-feature/caida-A-50W-{}.csv".format(num)
    saveName = "/data/sym/one-class-svm/data/online/bin-feature/caida-A-50W-{}.csv".format(num)
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
    #Packet size cols
    # Pkt1_cols = ['Pkt1-%d' % i for i in range(33)]
    # dfb[Pkt1_cols] = df['first'].apply(getBits(33))
    # Pkt2_cols = ['Pkt2-%d' % i for i in range(33)]
    # dfb[Pkt2_cols] = df['second'].apply(getBits(33))
    # Pkt3_cols = ['Pkt3-%d' % i for i in range(33)]
    # dfb[Pkt3_cols] = df['third'].apply(getBits(33))
    # Pkt4_cols = ['Pkt4-%d' % i for i in range(33)]
    # dfb[Pkt4_cols] = df['fourth'].apply(getBits(33))
    # Pkt4_cols = ['Pkt5-%d' % i for i in range(33)]
    # dfb[Pkt4_cols] = df['fifth'].apply(getBits(33))
    # statistical data cols
    
    # ip_ihl
    temp_cols = ["ip_ihl" + '-{}'.format(i) for i in range(4)]
    dfb[temp_cols] = df["ip_ihl"].apply(getBits(4))

    # ip_tos
    temp_cols = ["ip_tos" + '-{}'.format(i) for i in range(8)]
    dfb[temp_cols] = df["ip_tos"].apply(getBits(8))

    #ip_flags
    temp_cols = ["ip_flags" + '-{}'.format(i) for i in range(3)]
    dfb[temp_cols] = df["ip_flags"].apply(getBits(3))

    # ip_ttl
    temp_cols = ["ip_ttl" + '-{}'.format(i) for i in range(8)]
    dfb[temp_cols] = df["ip_ttl"].apply(getBits(8))

    # tcp_dataofs
    temp_cols = ["tcp_dataofs" + '-{}'.format(i) for i in range(4)]
    dfb[temp_cols] = df["tcp_dataofs"].apply(getBits(4))

    # tcp_flag
    # temp_cols = ["tcp_flag" + '-{}'.format(i) for i in range(5)]
    # dfb[temp_cols] = df["tcp_flag"].apply(getBits(5))

    # tcp_window
    temp_cols = ["tcp_window" + '-{}'.format(i) for i in range(16)]
    dfb[temp_cols] = df["tcp_window"].apply(getBits(16))

    # udp_len
    temp_cols = ["udp_len" + '-{}'.format(i) for i in range(16)]
    dfb[temp_cols] = df["udp_len"].apply(getBits(16))

    print(dfb.shape)
    dfb.to_csv(saveName, index=False)
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)
    for i in range(11,12):
        main(i)
        print("finish", i)
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)

    