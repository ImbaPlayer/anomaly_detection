# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-16 19:00:38
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-18 19:36:02
import pandas as pd
from datetime import datetime
import numpy as np

# inputName = "/data/xgr/sketch_data/equinix-nyc.dirB.20190117-140000.UTC.anon.pcap"

PACKET_NUMBER = 10
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1"]

def my_process(x):
    print(x)
    return x.var(ddof=0)

def new_extract(num):
    data_type = ALL_DATA_TYPE[0]
    inputName = "/data/sym/anomaly_detection/data/10-fold/{}/packet-level/{}-{}.csv".format(data_type, data_type, num)
    saveName = "/data/sym/anomaly_detection/data/10-fold/{}/dec-stat/{}-{}.csv".format(data_type, data_type, num)
    #指定分隔符为/t
    # time srcIP srcPort dstIP dstPort protocol length
    col_names = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", "length"]
    df = pd.read_csv(inputName, delimiter="|", names=col_names)
    # print(df)
    # print(df)
    mask = (df["protocol"]=="TCP") | (df["protocol"]=="IPv4") | (df["protocol"]=="UDP")
    tcp = df[mask]
    tcp = tcp.drop(["time"], axis=1)
    grouped = tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])

    tcp = grouped.head(PACKET_NUMBER)
    grouped = tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])


    tcp["mean"] = grouped["length"].transform('mean')
    tcp["var"] = grouped["length"].transform(lambda x: x.var(ddof=0))
    # tcp["var"] = grouped["length"].transform(lambda x: my_process(x))
    # tcp["var"] = grouped["length"].transform('var')
    tcp["max"] = grouped["length"].transform(max)
    tcp["min"] = grouped["length"].transform(min)

    tcp["flowSize"] = grouped["length"].transform(sum)
    tcp = tcp.drop(["length"], axis=1)

    # get first packet from every group
    grouped = tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])
    new_df = grouped.head(1)
    
    # print(tcp["var"])
    
    print(new_df.shape)
    new_df.to_csv(saveName, index=False)
    #df = pd.read_csv(saveName)
    #print(df)
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)
    for i in range(1):
        new_extract(i)
        print("finish", i)
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)