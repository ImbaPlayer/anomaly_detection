# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-16 19:00:38
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-30 19:02:22
import pandas as pd
from datetime import datetime
import numpy as np

# inputName = "/data/xgr/sketch_data/equinix-nyc.dirB.20190117-140000.UTC.anon.pcap"

PACKET_NUMBER = 10
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1", "univ2", "unibs"]

def get_head_packets(packets):
    result = packets.tolist()
    result.extend([0 for i in range(PACKET_NUMBER - len(result))])
    return pd.DataFrame(np.zeros((len(packets), 1)))

def get_exact_length(index, packets):
    result = packets.tolist()
    result.extend([0 for i in range(PACKET_NUMBER - len(result))])
    return result[index]

def new_extract(num):
    data_type = ALL_DATA_TYPE[4]
    inputName = "/data/dgl/anomaly_detection/data/10-fold/{}/packet-level/{}-{}.csv".format(data_type, data_type, num)
    saveName = "/data/dgl/anomaly_detection/data/10-fold/{}/dec-size/{}-{}.csv".format(data_type, data_type, num)

    #指定分隔符为/t
    # time srcIP srcPort dstIP dstPort protocol length
    col_names = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", "length"]
    df = pd.read_csv(inputName, delimiter="|", names=col_names)
    # print(df)
    # print(df)
    mask = (df["protocol"]=="TCP") | (df["protocol"]=="IPv4") | (df["protocol"]=="UDP")
    tcp = df[mask]
    tcp = tcp.drop(["time"], axis=1)
    grouped=tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])
    # get flow size
    tcp["flowSize"] = grouped["length"].transform(sum)

    for i in range(PACKET_NUMBER):
        tcp["pkt-size-{}".format(i)] = grouped["length"].transform(lambda x: get_exact_length(i, x))
    
    tcp = tcp.drop(["length"], axis=1)
    grouped = tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])
    new_df = grouped.head(1)
    # delete 5-tuple
    new_df.drop(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"], axis=1)
    print(new_df.shape)
    new_df.to_csv(saveName, index=False)
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)
    for i in range(3):
        new_extract(i)
        print("finish", i)
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)