# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-14 21:57:59
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-25 00:31:35
import pandas as pd
from datetime import datetime

# inputName = "/data/xgr/sketch_data/equinix-nyc.dirB.20190117-140000.UTC.anon.pcap"

PACKET_NUMBER = 10
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1"]
def new_extract(num):
    data_type = ALL_DATA_TYPE[2]
    # inputName = "/data/sym/one-class-svm/data/mean_of_five/packet-level/caida-B-50W-{}.csv".format(num)
    inputName = "/data/sym/anomaly_detection/data/10-fold/{}/packet-level/{}-{}.csv".format(data_type, data_type, num)
    saveName = "/data/sym/anomaly_detection/data/10-fold/{}/dec-time/{}-{}.csv".format(data_type, data_type, num)

    # inputName = "/data/sym/one-class-svm/data/mean_of_five/packet-level/univ1-50W-{}.csv".format(num)
    # saveName = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/univ1-50W-{0}-{1}.csv".format(PACKET_NUMBER, num)
    #指定分隔符为/t
    # time srcIP srcPort dstIP dstPort protocol length
    col_names = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", "length"]
    df = pd.read_csv(inputName, delimiter="|", names=col_names)
    # print(df)
    # print(df)
    mask = (df["protocol"]=="TCP") | (df["protocol"]=="IPv4") | (df["protocol"]=="UDP")
    tcp = df[mask]
    # tcp = tcp.drop(["time"], axis=1)
    grouped=tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])

    # get flowsize
    tcp["flowSize"] = grouped["length"].transform(sum)
    tcp = tcp.drop(["length"], axis=1)

    for i in range(PACKET_NUMBER - 1):
        tcp["interval-{}".format(i)] = grouped["time"].transform(lambda x: get_time_interval(i, x))
    grouped = tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])
    new_df = grouped.head(1)
    # delete 5-tuple
    new_df.drop(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"], axis=1)
    print(new_df.shape)
    new_df.to_csv(saveName, index=False)
 

def get_time_interval(index, packets):
    arrive_time = packets.tolist()
    if index >= len(arrive_time) - 1:
        return 0
    else:
        return arrive_time[index + 1] - arrive_time[index]

def get_interval(arrive_time):
    time_interval = []
    for i in range(len(arrive_time) - 1):
        time_interval.append(arrive_time[i+1] - arrive_time[i])
    for _ in range(PACKET_NUMBER - 1 - len(time_interval)):
        time_interval.append(0)
    return time_interval
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)
    for i in range(1,21):
        new_extract(i)
        print("finish", i)
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)